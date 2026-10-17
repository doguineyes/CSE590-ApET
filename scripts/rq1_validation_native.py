"""Project-only perturbations and true text-only generation; no ApET source edits."""

import math
import time

from rq1_native import (max_score_delta, measure_question, prepare_question,
                        score_embeddings, synchronize, zero_patch)
from rq1_validation import METHODS, neighbors, perturbation_summary
from smoke_original_apet import configure


def perturb_patch(embeddings, image_start, index, method, strength=1.0):
    import torch
    if method not in METHODS or not math.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError('Unknown method or invalid strength')
    adjacent = neighbors(index)
    if image_start < 0 or image_start+576 > embeddings.shape[1]:
        raise ValueError('Visual block is outside the sequence')
    changed = embeddings.clone()
    if strength == 0:
        return changed
    original = embeddings[:,image_start+index,:]
    if method == 'zero':
        replacement = torch.zeros_like(original)
    elif method == 'half':
        replacement = original*.5
    else:
        # Always read neighbors from the full, unmodified reference; no wrapping.
        replacement = embeddings[:,[image_start+i for i in adjacent],:].float().mean(dim=1).to(original.dtype)
    changed[:,image_start+index,:] = replacement if strength == 1 else original+strength*(replacement-original)
    return changed


def generate_text_only(tokenizer, model, question, seed, max_new_tokens=16):
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token
    settings=configure(model,'baseline')
    conversation=conv_templates['vicuna_v1'].copy()
    conversation.append_message(conversation.roles[0],question)
    conversation.append_message(conversation.roles[1],None)
    prompt=conversation.get_prompt()
    ids=tokenizer_image_token(prompt,tokenizer,IMAGE_TOKEN_INDEX,return_tensors='pt').unsqueeze(0)
    if (ids==IMAGE_TOKEN_INDEX).any():raise ValueError('Text-only prompt contains an image placeholder')
    limit=min(getattr(model.config,'tokenizer_model_max_length',None) or 2048,model.config.max_position_embeddings)
    if ids.shape[1]+max_new_tokens>limit:raise ValueError('Text-only prompt would exceed context limit')
    ids=ids.to(model.get_input_embeddings().weight.device)
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    synchronize();start=time.perf_counter()
    with torch.inference_mode():
        output=model.generate(ids,images=None,do_sample=False,use_cache=True,
            max_new_tokens=max_new_tokens,return_dict_in_generate=True,output_scores=True)
    synchronize()
    count=len(output.scores)
    if not 1<=count<=max_new_tokens:raise RuntimeError('Unexpected text-only generation length')
    # Native generate uses inputs_embeds and can emit a dummy BOS. Decode suffix only.
    generated=output.sequences[:,-count:]
    return {'status':'passed','settings':settings,'seed':seed,'question':question,'prompt':prompt,
        'response':tokenizer.batch_decode(generated,skip_special_tokens=True)[0].strip(),
        'generation_token_ids':generated[0].tolist(),'generated_tokens':count,
        'prompt_tokens_before_expansion':ids.shape[1],'expanded_prompt_tokens':ids.shape[1],
        'image_placeholders':0,'visual_tokens':0,'generation_seconds':time.perf_counter()-start}


def measure_alternatives(tokenizer,model,case,encoded,entry,tolerance,progress,
                         generation_tokens,free_generation):
    """Fresh zero reference first; alternative scores use exactly that context."""
    import torch
    patch_ids=[p['token_index'] for p in entry['patches']]
    before=prepare_question(tokenizer,model,case,encoded)
    if before['target_token_ids']!=entry['target_token_ids']:
        raise RuntimeError('Target token IDs differ from frozen reference; no retargeting permitted')
    del before
    record=measure_question(tokenizer,model,case,encoded,patch_ids,tolerance,progress,
        generation_tokens=generation_tokens,baseline_correct=free_generation['scoring']['correct'])
    if record['status']!='passed':raise RuntimeError('Fresh zeroing controls failed')
    if record['target_token_ids']!=entry['target_token_ids']:
        raise RuntimeError('Target token IDs differ from frozen reference; no retargeting permitted')
    prepared=prepare_question(tokenizer,model,case,encoded)
    baseline=record['controls']['reference']
    repeat=score_embeddings(model,prepared)
    shams={}
    for method in METHODS:
        sham=perturb_patch(prepared['embeddings'],prepared['image_start'],patch_ids[0],method,strength=0)
        if not torch.equal(sham,prepared['embeddings']):raise RuntimeError('Sham changed reference embeddings')
        shams[method]=score_embeddings(model,prepared,sham)
        del sham
    maximum=max(max_score_delta(baseline,s) for s in [repeat,*shams.values()])
    record['alternative_controls']={'reference_before':repeat,'shams':shams,'max_pre_delta':maximum,
        'tolerance':tolerance,'passed':maximum<=tolerance}
    record['historical_comparison']={'clip_hash_matches':encoded['clip_sha256']==entry['archived_clip_sha256'],
        'projected_hash_matches':encoded['projected_sha256']==entry['archived_projected_sha256'],
        'reference_score_delta':baseline['mean_logprob']-entry['archived_reference']['mean_logprob'],
        'note':'Historical differences are recorded, not used to change cases, targets or select favorable scores.'}
    record['alternatives']=[];record['status']='running_alternatives';progress(record)
    if maximum>tolerance:raise RuntimeError('Alternative-method reference/sham control failed')
    zero={t['token_index']:t for t in record['tokens']}
    for selected in entry['patches']:
        i=selected['token_index'];original=prepared['embeddings'][:,prepared['image_start']+i,:]
        row={**selected,'current_E':zero[i]['reconstruction_error'],
            'effects':{'zero':{'I':zero[i]['intervention_score'],'score':zero[i]['perturbed_score'],
                'delta_l2':original.float().norm().item(),'original_l2':original.float().norm().item()}},
            'neighbor_indices':neighbors(i)}
        for method in METHODS[1:]:
            synchronize();start=time.perf_counter()
            changed=perturb_patch(prepared['embeddings'],prepared['image_start'],i,method)
            delta=(changed[:,prepared['image_start']+i,:]-original).float().norm().item()
            score=score_embeddings(model,prepared,changed)
            del changed
            synchronize()
            row['effects'][method]={'I':baseline['mean_logprob']-score['mean_logprob'],'score':score,
                'delta_l2':delta,'original_l2':original.float().norm().item(),'seconds':time.perf_counter()-start}
        record['alternatives'].append(row);progress(record)
    after=score_embeddings(model,prepared)
    post=max_score_delta(baseline,after)
    record['alternative_controls'].update(reference_after=after,max_post_delta=post,passed=maximum<=tolerance and post<=tolerance)
    record['historical_comparison'].update(
        max_zero_I_delta=max(abs(r['effects']['zero']['I']-r['archived_I']) for r in record['alternatives']),
        max_E_delta=max(abs(r['current_E']-r['archived_E']) for r in record['alternatives']))
    record['comparison_by_stratum']=perturbation_summary(record['alternatives'])
    record['status']='passed' if record['alternative_controls']['passed'] else 'unstable_reference'
    progress(record)
    if record['status']!='passed':raise RuntimeError('Final reference drift failed')
    return record
