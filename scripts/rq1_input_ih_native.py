"""Input-stage physical retain-one. No native source or historical runner edits.

Slice prepared embeddings BEFORE block 1, preserve original gapped positions,
and let the native decoder construct its own shortened causal mask. Temporary
RoPE hooks only extend old native lookup tables to the original position extent.
"""

from contextlib import contextmanager
import time

from rq1_input_ih import exact_score, kept_positions, require, verify_reuse_identity
from rq1_layer_ih import probability
from rq1_native import (max_score_delta, native_parity, prepare_question, score_embeddings,
                        score_from_logits, synchronize, tensor_sha256, zero_patch)


@contextmanager
def rotary_extent(model, original_length, shortened_length):
    handles, seen = [], []
    def before(index):
        def hook(module, args, kwargs):
            if kwargs.get('seq_len') != shortened_length:
                raise RuntimeError('Unexpected native input-retention RoPE/cache length')
            changed=dict(kwargs);changed['seq_len']=original_length;seen.append(index)
            return args,changed
        return hook
    try:
        for index,block in enumerate(model.model.layers):
            handles.append(block.self_attn.rotary_emb.register_forward_pre_hook(before(index),with_kwargs=True))
        yield
        require(seen == list(range(len(model.model.layers))), 'Native rotary lookup did not visit every block once')
    finally:
        for handle in reversed(handles):handle.remove()


def score_retained_input(model, prepared, retained):
    import torch
    require(not model.training and not model.model.layer_list and model.visual_token_num == 576
            and model.model._use_sdpa and getattr(model.config,'rope_scaling',None) is None
            and bool(prepared['attention_mask'].all()), 'Requires uncompressed eval, native SDPA, unscaled RoPE and unpadded input')
    length=prepared['embeddings'].shape[1]
    require(torch.equal(prepared['position_ids'][0],torch.arange(length,device=prepared['position_ids'].device)),
            'Input reference position IDs must be contiguous before retention')
    keep=kept_positions(length,prepared['image_start'],retained)
    select=lambda value,dim=1:value.index_select(dim,torch.tensor(keep,dtype=torch.long,device=value.device))
    embeddings=select(prepared['embeddings']);positions=select(prepared['position_ids'])
    labels=select(prepared['labels']);mask=select(prepared['attention_mask'])
    require(labels[labels != -100].tolist() == prepared['labels'][prepared['labels'] != -100].tolist(),
            'Retention changed frozen target IDs')
    with rotary_extent(model,length,len(keep)),torch.inference_mode():
        output=model(inputs_embeds=embeddings,position_ids=positions,attention_mask=mask,use_cache=False,return_dict=True)
        require(output.logits.shape[1] == len(keep), 'Native logits were not physically shortened')
        score=score_from_logits(output.logits,labels)
    return score,{'kept_positions':keep,'shortened_length':len(keep)}


def measure_input_ih(model,tokenizer,case,encoded,archived,feature,order,progress,calibration=None):
    """Only new IH scores + three zeroing audit repeats; original scores are reused."""
    prepared=prepare_question(tokenizer,model,case,encoded)
    verify_reuse_identity(encoded,prepared,archived,feature)
    initial_hash=tensor_sha256(prepared['embeddings'])
    controls={'passed':False,'tolerance_per_answer_token':.001,'archived_reuse_exact':False,
              'archived_reference':archived['controls']['reference']}
    record={'status':'running','stage':'input','boundary':'before_block_1','layer':0,'window_size':1,
        'operator':'physical-retain-one-original-positions','target_token_ids':prepared['target_token_ids'],
        'first_reference_token_string':tokenizer.convert_ids_to_tokens(prepared['target_token_ids'][0]),
        'token_execution_order':order,'expanded_sequence_length':prepared['embeddings'].shape[1],
        'image_expanded_start':prepared['image_start'],'target_expanded_start':prepared['target_start'],
        'input_embeddings_sha256':initial_hash,'controls':controls,'tokens':[],
        'score_definition':'p(first frozen gold tokenizer piece | retain one) - p(first piece | remove all visual inputs)',
        'context':'Retain all nonvisual input embeddings and original positions. No decoder block has processed the image yet; null retains the image-placeholder positional gap and is not the ordinary text-only prompt.'}
    started=time.perf_counter();progress(record)
    controls['reference']=score_embeddings(model,prepared)
    controls['reference_repeat']=score_embeddings(model,prepared)
    controls['native_multimodal']=native_parity(model,prepared,encoded)
    controls['clone_noop']=score_embeddings(model,prepared,prepared['embeddings'].clone())
    for key in ['reference','reference_repeat','native_multimodal','clone_noop']:
        exact_score(controls[key],archived['controls']['reference'],key)
    old={t['token_index']:t for t in archived['tokens']}
    controls['zero_audit_repeats']=[]
    for i in archived['token_execution_order'][:3]:
        score=score_embeddings(model,prepared,zero_patch(prepared['embeddings'],prepared['image_start'],i))
        controls['zero_audit_repeats'].append({'token_index':i,'score':score})
        exact_score(score,old[i]['perturbed_score'],'zero audit token '+str(i))
    controls['archived_reuse_exact']=True
    controls['retain_all_noop'],_=score_retained_input(model,prepared,list(range(576)))
    controls['remove_all_before'],trace=score_retained_input(model,prepared,[])
    controls['remove_all_repeat'],_=score_retained_input(model,prepared,[])
    controls['max_reference_delta']=max(max_score_delta(controls['reference'],controls[k]) for k in
                                      ['reference_repeat','native_multimodal','clone_noop','retain_all_noop'])
    controls['remove_all_repeat_delta']=max_score_delta(controls['remove_all_before'],controls['remove_all_repeat'])
    require(max(controls['max_reference_delta'],controls['remove_all_repeat_delta']) <= .001,'Input IH initial controls failed')
    record.update(remove_all_probability=probability(controls['remove_all_before']),
                  remove_all_kept_positions=trace['kept_positions'],remove_all_shortened_length=trace['shortened_length'])
    progress(record)
    prior={t['token_index']:t for t in calibration['ih']['tokens']} if calibration else {}
    if calibration:
        exact_score(controls['remove_all_before'],calibration['ih']['controls']['remove_all_before'],'IH calibration null')
    for index in order:
        synchronize();tick=time.perf_counter()
        score,trace=score_retained_input(model,prepared,[index]);synchronize()
        item={'token_index':index,'patch_row':index//24,'patch_col':index%24,
            'reconstruction_error':old[index]['reconstruction_error'],'retained_score':score,
            'retained_probability':probability(score),'ih_first_probability':probability(score)-record['remove_all_probability'],
            'kept_positions':trace['kept_positions'],'shortened_length':trace['shortened_length'],
            'seconds':time.perf_counter()-tick}
        record['tokens'].append(item)
        if index in prior:exact_score(score,prior[index]['retained_score'],'IH calibration retained '+str(index))
        progress(record)
    controls['first_retained_repeat'],_=score_retained_input(model,prepared,[order[0]])
    controls['remove_all_after'],_=score_retained_input(model,prepared,[])
    controls['reference_after']=score_embeddings(model,prepared)
    exact_score(controls['reference_after'],archived['controls']['reference'],'final reference')
    controls['final_input_exact']=tensor_sha256(prepared['embeddings']) == initial_hash
    controls['first_retained_repeat_delta']=max_score_delta(record['tokens'][0]['retained_score'],controls['first_retained_repeat'])
    controls['remove_all_after_delta']=max_score_delta(controls['remove_all_before'],controls['remove_all_after'])
    controls['passed']=controls['archived_reuse_exact'] and controls['final_input_exact'] and max(
        controls['first_retained_repeat_delta'],controls['remove_all_after_delta']) <= .001
    record['calibration_retention_repeats_exact']=sum(i in prior for i in order)
    record['seconds']=time.perf_counter()-started
    record['status']='passed' if controls['passed'] else 'unstable_reference'
    progress(record);return record
