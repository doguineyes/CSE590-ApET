"""Research adapter for crossed selectors and exact native merge bypass.

Original ApET is invoked separately, unchanged. Stage1 uses CLIP space before the
projector; stage2 uses states after block16. Only adapter globals are scoped.
"""
from contextlib import contextmanager
import copy
import numpy as np
from rq1_native import replaced
from rq2_protocol import require
from rq2_native import snapshot_encoded,tensor_fingerprint
from rq2_random_native import fingerprints,pixels_fingerprint
from basis_localization_protocol import specification
from selector_grid_protocol import choose
import rq2_staged_native as staged
OLD_DECODER_SCOPE=staged.decoder_scope


def reconstruction_errors(features):
    import torch
    from llava.model.utils import fps
    idx=fps(features.unsqueeze(0),10)[0]
    b=features[idx].float();x=features.float()
    gram=b@b.t();gram.diagonal().add_(1e-5)
    weights=torch.linalg.solve(gram,b@x.t()).t()
    errors=torch.norm(x-weights@b,dim=-1)
    return errors,idx


def make_input(model,pixels,condition,seed,image_sha,full,native,raw,properties):
    import torch
    from llava.model.utils import fps
    spec=specification(condition);kind=spec['selector'];errors=properties['errors']
    snapshot=staged.clone_tree(full);basis_map=[]
    with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))),torch.inference_mode():
        staged.set_rng(full['rng_after_input'])
        if kind=='native':
            # Original native input snapshot is captured immediately before merge,
            # after its exact boolean-index basis assignment.
            selected=native['capture']['observation']['encoded']['selected_indices_sorted']
            projected=model.get_model().mm_projector(native['pre_merge'].clone())
            basis_map=native['capture']['observation']['basis_slot_to_source']
            snapshot['rng_after_input']=staged.clone_tree(native['rng_after_input'])
        elif kind=='fps':
            selected=sorted(fps(raw,96)[0].cpu().tolist())
            projected=full['encoded'][0].clone()
            snapshot['rng_after_input']=staged.rng_state()
        else:
            selected=choose(kind,96,properties['padding'],properties['detail'],errors,image_sha,spec['random_seed'],'input')
            projected=full['encoded'][0].clone()
            if kind=='error':snapshot['rng_after_input']=staged.clone_tree(native['rng_after_input'])
        require(len(selected)==len(set(selected))==96,'Input selector must retain96 unique slots')
        mask=torch.zeros_like(full['encoded'][1]);mask[:,selected]=True
        snapshot['encoded']=(projected,mask.to(projected.device))
    sources=list(selected)
    for item in basis_map:sources[selected.index(item['assignment_slot'])]=item['source_index']
    observation=dict(condition=condition,source_patches=tensor_fingerprint(raw),encoded=snapshot_encoded(snapshot['encoded']),
        selector=kind,basis_inserted=kind=='native',basis_slot_to_source=basis_map,
        selected_original_positions=selected,direct_source_original_positions=sources,
        selection_errors=np.asarray(errors).tolist() if kind in ('error','native') else None,merge=None)
    snapshot['capture']=dict(version=staged.VERSION,condition=condition,seed=seed,
        pixels=pixels_fingerprint(pixels),observation=observation,fingerprints=fingerprints(snapshot),
        rng_after_input=staged.tree_fingerprint(snapshot['rng_after_input']))
    staged.validate_input(snapshot,pixels,condition)
    return observation,snapshot


@contextmanager
def grid_scope(model,condition,image_sha,properties,selected_input,*,replay=None):
    if not condition.startswith(('grid-','loc-')):
        with OLD_DECODER_SCOPE(model,condition,image_sha,replay=replay) as data:yield data
        return
    import torch
    spec=specification(condition);inner=model.model
    kind=spec.get('decoder_selector',spec.get('selector'));native_prune=inner.layer_prune
    data=dict(condition=condition,calls=0,snapshot=None,mode='replay' if replay else 'capture')
    def prune(*args,**kwargs):
        require(not args and kwargs['rank_layer']==16 and kwargs['cur_num']==0,'Unexpected prune boundary')
        data['calls']+=1;require(data['calls']==1,'Multiple decoder pruning calls')
        supplied={k:staged.clone_tree(kwargs[k]) for k in ('features','position_ids','attention_mask','labels')}
        before=staged.tree_fingerprint(supplied);rng_before=staged.tree_fingerprint(staged.rng_state())
        if replay is not None:
            require(replay['before']==before and replay['rng_before']==rng_before,'Replay input differs')
            require(replay['after']==staged.tree_fingerprint(replay['output']),'Mutated replay')
            output=staged.clone_tree(replay['output']);staged.set_rng(replay['rng_after_state'])
            data['snapshot']=replay;obs=dict(mode='frozen_replay',source_exact=True)
        else:
            start=inner.image_token_posi[0];visual=supplied['features'][0,start:start+96]
            require(visual.shape[0]==96,'Wrong visual span')
            errors=basis=None
            pairs=[]
            if kind=='native':
                # Execute original pruning, replacing ONLY its merge function.
                from llava.model import modeling_llama_x as module
                from original_diagnostics import trace_returns,capture_scores
                records=[];merge_calls=[]
                def identity_merge(features,indices,k,scaling=1):
                    merge_calls.append(1)
                    return features
                with replaced(module,'token_merging',identity_merge),trace_returns({module.LlamaModel.layer_prune.__code__:lambda local:records.append(capture_scores(local,False))}):
                    output=native_prune(**kwargs)
                require(len(records)==len(merge_calls)==1,'Missing native-drop selection/merge bypass')
                rec=records[0];selected=rec['selected_indices_sorted'];e=rec['errors']
                pairs=list(zip(rec['selected_indices'][-10:],rec['basis_indices']))
                basis=torch.tensor(rec['basis_indices'])
                changed_visual=visual.float().clone()
                for slot,src in pairs:changed_visual[slot]=visual[src].float()
                expected=changed_visual[selected].half()
                require(torch.equal(output[2][0,start:start+32],expected),'Native-drop basis assignment/gather differs')
            else:
                if kind=='error':
                    errors,basis=reconstruction_errors(visual);e=errors.detach().cpu().tolist()
                else:e=None
                if kind=='fps':
                    from llava.model.utils import fps
                    selected=sorted(fps(visual.unsqueeze(0),32)[0].cpu().tolist())
                else:
                    pad=np.asarray(properties['padding'])[selected_input];detail=np.asarray(properties['detail'])[selected_input]
                    selected=choose(kind,32,pad,detail,e,image_sha,spec['random_seed'],'decoder')
                output=staged.random_prune(inner,supplied,selected)
                require(torch.equal(output[2][0,start:start+32],visual[selected]),'Drop modified retained states')
            require(len(selected)==len(set(selected))==32,'Final selector must retain32 unique slots')
            sources=[dict(pairs).get(i,i) for i in selected]
            obs=dict(method='basis_localization',selector=kind,basis_inserted=kind=='native',
                selected_compact_positions=selected,selected_original_positions=[selected_input[i] for i in selected],
                direct_source_input_compact_positions=sources,basis_slot_to_source=pairs,
                errors=e,basis_for_scoring=basis.cpu().tolist() if basis is not None else None,merge=None,
                source_visual=tensor_fingerprint(visual))
            data['snapshot']=dict(source_visual=visual.unsqueeze(0).detach().clone(),condition=condition,before=before,rng_before=rng_before,
                output=staged.clone_tree(output),after=staged.tree_fingerprint(output),
                rng_after_state=staged.rng_state(),observation=copy.deepcopy(obs))
        obs['geometry']=staged.prune_geometry(inner,supplied,output,spec)
        data['observation']=obs
        return output
    with replaced(inner,'layer_prune',prune):yield data
    require(data['calls']==1,'Missing decoder prefill');data['status']='passed'


@contextmanager
def adapter(properties,selected_input):
    def decoder(model,condition,image_sha,*,replay=None):
        return grid_scope(model,condition,image_sha,properties,selected_input,replay=replay)
    with replaced(staged,'specification',specification),replaced(staged,'decoder_scope',decoder):yield
