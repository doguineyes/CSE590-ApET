"""Research adapter: no basis insertion, optional all-target cosine averaging.

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
from selector_grid_protocol import specification,choose
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


def merge_all_targets(features,selected):
    import torch
    import torch.nn.functional as F
    n,d=features.shape;chosen=torch.tensor(selected,device=features.device)
    mask=torch.ones(n,dtype=torch.bool,device=features.device);mask[chosen]=False
    dropped=mask.nonzero().flatten();retained=features[chosen].clone()
    if len(dropped):
        sim=F.cosine_similarity(features[dropped].unsqueeze(1),retained.unsqueeze(0),dim=2)
        destinations=sim.argmax(dim=1)
        counts=torch.zeros(len(selected),dtype=torch.int,device=features.device)
        retained.scatter_add_(0,destinations[:,None].expand(-1,d),features[dropped])
        counts.scatter_add_(0,destinations,torch.ones_like(destinations,dtype=counts.dtype))
        retained/=1+counts[:,None]
    else:
        destinations=torch.empty(0,dtype=torch.long,device=features.device);counts=torch.zeros(len(selected),dtype=torch.int,device=features.device)
    out=features.clone();out[chosen]=retained
    return out,dict(target_indices=selected,dropped_indices=dropped.cpu().tolist(),
        assigned_target_indices=chosen[destinations].cpu().tolist(),merge_counts=counts.cpu().tolist(),
        own_weight=(1/(1+counts.float())).cpu().tolist(),basis_inserted=False,
        target_order='ascending input position; cosine ties choose first target')


def make_input(model,pixels,condition,seed,image_sha,full,native,raw,properties):
    import torch
    spec=specification(condition);errors=properties['errors']
    selected=choose(spec['selector'],96,properties['padding'],properties['detail'],errors,image_sha,spec['random_seed'],'input')
    snapshot=staged.clone_tree(full)
    with torch.inference_mode():
        if spec['compensation']=='merge':
            changed,graph=merge_all_targets(raw[0],selected)
            projected=model.get_model().mm_projector(changed.unsqueeze(0))
        else: projected=full['encoded'][0].clone();graph=None
        mask=torch.zeros_like(full['encoded'][1]);mask[:,selected]=True
        snapshot['encoded']=(projected,mask.to(projected.device))
    # E arms continue the same FPS RNG state as native after its input scoring.
    if spec['selector']=='error':snapshot['rng_after_input']=staged.clone_tree(native['rng_after_input'])
    observation=dict(condition=condition,source_patches=tensor_fingerprint(raw),encoded=snapshot_encoded(snapshot['encoded']),
        selector=spec['selector'],basis_inserted=False,selected_original_positions=selected,
        selected_padding_fraction=np.asarray(properties['padding'])[selected].tolist(),
        eligible_nonpadding=int((np.asarray(properties['padding'])<=1e-9).sum()),
        selection_errors=errors.tolist() if spec['selector']=='error' else None,merge=graph)
    snapshot['capture']=dict(version=staged.VERSION,condition=condition,seed=seed,
        pixels=pixels_fingerprint(pixels),observation=observation,fingerprints=fingerprints(snapshot),
        rng_after_input=staged.tree_fingerprint(snapshot['rng_after_input']))
    staged.validate_input(snapshot,pixels,condition)
    return observation,snapshot


@contextmanager
def grid_scope(model,condition,image_sha,properties,selected_input,*,replay=None):
    if not condition.startswith('grid-'):
        with OLD_DECODER_SCOPE(model,condition,image_sha,replay=replay) as data:yield data
        return
    import torch
    spec=specification(condition);inner=model.model
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
            if spec['selector']=='error':
                errors,basis=reconstruction_errors(visual)
                e=errors.detach().cpu().tolist()
            else:e=None
            pad=np.asarray(properties['padding'])[selected_input];detail=np.asarray(properties['detail'])[selected_input]
            selected=choose(spec['selector'],32,pad,detail,e,image_sha,spec['random_seed'],'decoder')
            changed={**supplied};graph=None
            if spec['compensation']=='merge':
                # Native decoder averages fp32 states and casts back to incoming dtype.
                values,graph=merge_all_targets(visual.float(),selected)
                changed['features']=supplied['features'].clone()
                changed['features'][0,start:start+96]=values.to(visual.dtype)
            output=staged.random_prune(inner,changed,selected)
            obs=dict(method='no_basis_selector',selector=spec['selector'],basis_inserted=False,
                selected_compact_positions=selected,selected_original_positions=[selected_input[i] for i in selected],
                errors=e,basis_for_scoring=basis.cpu().tolist() if basis is not None else None,merge=graph,
                source_visual=tensor_fingerprint(visual),
                detail_padding_policy='Inherited target original position, including after stage1 merging')
            data['snapshot']=dict(condition=condition,before=before,rng_before=rng_before,
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
