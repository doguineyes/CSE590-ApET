"""Scoped research hook. Original native implementation is not edited."""
from contextlib import contextmanager,ExitStack
import copy
import basis_localization_native as base
import rq2_staged_native as staged
from rq1_native import replaced
from rq2_protocol import require
from norm_intervention_protocol import specification,parent,swap_plan

def make_input(*args,**kwargs):
    with replaced(base,'specification',specification):return base.make_input(*args,**kwargs)

@contextmanager
def scope(model,condition,key,properties,selected,*,replay=None):
    import torch
    if not condition.startswith('loc-norm-'):
        with base.grid_scope(model,condition,key,properties,selected,replay=replay) as data:yield data
        return
    inner=model.model
    # Parent hook provides native basis placement, RNG, replay and geometry checks.
    with base.grid_scope(model,parent(condition),key,properties,selected,replay=replay) as data:
        hooked=inner.layer_prune
        def prune(*args,**kwargs):
            with ExitStack() as stack:
                if condition.endswith('fp32') and replay is None:
                    from llava.model import utils,modeling_llama_x
                    original=utils.fps
                    def fp32(x,k):return original(x.float(),k)
                    stack.enter_context(replaced(utils,'fps',fp32))
                    stack.enter_context(replaced(modeling_llama_x,'fps',fp32))
                    # Audit must use the same arithmetic as the experimental selector.
                    audit=base.fps_numerical_audit
                    stack.enter_context(replaced(base,'fps_numerical_audit',lambda x,ids:audit(x.float(),ids)))
                output=hooked(*args,**kwargs)
            if replay is not None:return output
            snap=data['snapshot'];obs=data['observation'];visual=snap['source_visual'][0]
            sources=obs['direct_source_input_compact_positions'];carriers=obs['selected_compact_positions']
            start=inner.image_token_posi[0]
            # Native layer_prune explicitly rounds its selected visual states to half,
            # even in the float32 tiny CPU fixture. Preserve that parent's policy.
            values=visual.half().to(output[2].dtype) if obs['selector']=='native' else visual.to(output[2].dtype)
            require(torch.equal(output[2][0,start:start+32],values[sources]),'Parent source mapping not exact')
            if condition.endswith('fp32'):pairs=[]
            else:
                def farthest(slots):
                    remaining=[s for j,s in enumerate(sources) if j not in slots]
                    dist=torch.full((96,),1e10,device=visual.device)
                    for src in remaining:dist=torch.minimum(dist,((visual-visual[src])**2).sum(-1))
                    dist[list(set(sources))]=-float('inf')
                    return int(dist.argmax())
                pairs=swap_plan(condition,visual.float().norm(dim=-1).cpu().tolist(),sources,carriers,
                    {p[0] for p in obs['basis_slot_to_source']},obs['errors'],key,farthest)
            changed=staged.clone_tree(output);new_sources=list(sources)
            for j,old,new in pairs:
                changed[2][0,start+j]=values[new];new_sources[j]=new
            untouched=[j for j in range(32) if j not in {p[0] for p in pairs}]
            require(torch.equal(changed[2][0,start:start+32][untouched],output[2][0,start:start+32][untouched]),'Unchanged slots differ')
            require(torch.equal(changed[2][0,start:start+32],values[new_sources]),'Replacement source mismatch')
            norms=visual.float().norm(dim=-1)
            obs['pool_norms']=norms.cpu().tolist()
            obs['high_norm_threshold']=float(10*base.np.median(obs['pool_norms']))
            obs['removed_basis_slots']=[carriers[j] in {p[0] for p in obs['basis_slot_to_source']} for j,_,_ in pairs]
            obs.update(parent=parent(condition),intervention=condition,replacements=[dict(output_slot=j,removed_source=o,added_source=n) for j,o,n in pairs],
                parent_direct_sources=list(sources),direct_source_input_compact_positions=new_sources,
                fixed_carriers=True,unchanged_slots_exact=True,decoder_fps_fp32=condition.endswith('fp32'))
            snap.update(condition=condition,output=staged.clone_tree(changed),after=staged.tree_fingerprint(changed),observation=copy.deepcopy(obs))
            return changed
        with replaced(inner,'layer_prune',prune):yield data

@contextmanager
def adapter(properties,selected):
    def decoder(model,condition,key,*,replay=None):return scope(model,condition,key,properties,selected,replay=replay)
    with replaced(staged,'specification',specification),replaced(staged,'decoder_scope',decoder):yield
