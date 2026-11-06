"""NumPy FPS precision replay and source-linked norm audit; never model inference."""
import argparse,collections,csv,hashlib,json,zipfile
from pathlib import Path
import numpy as np
from read_cpu_tensor_archive import read_tensors


def fps_replay(x,start,k,half):
    x=x.astype(np.float16 if half else np.float32);dist=np.full(len(x),1e10,np.float32);chosen=[];center=start;infsteps=0
    with np.errstate(over='ignore',invalid='raise'):
        for _ in range(k):
            chosen.append(int(center));delta=x-x[center];square=delta*delta
            # Torch half reductions accumulate in float32 then round to half.
            current=square.sum(axis=1,dtype=np.float32)
            if half:current=current.astype(np.float16)
            infsteps+=int(np.isinf(current).any());dist=np.minimum(dist,current);center=int(dist.argmax())
    return chosen,infsteps


def run(archive,seed_archive,out):
    out.mkdir(exist_ok=False,parents=True);rows=[];cache={}
    with zipfile.ZipFile(archive) as z,zipfile.ZipFile(seed_archive) as old:
        report=json.loads(z.read('report.json'));assert report['status']=='passed' and report['completed_predictions']==1600
        checks=json.loads(z.read('artifact-sha256.json'))
        for qname in sorted(n for n in z.namelist() if n.startswith('questions/') and n.endswith('.json')):
            assert hashlib.sha256(z.read(qname)).hexdigest()==checks[qname]
            q=json.loads(z.read(qname));cid=q['case']['case_id'];oq=json.loads(old.read(qname));dump=oq['feature_dump'];blob=old.read(dump['file']);assert hashlib.sha256(blob).hexdigest()==dump['sha256']
            clip=read_tensors(blob)['clip_patches'][0].astype(np.float32);clipnorm=np.linalg.norm(clip,axis=1)
            props=json.loads(z.read('properties/'+q['case']['image_sha256']+'.json'))
            for arm,a in q['arms'].items():
                if 'state_dump' not in a:continue
                sd=a['state_dump'];blob=z.read(sd['file']);assert hashlib.sha256(blob).hexdigest()==sd['sha256']
                x=read_tensors(blob)['layer16_visual_before_prune'][0];assert x.shape==(96,4096)
                norm=np.linalg.norm(x.astype(np.float32),axis=1);top=int(norm.argmax());enc=a['input_encoding'];dec=a['decoder']
                assert np.allclose(norm,sd['norms'],rtol=1e-5,atol=1e-4)
                if arm=='native-96-to-32':
                    slots=enc['input']['selected_indices_sorted'];mapping={b['assignment_slot']:b['source_index'] for b in enc['basis_slot_to_source']};sources=[mapping.get(i,i) for i in slots]
                    basis_sources={b['source_index'] for b in enc['basis_slot_to_source']};m2=dict(dec['E']['basis_slot_to_source']);retained=[m2.get(i,i) for i in dec['E']['selected_indices_sorted']]
                else:
                    sources=enc['direct_source_original_positions'];basis_sources={b['source_index'] for b in enc.get('basis_slot_to_source',[])};retained=dec['direct_source_input_compact_positions']
                orig=sources[top];r=dict(case_id=cid,condition=arm,max_norm=float(norm[top]),median_norm=float(np.median(norm)),max_median_ratio=float(norm[top]/np.median(norm)),top_compact=top,top_source_patch=orig,patch_row=orig//24,patch_col=orig%24,source_padding=props['padding'][orig],source_clip_norm_rank=1+int((clipnorm>clipnorm[orig]).sum()),source_is_input_basis=orig in basis_sources,top_source_retained=top in retained,tokens_above10median=int((norm>10*np.median(norm)).sum()),top_second_ratio=float(norm.max()/np.partition(norm,-2)[-2]))
                audit=dec.get('fps_numerics')
                if audit:
                    original=audit['centroid_order'];start=original[0];k=len(original);key=(sd['file'],start,k)
                    if key not in cache:cache[key]=(fps_replay(x,start,k,True),fps_replay(x,start,k,False))
                    (half,inf),(full,_) =cache[key];r.update(fps_k=k,recorded_overflow=audit['any_inf'],numpy_half_exact=half==original,half_overlap=len(set(half)&set(original)),fp32_overlap=len(set(full)&set(original)),fp32_order_equal=full==original,numpy_half_inf_steps=inf,fp32_selected=json.dumps(full))
                rows.append(r)
            print(cid,flush=True)
    columns=sorted(set().union(*(r.keys() for r in rows)))
    with (out/'per-arm-question.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=columns);w.writeheader();w.writerows(rows)
    summary={}
    for arm in sorted({r['condition'] for r in rows}):
        rs=[r for r in rows if r['condition']==arm];f=[r for r in rs if 'fps_k' in r]
        summary[arm]=dict(cases=len(rs),median_max_norm=float(np.median([r['max_norm'] for r in rs])),median_max_median_ratio=float(np.median([r['max_median_ratio'] for r in rs])),ratio_above10=sum(r['max_median_ratio']>10 for r in rs),top_from_input_basis=sum(r['source_is_input_basis'] for r in rs),top_retained=sum(r['top_source_retained'] for r in rs),median_clip_rank=float(np.median([r['source_clip_norm_rank'] for r in rs])),top_on_fully_padded_patch=sum(r['source_padding']>=1-1e-9 for r in rs),source_clip_rank1_count=sum(r['source_clip_norm_rank']==1 for r in rs),tokens_above10median_distribution=dict(collections.Counter(r['tokens_above10median'] for r in rs)),median_top_second_norm_ratio=float(np.median([r['top_second_ratio'] for r in rs])))
        if f:summary[arm].update(fps_k=f[0]['fps_k'],fp32_same_set=sum(r['fp32_overlap']==r['fps_k'] for r in f),native_overflow=sum(r['recorded_overflow'] for r in f),numpy_half_exact=sum(r['numpy_half_exact'] for r in f),fp32_same_order=sum(r['fp32_order_equal'] for r in f),mean_fp32_overlap=float(np.mean([r['fp32_overlap'] for r in f])),min_fp32_overlap=min(r['fp32_overlap'] for r in f))
    result=dict(archive=str(archive),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),arms=summary,limits=['NumPy CPU arithmetic is not CUDA; fp16 reproduction checked per trajectory. FP32 replay changes distance arithmetic only on fixed quantized layer16 states and same start index.','No answers recomputed. Precision effect on selection does not prove answer effect. High norm does not establish attention-sink behavior.','Source positions are direct lineage through basis replacements; contextual states may aggregate information from elsewhere.','FPS-only input contains96 FPS-selected tokens, not a distinguished10-token basis. Ratio10 is a descriptive threshold, not a prevalidated sink definition.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('seed_archive',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.archive,a.seed_archive,a.out)
