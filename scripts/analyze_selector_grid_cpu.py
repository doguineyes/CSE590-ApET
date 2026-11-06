"""CPU-only paired outcomes and positional coverage audit of selector-grid archives."""
import argparse, csv, hashlib, json, zipfile
from pathlib import Path
import numpy as np
from analyze_textvqa_evidence import native_paths
from textvqa_patch_properties import overlap_coverage


def run(archive, prior, output):
    output.mkdir(parents=True, exist_ok=False)
    matches={m['case_id']:m for m in json.loads((prior/'answer-region-candidates.json').read_text())}
    audit=json.loads((prior/'summary.json').read_text())['visual_audit']
    excluded=set(audit['stricter_excluded_ids'])
    rows=[]; scores={}; groups={}; overlaps=[]
    with zipfile.ZipFile(archive) as z:
        checks=json.loads(z.read('artifact-sha256.json'))
        for name,h in checks.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==h,name
        report=json.loads(z.read('report.json'))
        assert report['status']=='passed' and report['completed_predictions']==3100
        for name in sorted(z.namelist()):
            if not(name.startswith('questions/') and name.endswith('.json')):continue
            q=json.loads(z.read(name)); cid=q['case']['case_id']; groups[cid]=q['case']['image_id']
            prop=json.loads(z.read('properties/'+q['case']['image_sha256']+'.json'))
            pad=np.array(prop['padding']); detail=np.array(prop['detail'])
            area=np.load(prior/'maps'/(cid.replace(':','-')+'.npz'))['answer_patch_overlap']
            m=matches[cid]; unique=m['status']=='unique_candidate'; eligible=unique and cid not in excluded
            scores[cid]={k:a['scoring'] for k,a in q['arms'].items()}
            masks={}
            for k,a in q['arms'].items():
                if k=='baseline576':continue
                if k=='native-96-to-32':
                    s1,final,d1,_,d2,_=native_paths(dict(input_diagnostics=a['input_encoding'],layer16_diagnostics=a['decoder']))
                else:
                    s1=a['input_encoding']['selected_original_positions'];final=a['decoder']['selected_original_positions'];d1=s1;d2=final
                    assert len(s1)==96 and len(final)==32 and set(final)<=set(s1)
                masks[k]=(set(s1),set(final))
                row=dict(case_id=cid,image_id=groups[cid],condition=k,ocr_unique=unique,ocr_strict=eligible)
                for stage,ids,direct in [('s1',s1,d1),('final',final,d2)]:
                    row['coverage_'+stage]=overlap_coverage(direct,area)
                    row['padding_'+stage]=float(pad[ids].mean())
                    row['detail_mean_'+stage]=float(detail[ids].mean())
                    row['detail_share_'+stage]=float(detail[ids].sum()/detail.sum()) if detail.sum() else None
                rows.append(row)
            for other in ['native-96-to-32','grid-detail-drop-s0']:
                overlaps.append(dict(case_id=cid,other=other,stage1_overlap=len(masks['grid-error-drop-s0'][0]&masks[other][0]),final_overlap=len(masks['grid-error-drop-s0'][1]&masks[other][1])))
    def family(prefix):return [k for k in report['conditions'] if k.startswith(prefix)]
    random=family('grid-random-drop'); nonpad=family('grid-nonpad-drop')[:5]
    contrasts=[('detail_drop_minus_random', ['grid-detail-drop-s0'],random),('nonpad_minus_random_matched5',nonpad,random),('detail_merge_minus_native',['grid-detail-merge-s0'],['native-96-to-32']),('E_drop_minus_random',['grid-error-drop-s0'],random),('detail_drop_minus_E_drop',['grid-detail-drop-s0'],['grid-error-drop-s0'])]
    rng=np.random.default_rng(590); ids=sorted(scores); image_ids=sorted(set(groups.values()));stats={}
    for label,left,right in contrasts:
        delta={cid:np.mean([scores[cid][k]['consensus_score'] for k in left])-np.mean([scores[cid][k]['consensus_score'] for k in right]) for cid in ids}
        chunks=[np.array([delta[cid] for cid in ids if groups[cid]==im]) for im in image_ids]
        totals=np.array([x.sum() for x in chunks]);counts=np.array([len(x) for x in chunks]);sample=rng.integers(len(chunks),size=(10000,len(chunks)))
        boot=totals[sample].sum(1)/counts[sample].sum(1)
        stats[label]=dict(difference_pp=100*np.mean(list(delta.values())),paired_image_bootstrap95_pp=(100*np.quantile(boot,[.025,.975])).tolist(),seed_policy='Fixed tested seeds, averaged within each question; image-cluster bootstrap, not seed uncertainty.')
    assert sum(m['status']=='unique_candidate' and cid not in excluded for cid,m in matches.items())==30
    aggregate={}
    for k in report['conditions'][1:]:
        rs=[r for r in rows if r['condition']==k];d={}
        for stage in ['s1','final']:
            for metric in ['padding','detail_mean','detail_share']:
                vals=[r[metric+'_'+stage] for r in rs if r[metric+'_'+stage] is not None];d[metric+'_'+stage]=float(np.mean(vals))
            for subset in ['ocr_unique','ocr_strict']:
                vals=[r['coverage_'+stage] for r in rs if r[subset] and r['coverage_'+stage] is not None];d[subset+'_n']=len(vals);d[subset+'_coverage_'+stage]=float(np.mean(vals))
        aggregate[k]=d
    loss={};gain={}
    for k in ['native-96-to-32','grid-detail-merge-s0']:
        loss[k]=[cid for cid in ids if scores[cid]['baseline576']['full_credit'] and not scores[cid][k]['full_credit']]
        gain[k]=[cid for cid in ids if not scores[cid]['baseline576']['full_credit'] and scores[cid][k]['full_credit']]
    result=dict(source=str(archive),source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),questions=len(ids),images=len(image_ids),contrasts=stats,conditions=aggregate,losses=loss,gains=gain,shared_losses=sorted(set.intersection(*map(set,loss.values()))),shared_gains=sorted(set.intersection(*map(set,gain.values()))),overlap={other:{stage:float(np.mean([r[stage] for r in overlaps if r['other']==other])) for stage in ['stage1_overlap','final_overlap']} for other in ['native-96-to-32','grid-detail-drop-s0']},limits=['30-case strict OCR subset is 32 unique auto-matches minus two audited exclusions; NOT 30 individually human-certified cases.','Coverage counts patches intersecting OCR boxes, not semantic evidence, and contextual features may carry information from elsewhere.','Native coverage uses direct basis-source lineage; mask overlap and pixel properties describe carrier slots.','Pixel detail uses corrected padding-excluding metric, not the old exploratory metric.','Layer16 hidden states/norms not saved; outlier/sink hypothesis unavailable from this archive.','No causal stage attribution or mechanism equivalence from these descriptive comparisons.'])
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for filename,data in [('per-question.csv',rows),('mask-overlap.csv',overlaps)]:
        with (output/filename).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(dict(contrasts=stats,overlap=result['overlap'],shared_losses=result['shared_losses']),indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('prior',type=Path);p.add_argument('output',type=Path);a=p.parse_args();run(a.archive,a.prior,a.output)
