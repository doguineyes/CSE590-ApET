"""Describe basis-source OCR coverage; no causal answer attribution or GPU execution."""
import argparse,csv,hashlib,json,zipfile
from pathlib import Path
import numpy as np


def split_sources(selected, pairs, upstream):
    mapping=dict(pairs)
    assert len(mapping)==10 and set(mapping)<=set(selected)
    basis=set();ordinary=set()
    for slot in selected:
        (basis if slot in mapping else ordinary).update(upstream[mapping.get(slot,slot)])
    return basis,ordinary,{slot:set(upstream[mapping.get(slot,slot)]) for slot in selected}


def run(archive,prior,out):
    out.mkdir(parents=True,exist_ok=False)
    matches={x['case_id']:x for x in json.loads((prior/'answer-region-candidates.json').read_text())}
    excluded=set(json.loads((prior/'summary.json').read_text())['visual_audit']['stricter_excluded_ids'])
    rows=[];cases=[]
    with zipfile.ZipFile(archive) as z:
        for n,h in json.loads(z.read('artifact-sha256.json')).items():assert hashlib.sha256(z.read(n)).hexdigest()==h
        for n in sorted(z.namelist()):
            if not(n.startswith('questions/') and n.endswith('.json')):continue
            q=json.loads(z.read(n));cid=q['case']['case_id'];m=matches[cid]
            native=q['arms']['native-96-to-32'];detail=q['arms']['grid-detail-merge-s0'];base=q['arms']['baseline576']
            if base['scoring']['full_credit'] and native['scoring']['full_credit']!=detail['scoring']['full_credit']:
                cases.append(dict(case_id=cid,loss_type='native_only' if not native['scoring']['full_credit'] else 'detail_only',question=q['case']['question'],gold_answers=json.dumps(q['case']['gold_answers']),baseline=base['response'],native=native['response'],detail_merge=detail['response'],ocr_status=m['status']))
            if m['status']!='unique_candidate':continue
            relevant=set(np.flatnonzero(np.load(prior/'maps'/(cid.replace(':','-')+'.npz'))['answer_patch_overlap']>0).tolist())
            assert relevant
            e=q['arms']['grid-error-drop-s0'];s1=e['input_encoding']['selected_original_positions'];dec=e['decoder'];bcompact=set(dec['basis_for_scoring']);b={s1[i] for i in bcompact};final=set(dec['selected_original_positions'])
            assert len(b)==10 and len(final)==32 and not(b&final)
            pool=set(s1)&relevant;lost=pool-final;hits=b&pool
            errors=np.array(dec['errors']);lost_nonbasis=[i for i,orig in enumerate(s1) if orig in lost and i not in bcompact]
            r=dict(case_id=cid,strict=cid not in excluded,answer_patches=len(relevant),E_stage1_answer_count=len(pool),E_final_answer_count=len(final&relevant),E_basis_answer_count=len(hits),E_lost_answer_count=len(lost),E_lost_nonbasis_count=len(lost-hits),E_basis_max_error=float(errors[list(bcompact)].max()),E_lost_nonbasis_median_error=float(np.median(errors[lost_nonbasis])) if lost_nonbasis else None,E_basis_answer_fraction=len(hits)/len(pool) if pool else None,E_survival=len(final&relevant)/len(pool) if pool else None)
            ni=native['input_encoding'];ns=ni['input']['selected_indices_sorted'];pairs=[(x['assignment_slot'],x['source_index']) for x in ni['basis_slot_to_source']]
            b1,o1,d1=split_sources(ns,pairs,[{i} for i in range(576)])
            nd=native['decoder']['E'];b2,o2,d2=split_sources(nd['selected_indices_sorted'],nd['basis_slot_to_source'],[d1[i] for i in ns])
            for stage,bs,ordinary in [('s1',b1,o1),('final',b2,o2)]:
                bh=bs&relevant;oh=ordinary&relevant
                r.update({f'native_{stage}_basis_coverage':len(bh)/len(relevant),f'native_{stage}_ordinary_coverage':len(oh)/len(relevant),f'native_{stage}_overlap_coverage':len(bh&oh)/len(relevant),f'native_{stage}_total_coverage':len(bh|oh)/len(relevant),f'native_{stage}_basis_answer_count':len(bh),f'native_{stage}_ordinary_answer_count':len(oh)})
            rows.append(r)
    def summarize(rs):
        sums=lambda k:sum(r[k] for r in rs)
        total=sums('E_stage1_answer_count');basis=sums('E_basis_answer_count');lost=sums('E_lost_answer_count')
        rng=np.random.default_rng(590)
        # Conditional random-basis null: choose10 of the actual96 candidates independently per question.
        draws=sum((rng.hypergeometric(r['E_stage1_answer_count'],96-r['E_stage1_answer_count'],10,size=20000) for r in rs),np.zeros(20000,dtype=int))
        answer_counts=np.array([r['E_stage1_answer_count'] for r in rs]);basis_counts=np.array([r['E_basis_answer_count'] for r in rs]);idx=rng.integers(len(rs),size=(10000,len(rs)));den=answer_counts[idx].sum(1);valid=den>0
        frac=basis_counts[idx].sum(1)[valid]/den[valid]
        result=dict(questions=len(rs),questions_with_stage1_answer=sum(r['E_stage1_answer_count']>0 for r in rs),stage1_answer_patch_occurrences=total,final_answer_patch_occurrences=sums('E_final_answer_count'),basis_answer_patch_occurrences=basis,lost_answer_patch_occurrences=lost,lost_nonbasis_occurrences=sums('E_lost_nonbasis_count'),basis_fraction_pooled=basis/total,basis_fraction_image_bootstrap95=np.quantile(frac,[.025,.975]).tolist(),random_basis_fraction=10/96,random_basis_expected_hits=total*10/96,random_basis_null95_hits=np.quantile(draws,[.025,.975]).tolist(),basis_fraction_of_losses=basis/lost,mean_question_basis_fraction=float(np.mean([r['E_basis_answer_fraction'] for r in rs if r['E_basis_answer_fraction'] is not None])),pooled_survival=sums('E_final_answer_count')/total,mean_question_survival=float(np.mean([r['E_survival'] for r in rs if r['E_survival'] is not None])),basis_error_max=max(r['E_basis_max_error'] for r in rs))
        for stage in ['s1','final']:
            keys=[f'native_{stage}_{x}_coverage' for x in ['basis','ordinary','overlap','total']]
            for k in keys:result[k]=float(np.mean([r[k] for r in rs]))
            result[f'native_{stage}_basis_share_of_mean_coverage']=result[keys[0]]/result[keys[3]]
        return result
    result=dict(source=str(archive),source_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),unique32=summarize(rows),strict30=summarize([r for r in rows if r['strict']]),limits=['Basis-source lineage is not causal answer contribution; hidden states are contextual and merge-carried content is excluded.','30-case subset is automatically matched with two audited exclusions, not30 fully human-certified evidence regions.','Pooled patch occurrences weight questions with more candidate patches; mean per-question survival is also reported.','Random-basis null is a synthetic reference on the saved96-token pools, not an executed model intervention.','FPS distinctiveness is relative to feature geometry, not proof of semantic importance.','Near-zero ridge residuals of basis tokens need not be exactly zero; basis tokens were confirmed excluded from E-drop final selection.'])
    (out/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for name,data in [('per-question.csv',rows),('exclusive-loss-cases.csv',cases)]:
        with (out/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    print(json.dumps(result['strict30'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('prior',type=Path);p.add_argument('out',type=Path);a=p.parse_args();run(a.archive,a.prior,a.out)
