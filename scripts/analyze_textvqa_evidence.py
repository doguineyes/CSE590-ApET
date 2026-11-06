"""CPU OCR candidate mapping, patch-property audit and provisional seed analysis."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import statistics
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from rq1_protocol import textvqa_evaluator
from read_cpu_tensor_archive import read_tensors
from textvqa_patch_properties import (canvas_box,patch_overlap,padding_fraction,detail_energy,
    spearman,overlap_coverage,source_paths)

OCR_URL='https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_val.json'


def write_json(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def write_csv(path,rows):
    if not rows:return
    with path.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(blob):return hashlib.sha256(blob).hexdigest()


def match_answer(case,ocr,normalize,size):
    counts=Counter(normalize(a) for a in case['gold_answers'])
    refs={a for a,n in counts.items() if a and n>=3}
    words=ocr['ocr_info']; matches=[]
    for start in range(len(words)):
        # Multiword runs are only suggestions: OCR order is not guaranteed reading order.
        for length in range(1,min(6,len(words)-start)+1):
            text=' '.join(w['word'] for w in words[start:start+length])
            norm=normalize(text)
            if norm not in refs:continue
            boxes=[];flags=[]
            for w in words[start:start+length]:
                b=w['bounding_box']
                try:boxes.append(canvas_box(b,*size))
                except ValueError:flags.append('invalid_box')
                if abs(float(b.get('rotation',0)))>.01 or abs(float(b.get('yaw',0)))>10 or abs(float(b.get('roll',0)))>10:
                    flags.append('rotation_or_perspective')
            matches.append(dict(answer=norm,reference_count=counts[norm],ocr_indices=list(range(start,start+length)),
                ocr_text=text,boxes=boxes,flags=sorted(set(flags)),multiword=length>1))
    # Prefer a word match over a larger OCR run with the identical normalized answer.
    singles=[m for m in matches if not m['multiword']]
    candidates=singles if singles else matches
    if not candidates:status='unmatched'
    elif len(candidates)>1:status='ambiguous_multiple_matches'
    elif candidates[0]['multiword']:status='multiword_review'
    elif candidates[0]['flags']:status='geometry_review'
    else:status='unique_candidate'
    area=np.zeros(576)
    for m in candidates:
        for b in m['boxes']:area=np.maximum(area,patch_overlap(b))
    return dict(status=status,normalized_reference_counts=dict(counts),candidates=candidates,
                relevant_patch_ids=np.flatnonzero(area>0).tolist(),
                policy='At least3 agreeing annotations; official answer normalization; no fuzzy matching. Unique candidates remain provisional until visual audit.'),area


def property_row(q,tensors):
    dump=q['feature_dump'];clip=tensors['clip_patches'];pixels=tensors['pixels'];layer=tensors['native_layer16_visual']
    if clip.shape!=(1,576,1024) or layer.shape!=(1,96,4096):raise ValueError('Unexpected feature shapes')
    e=np.asarray(dump['input_diagnostics']['input']['errors']);norm=np.linalg.norm(clip[0].astype(float),axis=1)
    detail=detail_energy(pixels);pad=padding_fraction(*dump['image_size'])
    if not all(np.isfinite(v).all() for v in (e,norm,detail,pad,layer)):raise ValueError('Nonfinite feature')
    basis=dump['input_diagnostics']['input']['basis_indices'];mask=np.ones(576,bool);mask[basis]=False
    content=pad<1e-9
    e2=np.asarray(dump['layer16_diagnostics']['E']['errors']);n2=np.linalg.norm(layer[0].astype(float),axis=1)
    row=dict(case_id=q['case']['case_id'],image_id=q['case']['image_id'],padding_area_fraction=float(pad.mean()),
        rho_E_norm=spearman(e,norm),rho_E_detail=spearman(e,detail),rho_E_nonpadding=spearman(e,1-pad),
        rho_E_norm_content=spearman(e[content],norm[content]),rho_E_detail_content=spearman(e[content],detail[content]),
        mean_norm_content=float(norm[content].mean()),mean_norm_padding=float(norm[pad>.999].mean()) if (pad>.999).any() else None,
        rho_E_norm_nonbasis=spearman(e[mask],norm[mask]),rho_E_detail_nonbasis=spearman(e[mask],detail[mask]),
        rho_E_nonpadding_nonbasis=spearman(e[mask],1-pad[mask]),rho_layer16_E_norm=spearman(e2,n2),
        max_norm_over_median=float(norm.max()/np.median(norm)),basis_mean_norm=float(norm[basis].mean()),
        top96_E_norm_overlap=len(set(np.argsort(e)[-96:])&set(np.argsort(norm)[-96:]))/96,
        top96_E_detail_overlap=len(set(np.argsort(e)[-96:])&set(np.argsort(detail)[-96:]))/96)
    return row,dict(E=e,norm=norm,detail=detail,padding=pad,layer16_E=e2,layer16_norm=n2)


def native_paths(dump):
    diag=dump['input_diagnostics'];selected=diag['input']['selected_indices_sorted']
    basis=[(m['assignment_slot'],m['source_index']) for m in diag['basis_slot_to_source']]
    direct1,all1=source_paths(selected,basis,diag['input_merge'],[{i} for i in range(576)])
    layer=dump['layer16_diagnostics'];sel2=layer['E']['selected_indices_sorted'];b2=layer['E']['basis_slot_to_source']
    direct2,_=source_paths(sel2,b2,layer['merge'],[direct1[i] for i in selected])
    _,all2=source_paths(sel2,b2,layer['merge'],[all1[i] for i in selected])
    union=lambda d: sorted(set().union(*d.values()))
    return selected,[selected[i] for i in sel2],union(direct1),union(all1),union(direct2),union(all2)


def mean(values):
    values=[v for v in values if v is not None]
    return statistics.mean(values) if values else None


def paired_summary(questions,rows):
    # No within-question causal attribution; only paired descriptive contrasts.
    output={}
    for stage in ('s1','final'):
        differences=[];rhos=[];native_gaps=[]
        for q in questions:
            cid=q['case_id'];group=[r for r in rows if r['case_id']==cid and r['method']=='random']
            good=[r['coverage_'+stage] for r in group if r['full_credit']];bad=[r['coverage_'+stage] for r in group if not r['full_credit']]
            if good and bad:differences.append(mean(good)-mean(bad))
            rho=spearman([r['coverage_'+stage] for r in group],[r['score'] for r in group])
            if rho is not None:rhos.append(rho)
            native=next(r for r in rows if r['case_id']==cid and r['method']=='native')
            native_gaps.append(native['direct_source_coverage_'+stage]-mean([r['coverage_'+stage] for r in group]))
        interval=None
        if differences:
            rng=np.random.default_rng(5901105)
            boot=np.array([np.mean(rng.choice(differences,len(differences),replace=True)) for _ in range(5000)])
            interval=np.quantile(boot,[.025,.975]).tolist()
        output[stage]=dict(questions=len(questions),varying_full_credit_questions=len(differences),
            mean_correct_minus_wrong_coverage=mean(differences),paired_question_bootstrap95=interval,
            positive_differences=sum(d>1e-12 for d in differences),negative_differences=sum(d< -1e-12 for d in differences),
            ties=sum(abs(d)<=1e-12 for d in differences),mean_within_question_score_coverage_rho=mean(rhos),
            defined_rho_questions=len(rhos),mean_native_direct_minus_random_coverage=mean(native_gaps))
    return output


def run(archive,ocr_path,output,audit_path=None):
    output.mkdir(parents=True,exist_ok=False);(output/'images').mkdir();(output/'maps').mkdir()
    raw=ocr_path.read_bytes();ocr=json.loads(raw)
    if ocr['dataset_type']!='val' or ocr['dataset_name']!='textvqa_rosetta_ocr':raise ValueError('Wrong OCR source')
    lookup={r['image_id']:r for r in ocr['data']}
    if len(lookup)!=len(ocr['data']):raise ValueError('Duplicate OCR image IDs')
    normalize=textvqa_evaluator().answer_processor
    matches=[];properties=[];rows=[];question_records=[]
    with zipfile.ZipFile(archive) as z:
        checks=json.loads(z.read('artifact-sha256.json'))
        def read(name):
            b=z.read(name)
            if name in checks and sha(b)!=checks[name]:raise ValueError('Checksum mismatch '+name)
            return b
        report=json.loads(read('report.json'))
        if report['status']!='passed' or report['completed_predictions']!=2200:raise ValueError('Incomplete source run')
        for n in sorted(z.namelist()):
            if not(n.startswith('questions/') and n.endswith('.json')):continue
            q=json.loads(read(n));case=q['case'];cid=case['case_id'];key=case['image_sha256'];dump=q['feature_dump']
            if case['image_id'] not in lookup:raise ValueError('OCR image missing '+case['image_id'])
            blob=read(dump['file'])
            if sha(blob)!=dump['sha256']:raise ValueError('Tensor hash mismatch')
            tensors=read_tensors(blob)
            if sha(tensors['clip_patches'].tobytes())!=dump['input_diagnostics']['source_patches']['sha256']:raise ValueError('NumPy feature conversion changed bytes')
            pr,maps=property_row(q,tensors);properties.append(pr)
            match,area=match_answer(case,lookup[case['image_id']],normalize,dump['image_size'])
            match.update(case_id=cid,image_id=case['image_id'],question=case['question'],gold_answers=case['gold_answers'],image_size=dump['image_size'])
            matches.append(match)
            selected,final,direct1,all1,direct2,all2=native_paths(dump)
            for condition,arm in q['arms'].items():
                if condition=='baseline576':continue
                native=condition=='native-96-to-32'
                if native:s1,last=selected,final
                else:s1,last=[arm['decoder']['random_plan'][k] for k in ('input_original_positions','decoder_original_positions')]
                pad,detail,norm=[maps[k] for k in ('padding','detail','norm')]
                r=dict(case_id=cid,condition=condition,method='native' if native else 'random',ocr_status=match['status'],
                    full_credit=arm['scoring']['correct'],score=arm['scoring']['consensus_score'],answer=arm['response'],
                    coverage_s1=overlap_coverage(s1,area),coverage_final=overlap_coverage(last,area),
                    direct_source_coverage_s1=overlap_coverage(direct1 if native else s1,area),
                    direct_source_coverage_final=overlap_coverage(direct2 if native else last,area),
                    merged_path_coverage_s1=overlap_coverage(all1,area) if native else None,
                    merged_path_coverage_final=overlap_coverage(all2,area) if native else None,
                    slot_padding_s1=float(pad[s1].mean()),slot_padding_final=float(pad[last].mean()),
                    slot_detail_share_s1=float(detail[s1].sum()/detail.sum()),slot_norm_share_s1=float(norm[s1].sum()/norm.sum()),
                    direct_source_padding_s1=float(pad[direct1 if native else s1].mean()))
                rows.append(r)
            maps['answer_patch_overlap']=area
            np.savez_compressed(output/'maps'/(cid.replace(':','-')+'.npz'),**maps)
            render_maps(output/'images'/(cid.replace(':','-')+'-properties.png'),maps)
            canvas=Image.open(io.BytesIO(read('images/'+key+'-canvas.png'))).convert('RGB').resize((672,672))
            draw=ImageDraw.Draw(canvas)
            for i in range(25):
                draw.line((i*28,0,i*28,672),fill=(150,150,150),width=1);draw.line((0,i*28,672,i*28),fill=(150,150,150),width=1)
            for m in match['candidates']:
                for b in m['boxes']:draw.rectangle(tuple(v*2 for v in b),outline='red',width=4)
            canvas.save(output/'images'/(cid.replace(':','-')+'.png'))
            question_records.append(dict(case_id=cid,baseline=q['arms']['baseline576']['response'],native=q['arms']['native-96-to-32']['response']))
            print(cid,match['status'],flush=True)
        write_json(output/'ocr-cohort-subset.json',dict(source_url=OCR_URL,source_sha256=sha(raw),
            data=[lookup[m['image_id']] for m in matches]))
        # Mechanical subset only; no outcome-based question selection.
        unique=[m for m in matches if m['status']=='unique_candidate']
        summary=dict(analysis_sources_sha256={name:sha((Path(__file__).parent/name).read_bytes()) for name in ('analyze_textvqa_evidence.py','textvqa_patch_properties.py','read_cpu_tensor_archive.py')},version='textvqa-evidence-cpu-v1',source_archive=str(archive),source_archive_sha256=sha(archive.read_bytes()),
            ocr_url=OCR_URL,ocr_sha256=sha(raw),ocr_source_metadata={k:v for k,v in ocr.items() if k!='data'},
            questions=len(matches),status_counts=dict(Counter(m['status'] for m in matches)),
            feature_correlations={k:dict(median=float(np.median([p[k] for p in properties if p[k] is not None])),
                defined_questions=sum(p[k] is not None for p in properties)) for k in properties[0] if k.startswith('rho_')},
            unique_candidate_analysis=paired_summary(unique,rows),
            limits='Provisional automatic OCR matches, not ground-truth evidence. No fuzzy matching. Contextual states; coverage is positional/path proxy. Stage outcomes observational, not causal mediation. Excludes ambiguous/unmatched only from provisional coverage analysis, never from cohort performance.')
    if audit_path is not None:
        audit_data=json.loads(audit_path.read_text())
        reviewed={r['case_id']:r for r in audit_data['cases']}
        excluded={cid for cid,r in reviewed.items() if r['exclude_from_stricter_sensitivity']}
        if not set(reviewed)<=set(m['case_id'] for m in matches):raise ValueError('Unknown reviewed case')
        summary['visual_audit']=dict(source=str(audit_path),sha256=sha(audit_path.read_bytes()),reviewed_cases=len(reviewed),
            stricter_excluded_ids=sorted(excluded),scope=audit_data['scope'])
        summary['stricter_sensitivity']=paired_summary([m for m in unique if m['case_id'] not in excluded],rows)
        for m in matches:
            if m['case_id'] in reviewed:m['visual_review']=reviewed[m['case_id']]
        write_json(output/'visual-audit-record.json',audit_data)
    write_json(output/'answer-region-candidates.json',matches);write_json(output/'summary.json',summary)
    write_csv(output/'patch-properties.csv',properties);write_csv(output/'seed-coverage.csv',rows)
    audit=sorted([m for m in matches if m['status']!='unmatched'],key=lambda m:sha(('ocr-audit-v1|'+m['case_id']).encode()))[:20]
    write_json(output/'visual-audit20.json',[m.get('visual_review',dict(case_id=m['case_id'],status=m['status'],review='pending')) for m in audit])
    (output/'README.txt').write_text('Open index.html. Results are provisional OCR candidates, not validated answer evidence. Read summary.json limits. Pixel maps in maps/ use row-major24x24 order. Native slot coverage differs from direct-source coverage after basis replacement; merged-path coverage is provenance only. See the tracked CPU analysis protocol.\n')
    gallery(output,matches,question_records,audit)
    return summary


def render_maps(path,maps):
    panel=Image.new('RGB',(1024,290),'white');draw=ImageDraw.Draw(panel)
    for i,(key,label) in enumerate([('E','Reconstruction E'),('norm','Feature norm'),('detail','Pixel detail'),('padding','Padding fraction')]):
        a=maps[key].reshape(24,24)
        lo,hi=(0,1) if key=='padding' else np.quantile(a,[.05,.95])
        t=np.clip((a-lo)/max(float(hi-lo),1e-12),0,1)
        rgb=np.stack([40+210*t,65+160*t,170-135*t],axis=-1).astype('uint8')
        im=Image.fromarray(rgb).resize((240,240),Image.Resampling.NEAREST)
        panel.paste(im,(i*256,25));draw.text((i*256+3,4),label,fill='black')
        draw.text((i*256+3,270),f'{lo:.3g} to {hi:.3g}',fill='black')
    panel.save(path)


def gallery(output,matches,records,audit):
    esc=lambda x:html.escape(str(x));lookup={r['case_id']:r for r in records};auditids={m['case_id'] for m in audit}
    parts=['<!doctype html><meta charset="utf-8"><title>TextVQA OCR mapping candidates</title><style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:24px}article{border-top:1px solid #ccc;padding:20px 0}img{width:560px;max-width:100%}pre{white-space:pre-wrap}.warn{color:#963}</style>',
    '<h1>TextVQA answer-region candidates</h1><p>Red boxes: normalized reference-matching OCR. Grid: actual padded336×336 model canvas,24×24 patches. No boxes means no match, not no visual evidence. All mappings are provisional. Audit20 selected among matched candidates by case-ID hash, without outcomes.</p>']
    for m in sorted(matches,key=lambda m:(m['case_id'] not in auditids,m['case_id'])):
        r=lookup[m['case_id']];parts.append('<article><h2>'+esc(m['case_id'])+(' — audit20' if m['case_id'] in auditids else '')+'</h2><p>'+esc(m['question'])+'</p><p class="warn">'+esc(m['status'])+'</p><p>'+esc(m.get('visual_review',{}).get('note','Not individually visually reviewed'))+'</p><p>References: '+esc(m['gold_answers'])+'</p><img src="images/'+m['case_id'].replace(':','-')+'.png"><p>Baseline: '+esc(r['baseline'])+'; native: '+esc(r['native'])+'</p><pre>'+esc(json.dumps(m['candidates'],ensure_ascii=False,indent=2))+'</pre><p>Patch properties (blue=low, yellow=high; per-image5–95% scale, padding0–1):</p><img style="width:1000px" src="images/'+m['case_id'].replace(':','-')+'-properties.png"></article>')
    (output/'index.html').write_text('\n'.join(parts))
    font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc',16) if Path('/System/Library/Fonts/Helvetica.ttc').exists() else ImageFont.load_default()
    for block in range(4):
        sheet=Image.new('RGB',(1000,820),'white');d=ImageDraw.Draw(sheet)
        for i,m in enumerate(audit[block*5:block*5+5]):
            x=(i%3)*333;y=(i//3)*410
            im=Image.open(output/'images'/(m['case_id'].replace(':','-')+'.png')).resize((320,320));sheet.paste(im,(x,y+55))
            d.text((x+3,y+3),m['case_id'],font=font,fill='black');d.text((x+3,y+25),m['status'],font=font,fill='black')
        sheet.save(output/f'audit-sheet-{block+1}.png')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('ocr',type=Path);p.add_argument('output',type=Path);p.add_argument('--visual-audit',type=Path);a=p.parse_args()
    print(json.dumps(run(a.archive,a.ocr,a.output,a.visual_audit),indent=2))
