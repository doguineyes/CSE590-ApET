"""Bounded RQ1 validity checks on Kaggle original runtime; one model load.

Use -I with environments/original/.venv/bin/python. New outputs only; always
export a ZIP, including partial evidence on failure. No changes to native ApET.
"""

import argparse
from datetime import datetime,timezone
import html
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parent))
from check_original_env import REPO,provenance,validate_runtime,verify_native_reference
from rq1_protocol import sha256,write_json
from rq1_validation import (CONDITIONS,INSTRUCTION,METHODS,PROFILE,VERSION,rotate_options,
                            score_rotated,shortcut_summary,validate_profile)
from run_rq1_input import case_seed,generation_score,interrupt_run,load_model,save_image
from smoke_original_apet import MODEL_ID,VISION_ID,generate_native_answer


def copy_evidence(args,report,data):
    evidence=args.output/'evidence';evidence.mkdir()
    shutil.copy2(PROFILE,evidence/'validation-profile.json')
    for dataset in data:
        shutil.copy2(args.inputs/dataset/'manifest.json',evidence/(dataset+'-manifest.json'))
    sources=['scripts/run_rq1_validation.py','scripts/rq1_validation.py','scripts/rq1_validation_native.py',
        'scripts/prepare_rq1_validation.py','scripts/render_rq1_gallery.py','scripts/summarize_rq1_pilot.py',
        'scripts/run_rq1_input.py','scripts/rq1_native.py','scripts/rq1_protocol.py',
        'scripts/check_original_env.py','scripts/smoke_original_apet.py','scripts/original_diagnostics.py',
        'scripts/mmstar_scoring.py','scripts/eval_mmstar.py','scripts/gqa_scoring.py',
        'llava/eval/m4c_evaluator.py','research/protocols/rq1_textvqa_input_v1.json',
        'research/provenance/original_llava_reference.json','environments/original/uv.lock',
        'environments/original/pyproject.toml',*report['implementation']['native_source_sha256']]
    for rel in dict.fromkeys(sources):
        target=evidence/'sources'/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/rel,target)
    report['evidence_sha256']={str(p.relative_to(evidence)):sha256(p) for p in evidence.rglob('*') if p.is_file()}


def image_for(args,data,case):
    from types import SimpleNamespace
    dataset=case['case_id'].split(':')[0]
    return save_image(SimpleNamespace(manifest=args.inputs/dataset/'manifest.json',output=args.output),data[dataset],case)


def run_shortcuts(args,report,profile,data,lookup,loaded):
    from PIL import Image
    from rq1_validation_native import generate_text_only
    tokenizer,model,processor=loaded
    blank=Image.new('RGB',(336,336),tuple(int(v*255) for v in processor.image_mean))
    blank.save(args.output/'blank-input.png')
    report['blank_image']={'rgb':list(blank.getpixel((0,0))),'sha256':sha256(args.output/'blank-input.png')}
    rows=[]
    with (args.output/'shortcuts.jsonl').open('x') as stream:
        def save(row):
            rows.append(row);stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            report['shortcut_summary']=shortcut_summary(rows)
            report['shortcut_progress']={'completed_trials':sum(r['kind']=='trial' for r in rows),
                'expected_trials':72,'completed_repeats':sum(r['kind']=='reference_repeat' for r in rows),'expected_repeats':6}
            write_json(args.output/'report.json',report)
            print(f"Shortcut {row['case_id']} {row['condition']} shift={row['shift']} {row['kind']}: {row['response']!r}; correct={row['scoring']['correct']}",flush=True)
        for cid in profile['shortcuts']['case_ids']:
            case=lookup[cid];image=image_for(args,data,case);seed=case_seed(case,profile['seed'])
            for condition in CONDITIONS:
                for shift in range(4):
                    report['active_step']={'phase':'shortcuts','case_id':cid,'condition':condition,'shift':shift}
                    write_json(args.output/'report.json',report)
                    variant=rotate_options(case['question'],case['canonical_target'],shift)
                    question=variant['question']+'\n'+INSTRUCTION
                    row={'case_id':cid,'kind':'trial','condition':condition,'shift':shift,
                        'variant':variant,'image_sha256':case['image_sha256'] if condition=='real' else None}
                    if condition=='question_only':row.update(generate_text_only(tokenizer,model,question,seed))
                    else:
                        generate_native_answer(tokenizer,model,processor,image if condition=='real' else blank,
                            question,'baseline',seed=seed,max_new_tokens=16,require_nonempty=False,record=row)
                        row['visual_tokens']=576;row['image_placeholders']=1
                    row['scoring']=score_rotated(row['response'],variant);save(row)
            variant=rotate_options(case['question'],case['canonical_target'],0)
            report['active_step']={'phase':'shortcut_reference_repeat','case_id':cid}
            write_json(args.output/'report.json',report)
            row={'case_id':cid,'kind':'reference_repeat','condition':'real','shift':0,'variant':variant}
            generate_native_answer(tokenizer,model,processor,image,case['generation_question'],'baseline',
                seed=seed,max_new_tokens=16,require_nonempty=False,record=row)
            row['scoring']=score_rotated(row['response'],variant);save(row)


def run_perturbations(args,report,profile,data,lookup,loaded):
    from PIL import Image
    import torch
    from rq1_native import encode_image,observe_generation_tokens,replaced
    from rq1_validation_native import measure_alternatives
    tokenizer,model,processor=loaded
    (args.output/'questions').mkdir();(args.output/'features').mkdir()
    current_image=None;encoded=None;summaries=[]
    for entry in profile['perturbations']:
        cid=entry['case_id'];case=lookup[cid];image=image_for(args,data,case)
        seed=case_seed(case,profile['seed'])
        if seed!=entry['seed']:raise ValueError('Compression seed differs from frozen source')
        if current_image!=case['image_sha256']:
            del encoded
            encoded=encode_image(model,processor,image,seed);current_image=case['image_sha256']
            Image.fromarray(encoded['canvas']).save(args.output/'images'/(current_image+'-canvas.png'))
            tensor=args.output/'features'/(current_image+'.pt')
            torch.save({'clip_with_cls':encoded['clip'].detach().cpu().contiguous(),
                'projected_patches':encoded['projected'].detach().cpu().contiguous()},tensor)
            write_json(tensor.with_suffix('.json'),{'image_id':case['image_id'],'image_sha256':current_image,
                'seed':seed,'clip_sha256':encoded['clip_sha256'],'projected_sha256':encoded['projected_sha256'],
                'tensor_file_sha256':sha256(tensor),'diagnostics':encoded['diagnostics'],'processor':processor.to_dict()})
        free,tokens={},{}
        cached=lambda *a,**kw:(encoded['projected'].clone(),encoded['mask'].clone())
        with replaced(model,'encode_images',cached),observe_generation_tokens(model,tokenizer,tokens):
            generate_native_answer(tokenizer,model,processor,image,case['generation_question'],'baseline',
                seed=seed,max_new_tokens=16,require_nonempty=False,record=free)
        free['scoring']=generation_score(cid.split(':')[0],case,free['response'])
        path=args.output/'questions'/(cid.replace(':','-')+'.json')
        last_progress=None
        def progress(record):
            nonlocal last_progress
            record['free_generation']=free;record['image_sha256']=case['image_sha256']
            write_json(path,record)
            zero_count=len(record.get('tokens',[]));alt_count=len(record.get('alternatives',[]))
            tag=(record['status'],zero_count,alt_count)
            if tag!=last_progress and ((alt_count and alt_count%8==0) or (not alt_count and zero_count%16==0)):
                print(f"{cid}: zero={zero_count}/32, alternatives={alt_count}/32 patches; {record['status']}",flush=True)
                last_progress=tag
            report['active_step']={'phase':'perturbations','case_id':cid,'zero_patches':zero_count,'alternative_patches':alt_count}
            write_json(args.output/'report.json',report)
        if not free['scoring']['correct']:
            write_json(path,{'case_id':cid,'status':'baseline_changed','free_generation':free,'generation_tokens':tokens})
            raise RuntimeError('Frozen reference no longer answered correctly; saved failure without retargeting')
        result=measure_alternatives(tokenizer,model,case,encoded,entry,profile['control_tolerance'],progress,tokens,free)
        summaries.append({'case_id':cid,'status':result['status'],'patches':len(result['alternatives']),
            'historical_comparison':result['historical_comparison'],'by_stratum':result['comparison_by_stratum']})
        report['perturbation_questions']=summaries
        report['perturbation_progress']={'completed_questions':len(summaries),'expected_questions':6,
            'scored_token_methods':sum(s['patches']*3 for s in summaries),'expected_token_methods':576}
        write_json(args.output/'report.json',report)
        print(f"Completed alternatives {cid}: 32 patches × 3 methods; full controls passed",flush=True)


def write_review(output,report):
    from summarize_rq1_pilot import write_csv
    parts=['<!doctype html><meta charset="utf-8"><title>RQ1 validity diagnostics</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:1em}table{border-collapse:collapse}td,th{padding:.5em;border:1px solid #ccc}</style>',
        '<h1>RQ1 validity diagnostics</h1><p>Execution status: '+html.escape(report['status'])+'. A passed run means completed execution/controls; it does not certify image grounding or prove token importance.</p>',
        '<p>Original cohorts, ApET source and dependency lock are unchanged. No automatic filtering, target changes or selected-case replacements. Diagnostic patch strata are post-hoc and are shown separately.</p>']
    def table(headers,rows):
        return '<table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</table>'
    shortcut=report.get('shortcut_summary',{})
    if shortcut:
        parts += ['<h2>Image / option reliance</h2>',table(['Case','Real correct /4','Question-only /4','Blank /4','Repeat same semantic choice'],[
            [r['case_id'],r['real']['correct'],r['question_only']['correct'],r['blank']['correct'],r['original_repeat_same_semantic_choice']] for r in shortcut['cases']]),
            '<p>Question-only success and option-order changes are diagnostic flags, not automatic evidence of a shortcut. Question-only removes 576 visual positions; blank-image trials retain them. Scoring maps displayed letters back to original answer content.</p>']
    flat=[]
    for path in sorted((output/'questions').glob('*.json')) if (output/'questions').exists() else []:
        record=json.loads(path.read_text())
        for row in record.get('alternatives',[]):
            for method,effect in row['effects'].items():
                flat.append({'case_id':record['case_id'],'token_index':row['token_index'],'stratum':row['stratum'],
                    'region':row['region'],'fps_basis_source':row['fps_basis_source'],'E':row['current_E'],
                    'method':method,'I':effect['I'],'perturbation_l2':effect['delta_l2'],'original_l2':effect['original_l2'],
                    'archived_I':row['archived_I']})
    write_csv(output/'perturbations.csv',flat)
    if flat:
        parts += ['<h2>Alternative interventions</h2><p><a href="perturbations.csv">All patch/method scores</a>. Different perturbation sizes are recorded; equal I magnitude is not expected. Half-strength is a dose check along the zeroing direction, not independent confirmation. Neighbor replacement can preserve task evidence.</p>']
        for entry in report.get('perturbation_questions',[]):
            rows=[]
            for stratum,methods in entry['by_stratum'].items():
                for method,stats in methods.items():
                    rows.append([stratum,method,stats['patches'],stats['median_signed_I'],stats['rho_E_I'],stats['rho_vs_fresh_zero'],
                        str(stats['sign_agreement_beyond_1e_minus4'])+'/'+str(stats['sign_comparison_patches'])])
            parts += ['<h3>'+html.escape(entry['case_id'])+'</h3>',table(['Stratum','Method','Patches','Median I','E–I rho','Rho vs zero','Sign agreement above 0.0001'],rows)]
    parts.append('<p>All correlations are descriptive within a question and patch stratum. No pooled-token inference or native-compression accuracy claim.</p>')
    (output/'index.html').write_text('\n'.join(parts), encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['all','shortcuts','perturbations'],default='all')
    p.add_argument('--model',default=MODEL_ID);p.add_argument('--vision-model',default=VISION_ID)
    p.add_argument('--cache-dir',type=Path,default=Path('/tmp/apet-original-hf-cache'))
    p.add_argument('--download-budget-gib',type=float,default=0)
    args=p.parse_args();args.inputs=args.inputs.resolve();args.output=args.output.resolve()
    if args.output.exists() or args.output.with_suffix('.zip').exists():p.error('Choose a new output path; previous results are preserved')
    if not math.isfinite(args.download_budget_gib) or args.download_budget_gib<0:p.error('Invalid download budget')
    if args.inputs==args.output or args.inputs in args.output.parents:p.error('Output must be outside accepted input folder')
    os.environ['MPLBACKEND']='Agg';os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES']='0'
    signal.signal(signal.SIGTERM,interrupt_run)
    args.output.mkdir(parents=True);start=time.perf_counter()
    report={'status':'running','version':VERSION,'mode':args.mode,
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    try:
        report.update(provenance('RQ1 validity diagnostics on unchanged original ApET; selected cases, no compression variant'))
        report['status']='running';validate_runtime(report);verify_native_reference(report)
        profile,data,lookup=validate_profile(args.inputs)
        report['profile_sha256']=sha256(PROFILE);report['protocol']=profile
        copy_evidence(args,report,data)
        loaded=load_model(args,report)
        if args.mode in ['all','shortcuts']:run_shortcuts(args,report,profile,data,lookup,loaded)
        if args.mode in ['all','perturbations']:run_perturbations(args,report,profile,data,lookup,loaded)
        import torch
        report['peak_allocated_gib_final_segment']={str(i):torch.cuda.max_memory_allocated(i)/1024**3 for i in range(torch.cuda.device_count())}
        report['memory_note']='Native generation helper resets peak counters per generation; final-segment values are not the whole-run peak. Individual shortcut generations retain their own peak records.'
        report['status']='passed'
    except KeyboardInterrupt:
        report['status']='interrupted';report['error']='Interrupted; completed records preserved'
    except Exception as error:
        report['status']='failed';report['error']=f'{type(error).__name__}: {error}'
        report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
    finally:
        report['wall_seconds']=time.perf_counter()-start;report['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_json(args.output/'report.json',report)
        try:write_review(args.output,report)
        except Exception as error:
            report['review_error']=str(error);write_json(args.output/'report.json',report)
        with zipfile.ZipFile(args.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(args.output.rglob('*')):
                if path.is_file():z.write(path,path.relative_to(args.output))
        print('Status:',report['status'],'Download:',args.output.with_suffix('.zip'),flush=True)
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
