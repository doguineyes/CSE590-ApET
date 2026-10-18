"""Add 1x1 input IH to all 30 canonical RQ1 cases, reusing verified E/zero scores.

Original isolated Kaggle Python only. Six-case calibration precedes the gated
30-case full grid; each command loads weights once and always saves ZIP evidence.
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
from rq1_input_ih import (PROFILE,VERSION,comparison,file_at,require,source_hashes,
                          validate_bundle,validate_calibration,verify_record)
from rq1_protocol import sha256,write_json
from run_rq1_input import (generation_score,interrupt_run,load_model,save_image)
from smoke_original_apet import MODEL_ID,VISION_ID,generate_native_answer
from summarize_rq1_pilot import write_csv


def copy_evidence(args,report,index):
    root=args.output/'evidence';root.mkdir()
    shutil.copytree(args.inputs/'references',root/'references')
    for dataset in ['gqa','mmstar','textvqa']:
        dst=root/'inputs'/dataset/'manifest.json';dst.parent.mkdir(parents=True)
        shutil.copy2(args.inputs/dataset/'manifest.json',dst)
    shutil.copy2(PROFILE,root/'input-ih-profile.json')
    extra=['research/tests/test_rq1_input_ih.py','research/tests/test_rq1_input_ih_native.py',
           'research/tests/test_rq1_layer_native.py','scripts/prepare_rq1_ih_inputs.py',
           'research/protocols/rq1_score_reanalysis_v1.json',
           'research/RQ1_INPUT_IH.md','research/notebooks/kaggle_rq1_input_pilot.ipynb']
    for relative in list(report['measurement_sources_sha256'])+extra:
        dst=root/'sources'/relative;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/relative,dst)
    if args.calibration_report:
        shutil.copy2(args.calibration_report,root/'prerequisite-input-ih-calibration.json')
        for cid in report['calibration_prerequisite']['case_ids']:
            relative='questions/'+cid.replace(':','-')+'.json';dst=root/'prerequisite-calibration'/relative
            dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(args.calibration_report.parent/relative,dst)
    report['evidence_sha256']={str(p.relative_to(root)):sha256(p) for p in root.rglob('*') if p.is_file()}


def run_panel(args,report,profile,manifests,records,calibration,loaded):
    from PIL import Image
    from rq1_input_ih_native import measure_input_ih
    from rq1_native import encode_image,observe_generation_tokens,replaced
    tokenizer,model,processor=loaded
    selected=profile['calibration_case_ids'] if args.tokens == 32 else profile['case_ids']
    lookup={r['case']['case_id']:r for r in records}
    (args.output/'questions').mkdir();(args.output/'features').mkdir()
    report.update(questions=[],question_sha256={},fresh_feature_sha256={},expected_case_ids=selected)
    encoded=None;current=None
    for cid in selected:
        item=lookup[cid];case=item['case'];archived=item['reference'];dataset=item['entry']['dataset']
        # save_image uses the existing manifest argument and validates encoded bytes in validate_bundle.
        args.manifest=args.inputs/dataset/'manifest.json'
        image=save_image(args,manifests[dataset],case)
        if current != case['image_sha256']:
            del encoded
            encoded=encode_image(model,processor,image,archived['seed']);current=case['image_sha256']
            Image.fromarray(encoded['canvas']).save(args.output/'images'/(current+'-canvas.png'))
            write_json(args.output/'features'/(current+'.json'),{
                'image_sha256':current,'seed':encoded['seed'],'clip_sha256':encoded['clip_sha256'],
                'projected_sha256':encoded['projected_sha256'],'diagnostics':encoded['diagnostics'],
                'processor':processor.to_dict(),'note':'Strictly matched to original feature hashes; original tensor files remain in the canonical source ZIPs.'})
            report['fresh_feature_sha256']['features/'+current+'.json']=sha256(args.output/'features'/(current+'.json'))
        free,tokens={},{}
        cached=lambda *a,**kw:(encoded['projected'].clone(),encoded['mask'].clone())
        with replaced(model,'encode_images',cached),observe_generation_tokens(model,tokenizer,tokens):
            generate_native_answer(tokenizer,model,processor,image,case['generation_question'],'baseline',
                seed=archived['seed'],max_new_tokens=16,require_nonempty=False,record=free)
        free['scoring']=generation_score(dataset,case,free['response'])
        q={'case_id':cid,'image_id':case['image_id'],'image_sha256':case['image_sha256'],
           'question':case['question'],'measurement_target':archived['measurement_target'],
           'archived_reference_file':item['entry']['question_file'],
           'archived_reference_sha256':sha256(file_at(args.inputs,item['entry']['question_file'])),
           'fresh_feature_file':'features/'+current+'.json','free_generation':free,'generation_tokens':tokens,'status':'running'}
        path=args.output/'questions'/(cid.replace(':','-')+'.json');write_json(path,q)
        require(free['scoring']['correct'] and tokens['generated_token_ids'][:len(archived['target_token_ids'])]
                == archived['target_token_ids'],'Fresh baseline or frozen target prefix changed; no retargeting')
        order=archived['token_execution_order'][:args.tokens]
        def progress(part):
            q['ih']=part;n=len(part['tokens'])
            report['active_step']={'case_id':cid,'positions':n,'expected':args.tokens,'status':part['status']}
            report['progress']={'completed_questions':len(report['questions']),'expected_questions':len(selected),
                'completed_retention_scores':len(report['questions'])*args.tokens+n,
                'expected_retention_scores':len(selected)*args.tokens}
            if n % 16 == 0 or part['status'] != 'running':
                write_json(path,q);write_json(args.output/'report.json',report)
            if n % 64 == 0 or n == args.tokens or part['status'] != 'running':
                print(f'{cid} input IH: {n}/{args.tokens}; {part["status"]}',flush=True)
        try:
            ih=measure_input_ih(model,tokenizer,case,encoded,archived,item['feature'],order,progress,calibration.get(cid))
        finally:
            # The progress record shares its mutable controls/token list, so
            # failed reuse checks also preserve the actual mismatching scores.
            write_json(path,q)
        q['ih']=ih;q['status']=ih['status']
        _,q['comparison']=comparison(archived,ih,item['feature']['diagnostics']['input']['basis_indices'])
        verify_record(q,archived,item['feature'])
        write_json(path,q);report['question_sha256'][str(path.relative_to(args.output))]=sha256(path)
        report['questions'].append({'case_id':cid,'status':q['status'],'positions':args.tokens,
            'archived_reuse_exact':ih['controls']['archived_reuse_exact'],'seconds':ih['seconds'],**q['comparison']['all']})
        report['progress']['completed_questions']=len(report['questions'])
        write_json(args.output/'report.json',report)
        print(f'Completed {cid}: {ih["seconds"]:.1f}s; E/IH rho={q["comparison"]["all"]["rho_E_ih_first"]}',flush=True)
    report['fresh_zero_audit_scores']=3*len(selected)
    report['archived_zero_scores_joined']=len(selected)*args.tokens
    report['new_IH_retention_scores']=len(selected)*args.tokens
    report['calibration_retention_repeats_exact']=sum(
        len(calibration[cid]['ih']['tokens']) for cid in selected if cid in calibration)


def write_review(output,report):
    from render_rq1_gallery import photo_bounds,patch_regions
    index=json.loads((output/'evidence/references/index.json').read_text())
    flags=json.loads((output/'evidence/sources/research/protocols/rq1_score_reanalysis_v1.json').read_text())
    lookup={e['case_id']:e for e in index['cases']};images={}
    for d in ['gqa','mmstar','textvqa']:
        images.update(json.loads((output/'evidence/inputs'/d/'manifest.json').read_text())['images'])
    flat,summaries=[],[]
    for path in sorted((output/'questions').glob('*.json')):
        q=json.loads(path.read_text())
        if q.get('status') != 'passed':continue
        entry=lookup[q['case_id']]
        archived=json.loads(file_at(output/'evidence',entry['question_file']).read_text())
        feature=json.loads(file_at(output/'evidence',entry['feature_file']).read_text())
        verify_record(q,archived,feature)
        rows,s=comparison(archived,q['ih'],feature['diagnostics']['input']['basis_indices'])
        regions=patch_regions(photo_bounds(images[q['image_id']]['size']))
        flat += [{'case_id':q['case_id'],'stage':'input',**r,**regions[r['token_index']]} for r in rows]
        summaries.append({'case_id':q['case_id'],'dataset':q['case_id'].split(':')[0],
            'mmstar_grounding_flag':q['case_id'] in flags['mmstar_grounding_flags'],
            'mmstar_prior_quality_flag':q['case_id'] in flags['mmstar_prior_quality_flags'],
            'target':q['measurement_target'],'first_piece':q['ih']['first_reference_token_string'],
            'positions':s['positions'],**s['all'],
            **{'nonbasis_'+k:v for k,v in s['nonbasis'].items()},
            'full_probability':rows[0]['full_probability'],'remove_all_probability':rows[0]['remove_all_probability'],
            'best_retained_probability':max(r['retained_probability'] for r in rows)})
    write_csv(output/'comparison.csv',flat);write_csv(output/'questions.csv',summaries)
    write_json(output/'summary.json',{'status':report['status'],'questions':summaries,
        'scope':'Same canonical 30-case input panel; calibration is a prefix. Source-informed 1x1 retention adaptation, first tokenizer piece only. No independent importance or population claim.'})
    cols=['case_id','positions','rho_E_zero_first','rho_E_ih_first','rho_zero_ih_first','full_probability','remove_all_probability','best_retained_probability']
    body=['<!doctype html><meta charset="utf-8"><title>RQ1 input E / zero / IH</title>',
        '<style>body{font:16px system-ui;max-width:1400px;margin:2em auto;padding:1em}table{border-collapse:collapse}td,th{padding:.5em;border:1px solid #ccc}</style>',
        '<h1>Input E / zero-one / retain-one</h1><p>Status: '+html.escape(report['status'])+'</p>',
        '<p>E is the original native pre-projector CLIP residual. Both I columns are first frozen target-piece probability differences. Original whole-answer zeroing is retained in the token CSV. Old E/zero records are reused only after exact feature/reference/zero-audit checks.</p>',
        '<p>Retention physically deletes all other visual input embeddings before block 1, keeps all nonvisual embeddings and original position gaps, and uses native SDPA. The null has not processed the image, but preserves the placeholder position gap. This source-informed 1×1 adaptation is not a full IH benchmark reproduction. Zeroing and deletion change both context and operator.</p>',
        '<p><a href="comparison.csv">Matched token scores and padding/basis labels</a> · <a href="questions.csv">Question summaries</a> · <a href="report.json">Provenance and controls</a></p>',
        '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr>']
    for r in summaries:body.append('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in cols)+'</tr>')
    body += ['</table><p>Small selected panel; no pooled-token significance. First-token probability is not whole-answer correctness. Calibration maxima concern only sampled positions. Input and layer-16 results remain labeled separately.</p>']
    (output/'index.html').write_text('\n'.join(body),encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,choices=[32,576],default=32);p.add_argument('--calibration-report',type=Path)
    p.add_argument('--model',default=MODEL_ID);p.add_argument('--vision-model',default=VISION_ID)
    p.add_argument('--cache-dir',type=Path,default=Path('/tmp/apet-original-hf-cache'))
    p.add_argument('--download-budget-gib',type=float,default=0)
    args=p.parse_args();args.inputs=args.inputs.resolve();args.output=args.output.resolve()
    if args.calibration_report:args.calibration_report=args.calibration_report.resolve()
    if args.output.exists() or args.output.with_suffix('.zip').exists():p.error('Choose a new output path')
    if args.inputs == args.output or args.inputs in args.output.parents:p.error('Output must be outside inputs')
    if not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:p.error('Invalid download budget')
    if args.tokens == 32 and args.calibration_report:p.error('Calibration report applies only to full sweep')
    os.environ['MPLBACKEND']='Agg';os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES']='0'
    signal.signal(signal.SIGTERM,interrupt_run);args.output.mkdir(parents=True)
    started=time.perf_counter()
    report={'version':VERSION,'status':'running',
            'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    try:
        report.update(provenance('Project input-stage 1x1 physical IH; strict reuse of archived native E and zero-one'))
        report['status']='running';validate_runtime(report);verify_native_reference(report)
        profile,index,manifests,records=validate_bundle(args.inputs)
        report.update(profile=profile,profile_sha256=sha256(PROFILE),reference_index_sha256=profile['reference_index_sha256'],
                      measurement_sources_sha256=source_hashes())
        calibration={}
        if args.tokens == 576:
            report['calibration_prerequisite'],calibration=validate_calibration(args.calibration_report,args.model,args.vision_model)
        copy_evidence(args,report,index)
        write_json(args.output/'report.json',report)
        print('Validated inputs and reference provenance. Loading original LLaVA once.',flush=True)
        loaded=load_model(args,report);run_panel(args,report,profile,manifests,records,calibration,loaded)
        import torch
        report['peak_allocated_gib']={str(i):torch.cuda.max_memory_allocated(i)/1024**3 for i in range(torch.cuda.device_count())}
        report['status']='passed'
    except KeyboardInterrupt:
        report['status'],report['error']='interrupted','Partial evidence preserved'
    except Exception as error:
        report['status'],report['error']='failed',f'{type(error).__name__}: {error}'
        report['traceback']=traceback.format_exc();print(report['traceback'],flush=True)
    finally:
        report['wall_seconds']=time.perf_counter()-started;report['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_json(args.output/'report.json',report)
        try:write_review(args.output,report)
        except Exception as error:
            report['review_error']=str(error);write_json(args.output/'report.json',report)
        with zipfile.ZipFile(args.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for path in sorted(args.output.rglob('*')):
                if path.is_file():z.write(path,path.relative_to(args.output))
        print('Status:',report['status'],'Download:',args.output.with_suffix('.zip'),flush=True)
    return int(report['status'] != 'passed')


if __name__=='__main__':raise SystemExit(main())
