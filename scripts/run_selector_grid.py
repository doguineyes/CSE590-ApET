"""Frozen no-basis selector grid; TextVQA100 first, GQA800 in image shards."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import unittest
import zipfile
os.environ['MPLBACKEND']='Agg'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from check_original_env import REPO,provenance,validate_runtime,verify_native_reference
from rq1_protocol import sha256,write_json
from rq2_protocol import require
from selector_grid_protocol import VERSION,conditions,specification,choose
from textvqa_patch_properties import padding_fraction,detail_without_padding
from run_rq1_input import load_model,save_image,case_seed,interrupt_run
from run_rq2_random_calibration import prepare_pixels
from run_rq2_staged_calibration import compare_replay
from run_rq2_staged_textvqa100 import measurement_hashes as old_hashes
from smoke_original_apet import MODEL_ID,VISION_ID
REFERENCE=REPO/'research/protocols/selector_grid_textvqa_reference_v1.json'
TESTS=('research.tests.test_rq2_staged_native','research.tests.test_selector_grid',
       'research.tests.test_rq2_staged_textvqa_protocol','research.tests.test_rq2_staged_gqa_protocol')


def signature():
    extra=('scripts/selector_grid_protocol.py','scripts/selector_grid_native.py','scripts/run_selector_grid.py',
           'scripts/textvqa_patch_properties.py','scripts/prepare_selector_grid_reference.py','research/tests/test_selector_grid.py',
           'research/protocols/selector_grid_textvqa_reference_v1.json','research/protocols/selector_grid_v1.json',
           'scripts/rq2_staged_gqa_protocol.py','scripts/eval_original_gqa_subset.py',
           'research/protocols/rq2_staged_gqa800_v1.json')
    return {**old_hashes(),**{p:sha256(REPO/p) for p in extra}}


def check(output):
    suite=unittest.defaultTestLoader.loadTestsFromNames(TESTS)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    passed=result.wasSuccessful() and not result.skipped
    write_json(output,dict(version=VERSION,status='passed' if passed else 'failed',tests=result.testsRun,
                          skipped=len(result.skipped),modules=list(TESTS),source_sha256=signature()))
    return int(not passed)


def validate_calibration(path):
    require(path is not None,'Supply passed TextVQA grid calibration report')
    r=json.loads(path.read_text())
    require(r.get('version')==VERSION and r.get('status')=='passed' and r.get('mode')=='calibration'
        and r.get('dataset')=='textvqa' and r.get('completed_questions')==5 and r.get('completed_predictions')==155
        and r.get('source_sha256')==signature(),'Incomplete/outdated calibration')
    return r


def load_inputs(args):
    if args.dataset=='textvqa':
        from rq2_staged_textvqa_protocol import validate_inputs
        _,data,_=validate_inputs(args.inputs);manifest=args.inputs/'textvqa/manifest.json'
        cases=data['cases'][:5] if args.mode=='calibration' else data['cases']
    else:
        from rq2_staged_gqa_protocol import validate_inputs,selected_ids
        profile,data,_=validate_inputs(args.gqa_manifest);manifest=args.gqa_manifest
        ids=set(selected_ids(profile,args.shards));cases=[c for c in data['cases'] if c['case_id'] in ids]
    return data,manifest,cases


def score(row,dataset):
    return row['scoring']['consensus_score'] if dataset=='textvqa' else float(row['scoring']['correct'])


def summarize(questions,dataset):
    names=list(questions[0]['arms']) if questions else []
    summary={}
    for name in names:
        base=[q['arms']['baseline576']['scoring']['correct'] for q in questions]
        good=[q['arms'][name]['scoring']['correct'] for q in questions]
        summary[name]=dict(questions=len(questions),mean_score=float(np.mean([score(q['arms'][name],dataset) for q in questions])),
            full_credit=sum(good),APR=sum(b and g for b,g in zip(base,good))/sum(base) if sum(base) else None,
            losses=[q['case']['case_id'] for q,b,g in zip(questions,base,good) if b and not g],
            gains=[q['case']['case_id'] for q,b,g in zip(questions,base,good) if not b and g])
    return summary


def run(args,report,data,cases,loaded):
    import torch
    from PIL import Image
    import rq2_staged_native as staged
    from selector_grid_native import make_input,adapter
    tokenizer,model,processor=loaded
    require(len(model.model.layers)==32,'Require32-block native LLaVA')
    reference=json.loads(REFERENCE.read_text())['cases'] if args.dataset=='textvqa' else {}
    names=conditions(args.dataset)
    report.update(conditions=names,case_ids=[c['case_id'] for c in cases],expected_predictions=len(cases)*len(names),
                  completed_questions=0,completed_predictions=0,questions=[])
    for folder in ('questions','images','properties'):(args.output/folder).mkdir(exist_ok=True)
    completed=[]
    with (args.output/'predictions.jsonl').open('x') as stream:
        for index,case in enumerate(cases):
            cid,key=case['case_id'],case['image_sha256'];seed=case_seed(case,590)
            image=save_image(args,data,case);pixels,canvas=prepare_pixels(model,processor,image)
            Image.fromarray(canvas).save(args.output/'images'/(key+'-canvas.png'))
            raw=[]
            def vision_hook(module,inputs,output):raw.append(output[0][:,1:].detach().clone())
            hook=model.get_vision_tower().register_forward_hook(vision_hook)
            try:full_obs,full=staged.capture_input(model,pixels,seed,'baseline576',key)
            finally:hook.remove()
            require(len(raw)==1,'Expected one CLIP capture');raw=raw[0]
            native_obs,native=staged.capture_input(model,pixels,seed,'native-96-to-32',key)
            require(full_obs['source_patches']==native_obs['source_patches'],'Reference CLIP source differs')
            props=dict(padding=padding_fraction(*image.size),detail=detail_without_padding(pixels.detach().cpu().numpy(),*image.size),
                       errors=np.asarray(native_obs['input']['errors']))
            offline=dict(status='passed',source_patches=full_obs['source_patches'])
            if cid in reference:
                old=reference[cid]
                require(old['image_sha256']==key and old['source_patches_sha256']==full_obs['source_patches']['sha256'],'Offline reference image/features changed')
                require(np.array_equal(props['padding'],old['padding']) and np.allclose(props['detail'],old['detail'],rtol=1e-12,atol=1e-12), 'CPU pixel properties differ')
                for selector,label in [('detail','expected_detail_top96'),('error','expected_E_top96')]:
                    require(choose(selector,96,props['padding'],props['detail'],props['errors'],key,0,'input')==old[label], 'Offline top96 differs: '+selector)
                offline['archived_CPU_top96_exact']=True
            prop_path=args.output/'properties'/(key+'.json')
            write_json(prop_path,dict(image_sha256=key,image_size=list(image.size),**{k:v.tolist() for k,v in props.items()},offline=offline))
            q=dict(case=case,status='running',arms={},controls=dict(offline=offline),historical_drift={})
            qpath=args.output/'questions'/(cid.replace(':','-')+'.json');write_json(qpath,q)
            for condition in names:
                print(f'{index+1}/{len(cases)} {cid}: {condition}',flush=True)
                if not condition.startswith('grid-'):
                    snap=full if condition=='baseline576' else native
                    row,layer=staged.generate_condition(tokenizer,model,processor,image,case,pixels,condition,snap,seed)
                else:
                    spec=specification(condition)
                    selected=choose(spec['selector'],96,props['padding'],props['detail'],props['errors'],key,spec['random_seed'],'input')
                    with adapter(props,selected):
                        obs,snap=make_input(model,pixels,condition,seed,key,full,native,raw,props)
                        require(obs['selected_original_positions']==selected,'Input selection mismatch')
                        row,layer=staged.generate_condition(tokenizer,model,processor,image,case,pixels,condition,snap,seed)
                        if index==0 and condition in ('grid-random-drop-s101','grid-error-merge-s0'):
                            repeat,_=staged.generate_condition(tokenizer,model,processor,image,case,pixels,condition,snap,seed,decoder_replay=layer)
                            q['controls'][condition+'-replay']=compare_replay(row,repeat)
                            probe,_=staged.generate_condition(tokenizer,model,processor,image,case,pixels,condition,snap,seed,probe=True)
                            q['controls'][condition+'-cache']=probe
                    # Historical pure-random indices must remain exactly reproducible.
                    if spec['selector']=='random' and spec['compensation']=='drop':
                        from rq_next_seed_protocol import random_plan
                        plan=random_plan(key,spec['random_seed'],96,32)
                        require(selected==plan['input_original_positions'] and row['decoder']['selected_original_positions']==plan['decoder_original_positions'],'Pure-random mask changed')
                if cid in reference:
                    historic=condition
                    if condition.startswith('grid-random-drop-'):historic='pure-random-96-to-32-'+condition.rsplit('-',1)[1]
                    previous=reference[cid]['historical'].get(historic)
                    if previous:
                        q['historical_drift'][condition]=dict(generated_ids_equal=row['generation_tokens']['generated_token_ids']==previous['generated_ids'],
                            previous=previous['response'],current=row['response'],score_delta=score(row,'textvqa')-previous['scoring']['consensus_score'])
                q['arms'][condition]=row;stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                report['completed_predictions']+=1;write_json(qpath,q)
                del layer
            if index==0:
                identity='pure-random-576-to-576-s101'
                _,snap=staged.capture_input(model,pixels,seed,identity,key)
                row,_=staged.generate_condition(tokenizer,model,processor,image,case,pixels,identity,snap,seed)
                q['controls']['identity576']=compare_replay(q['arms']['baseline576'],row)
            q['merge_pair_masks']={}
            for name,row in q['arms'].items():
                if '-merge-' not in name:continue
                drop=name.replace('-merge-','-drop-');other=q['arms'][drop]
                require(row['input_encoding']['selected_original_positions']==other['input_encoding']['selected_original_positions'],'Stage1 paired mask differs')
                q['merge_pair_masks'][name]=dict(input_equal=True,decoder_equal=row['decoder']['selected_compact_positions']==other['decoder']['selected_compact_positions'],
                    decoder_overlap=len(set(row['decoder']['selected_compact_positions'])&set(other['decoder']['selected_compact_positions'])))
            q['status']='passed';write_json(qpath,q);completed.append(q)
            report['completed_questions']+=1;report['questions'].append(dict(case_id=cid,status='passed'));write_json(args.output/'report.json',report)
            del full,native,snap,raw
    report['summary']=summarize(completed,args.dataset)
    report['historical_changed_ids']={name:[q['case']['case_id'] for q in completed if name in q['historical_drift'] and not q['historical_drift'][name]['generated_ids_equal']]
        for name in names}
    require(report['completed_predictions']==report['expected_predictions'],'Incomplete grid')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--check',action='store_true')
    p.add_argument('--mode',choices=('calibration','full'),default='calibration');p.add_argument('--dataset',choices=('textvqa','gqa'),default='textvqa')
    for name in ('inputs','gqa-manifest','calibration-report','prerequisite-report','setup-log','runtime-report'):p.add_argument('--'+name,type=Path)
    p.add_argument('--shards',type=int,nargs='+',default=[0]);p.add_argument('--expected-gpus',type=int,default=2)
    p.add_argument('--model',default=MODEL_ID);p.add_argument('--vision-model',default=VISION_ID)
    p.add_argument('--cache-dir',type=Path,default=Path('/tmp/apet-original-hf-cache'));p.add_argument('--download-budget-gib',type=float,default=20)
    args=p.parse_args(argv)
    for k,v in vars(args).items():
        if isinstance(v,Path):setattr(args,k,v.resolve())
    if args.output.exists() or args.output.with_suffix('.zip').exists():p.error('Choose fresh output path')
    if args.check:return check(args.output)
    if args.dataset=='gqa' and args.mode!='full':p.error('Calibrate TextVQA first; GQA runs use that passed calibration')
    if any(getattr(args,k) is None for k in ('prerequisite_report','setup_log','runtime_report')):p.error('Missing prerequisite/runtime evidence')
    if (args.dataset=='textvqa' and args.inputs is None) or (args.dataset=='gqa' and args.gqa_manifest is None):p.error('Missing dataset')
    input_root=args.inputs if args.dataset=='textvqa' else args.gqa_manifest.parent
    if args.output==input_root or input_root in args.output.parents:p.error('Output must be outside inputs')
    args.output.mkdir(parents=True);signal.signal(signal.SIGTERM,interrupt_run);started=time.perf_counter()
    report=dict(version=VERSION,mode=args.mode,dataset=args.dataset,status='running',
        limits='No-basis research arms; fresh native reference. E recomputed after block16; merge/drop total effect includes possible later mask changes. No OCR supervision; no speed claim.')
    try:
        report.update(provenance('No-basis selector grid'));report['status']='running'
        validate_runtime(report);verify_native_reference(report)
        report['source_sha256']=signature();report['protocol']=json.loads((REPO/'research/protocols/selector_grid_v1.json').read_text());proof=json.loads(args.prerequisite_report.read_text())
        require(proof.get('status')=='passed' and proof.get('skipped')==0 and proof.get('modules')==list(TESTS) and proof.get('source_sha256')==signature(),'Run current zero-skip prerequisites')
        if args.mode=='full':report['calibration']=validate_calibration(args.calibration_report)
        data,args.manifest,cases=load_inputs(args)
        evidence=args.output/'evidence';evidence.mkdir()
        for name in signature():
            dest=evidence/'sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
        for source,name in [(args.prerequisite_report,'prerequisites.json'),(args.runtime_report,'runtime.json'),(args.setup_log,'setup.log'),(args.manifest,'manifest.json'),(REPO/'environments/original/uv.lock','uv.lock')]:shutil.copy2(source,evidence/name)
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count()==args.expected_gpus,'Require T4x2')
        run(args,report,data,cases,load_model(args,report));report['status']='passed'
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc());print(report['traceback'],flush=True)
    finally:
        report.update(wall_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.output/'report.json',report)
        write_json(args.output/'artifact-sha256.json',{str(f.relative_to(args.output)):sha256(f) for f in args.output.rglob('*') if f.is_file()})
        with zipfile.ZipFile(args.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(args.output.rglob('*')):
                if f.is_file():z.write(f,f.relative_to(args.output))
        print('Status:',report['status'],'Download:',args.output.with_suffix('.zip'),flush=True)
    return int(report['status']!='passed')

if __name__=='__main__':raise SystemExit(main())
