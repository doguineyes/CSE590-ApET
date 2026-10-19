"""Matched E/random x merge/no-merge panel; optional one-image Brolio oracle."""
import argparse
from datetime import datetime, timezone
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_original_env import provenance, validate_runtime, verify_native_reference, REPO
from rq1_protocol import sha256, write_json
from rq2_protocol import source_hashes, require
from rq2_frozen_protocol import REPEAT_CASE_IDS
from rq2_crossed_protocol import (VERSION, PROFILE, CONDITIONS, RANDOM_SEEDS, ORACLE_CASE,
    ORACLE_TARGET, FIXED_REPEAT_CONDITIONS, FRESH_CONDITIONS, load_profile, validate_bundle, panel, summarize)
from rq2_native import assert_selection_equal
from run_rq1_input import load_model, save_image, interrupt_run
from run_rq2_input_merge import copy_evidence
from run_rq2_input_merge_v2 import (save_bank, row_repeat_check,
                                    calibration_identity as v2_identity)
from smoke_original_apet import MODEL_ID, VISION_ID
from summarize_rq1_pilot import write_csv

MEASUREMENT = ('scripts/rq2_frozen_native.py', 'scripts/rq2_frozen_protocol.py',
    'scripts/run_rq2_input_merge.py', 'scripts/run_rq2_input_merge_v2.py',
    'scripts/rq2_native.py', 'scripts/rq2_parity_diagnostic.py',
    'scripts/rq2_crossed_protocol.py', 'scripts/rq2_crossed_native.py',
    'scripts/run_rq2_crossed.py', 'research/protocols/rq2_input_merge_v2.json',
    'research/protocols/rq2_crossed_input_v1.json')
AUXILIARY = ('research/tests/test_rq2_crossed_protocol.py',
    'research/tests/test_rq2_crossed_native.py', 'research/tests/test_rq2_crossed_runner.py',
    'research/tests/test_rq2_native.py', 'research/tests/test_rq2_frozen_native.py',
    'research/notebooks/kaggle_rq2_crossed.ipynb', 'research/RQ2_CROSSED.md')


def measurement_hashes():
    return source_hashes(extra=MEASUREMENT)


def identity(args, report, profile):
    data = v2_identity(args, report, {**profile, 'calibration': {'case_ids': profile['calibration_case_ids']}})
    data.update(protocol_version=VERSION, profile_sha256=sha256(PROFILE),
                measurement_source_sha256=measurement_hashes())
    return data


def validate_calibration(path, args):
    require(path is not None and path.is_file(), 'Full crossed panel needs the passed crossed calibration report')
    profile = load_profile(); r = json.loads(path.read_text()); ids = profile['calibration_case_ids']
    require(r.get('version') == VERSION and r.get('mode') == 'calibration'
            and r.get('status') == 'passed' and r.get('calibration_passed') is True,
            'Expected a passed crossed calibration, not v2 or an oracle report')
    require(r.get('measurement_sources_sha256') == measurement_hashes() and r.get('profile_sha256') == sha256(PROFILE)
            and r.get('arguments', {}).get('model') == args.model
            and r.get('arguments', {}).get('vision_model') == args.vision_model,
            'Crossed calibration source/profile/checkpoint identity changed')
    require(r.get('expected_case_ids') == ids and [q['case_id'] for q in r.get('questions', [])] == ids
            and all(q['status'] == 'passed' for q in r['questions'])
            and r.get('progress') == {'completed_questions': len(ids), 'expected_questions': len(ids)},
            'Crossed calibration incomplete')
    for field, prefix in [('evidence_sha256','evidence/'), ('question_sha256',''), ('feature_sha256','')]:
        require(bool(r.get(field)), 'Calibration evidence map missing')
        for name, digest in r[field].items():
            p = (path.parent / prefix / name).resolve()
            require(path.parent.resolve() in p.parents and p.is_file() and sha256(p) == digest,
                    'Missing/changed crossed calibration evidence: '+name)
    rows = []
    for cid in ids:
        name = 'questions/'+cid.replace(':','-')+'.json'
        require(name in r['question_sha256'], 'Calibration question missing hash')
        q = json.loads((path.parent/name).read_text())
        require(q['status'] == 'passed' and q['case_id'] == cid and set(q['arms']) == set(CONDITIONS)
                and q['controls']['archived_source_exact'] is True
                and q['controls']['same_image_frozen_reuse_exact'] is True
                and q['controls']['baseline_reference_max_delta'] == 0
                and q['arms']['baseline576']['scoring']['correct'] is True, 'Calibration source/baseline controls failed')
        require(set(q['controls']['paired_selection']) == {'E', *map(str, RANDOM_SEEDS)}
                and all(v['status'] == 'passed' for v in q['controls']['paired_selection'].values()),
                'Calibration selection-pair checks missing')
        for condition, row in q['arms'].items():
            require(row['condition'] == condition and row['case_id'] == cid, 'Calibration row identity changed')
            for kind in ('generation','reference_scoring'):
                require(row['frozen_replay'][kind]['status'] == 'passed'
                        and row['frozen_replay'][kind]['image_prefill_calls'] == 1, 'Calibration replay failed')
            rows.append(row)
        if cid in REPEAT_CASE_IDS:
            require(set(q['controls']['fixed_replay_repeats']) == set(FIXED_REPEAT_CONDITIONS)
                    and all(v['status'] == 'passed' for v in q['controls']['fixed_replay_repeats'].values()),
                    'Calibration fixed-tuple repeats missing')
            fresh = q['controls']['fresh_native_repeats']
            require(len(fresh) == 4
                    and sorted(v['condition'] for v in fresh) == sorted(list(FRESH_CONDITIONS)*2)
                    and all(v['comparison']['identity_controls']['status'] == 'passed' for v in fresh),
                    'Calibration fresh native controls missing')
    require(summarize(rows) == r.get('summaries'), 'Crossed calibration summary differs from its rows')
    return r


def write_feature(args, report, name, snapshot):
    import torch
    path = args.output/'features'/name
    torch.save({k: tuple(t.detach().cpu() for t in value) if k == 'encoded'
                else value.detach().cpu() if k in ('pre_merge','post_merge') else value
                for k,value in snapshot.items()}, path)
    relative = str(path.relative_to(args.output)); report['feature_sha256'][relative] = sha256(path)
    return relative


def run_oracle(args, report, item, loaded, image, base, snapshots):
    from rq2_crossed_native import oracle_snapshot, matched_oracle_control
    from rq2_frozen_native import generate_from_snapshot
    tokenizer,model,processor = loaded; case = item['case']; archived = item['reference']; seed = archived['seed']
    original = snapshots['E-merge']; control = matched_oracle_control(original['capture']['encoding'])
    out = {'case_id': case['case_id'], 'kind': 'post-hoc oracle diagnostic, not a practical rule',
           'prespecified_target': ORACLE_TARGET, 'matched_control_target': control, 'rows': {}}
    for label,target in [('oracle-sham',None), ('oracle-restore-317',ORACLE_TARGET), ('oracle-matched-control',control)]:
        modified = oracle_snapshot(model, base['pixels'], original, seed, target)
        write_feature(args, report, label+'.pt', modified)
        row = generate_from_snapshot(tokenizer,model,processor,image,case,base['pixels'],
            'native-input288',modified,seed,archived['measurement_target'])
        row.update(condition=label, arm=label, native_replay_arm='native-input288', oracle=modified['capture']['oracle'])
        out['rows'][label] = row
        print(label+': '+repr(row['response']),flush=True)
    require(out['rows']['oracle-sham']['encoding'] == snapshots['E-merge']['capture']['encoding'],
            'Oracle sham changed the original encoding')
    write_json(args.output/'oracle.json',out);report['oracle'] = out
    report['oracle_sha256'] = sha256(args.output/'oracle.json')


def run_panel(args, report, profile, manifests, records, loaded):
    from PIL import Image
    from rq1_native import encode_image, max_score_delta
    from rq2_crossed_native import capture_condition, generate_condition
    from rq2_frozen_native import compare_realizations
    tokenizer,model,processor = loaded; selected = panel(records,args.mode)
    report.update(expected_case_ids=[r['case']['case_id'] for r in selected], questions=[],
        question_sha256={},feature_sha256={},image_bank={}, progress={'completed_questions':0,'expected_questions':len(selected)})
    for folder in ('questions','features'): (args.output/folder).mkdir()
    bank = {}; rows = []
    last_use = {r['case']['image_sha256']: i for i,r in enumerate(selected)}
    with (args.output/'predictions.jsonl').open('x',encoding='utf-8') as stream:
        for position,item in enumerate(selected):
            case,archived = item['case'],item['reference'];cid=case['case_id'];key=case['image_sha256'];seed=archived['seed']
            args.manifest=args.inputs/cid.split(':')[0]/'manifest.json';image=save_image(args,manifests[cid.split(':')[0]],case)
            q={'case_id':cid,'status':'running','case':item['rq2'],'arms':{},'controls':{}}
            path=args.output/'questions'/(cid.replace(':','-')+'.json');write_json(path,q)
            report['active_case']=cid
            try:
                conditions = list(CONDITIONS) if args.mode != 'oracle' else ['baseline576','E-merge','E-no-merge']
                if key not in bank:
                    base=encode_image(model,processor,image,seed)
                    Image.fromarray(base['canvas']).save(args.output/'images'/(key+'-canvas.png'))
                    observations,snapshots={},{}
                    for condition in conditions:
                        print(cid+': capture '+condition,flush=True)
                        observations[condition],snapshots[condition]=capture_condition(model,base['pixels'],seed,condition,
                            observations.get('E-merge'),key)
                        q['capture_prefix']=observations
                        write_json(path,q)
                    paired={'E':assert_selection_equal(observations['E-merge'],observations['E-no-merge'])}
                    if args.mode != 'oracle':
                        paired.update({str(s):assert_selection_equal(observations[f'random-{s}-merge'],observations[f'random-{s}-no-merge']) for s in RANDOM_SEEDS})
                    meta=save_bank(args,report,key,seed,observations,snapshots)
                    bank[key]={'base':base,'seed':seed,'observations':observations,'snapshots':snapshots,'meta':meta,'paired':paired}
                    report['image_bank'][key]=meta
                    q.pop('capture_prefix',None)
                cache=bank[key];base=cache['base'];snapshots=cache['snapshots'];observations=cache['observations']
                require(cache['seed']==seed and all(base[k]==archived[k] for k in ('seed','clip_sha256','projected_sha256')),
                        'Crossed archived image source/seed changed')
                require(all(row['source_clip']['sha256']==archived['clip_sha256'] for row in observations.values())
                        and observations['baseline576']['encoded']['features']['sha256']==archived['projected_sha256'],
                        'Crossed source/baseline features differ')
                reference=observations['E-merge']
                require(reference['input']['errors']==item['feature']['diagnostics']['input']['errors']
                        and reference['input']['basis_indices']==item['feature']['diagnostics']['input']['basis_indices'],
                        'Crossed archived E/FPS changed')
                q['controls'].update(archived_source_exact=True,paired_selection=cache['paired'],same_image_frozen_reuse_exact=True)
                q['feature_bank']=cache['meta']
                for condition in conditions:
                    row=generate_condition(tokenizer,model,processor,image,case,base['pixels'],condition,snapshots[condition],seed,archived['measurement_target'])
                    q['arms'][condition]=row;write_json(path,q)
                    require(row['encoding']==observations[condition] and row['reference_score']['prompt']==archived['prompt']
                            and row['reference_score']['target_token_ids']==archived['target_token_ids'], 'Crossed tuple/target identity changed')
                full=q['arms']['baseline576']
                require(full['scoring']['correct'] and full['generation_tokens']['generated_token_ids'][:len(archived['target_token_ids'])]==archived['target_token_ids'],
                        'Fresh full baseline no longer correct; no retargeting')
                q['controls']['baseline_reference_max_delta']=max_score_delta(full['reference_score'],archived['controls']['reference'])
                require(q['controls']['baseline_reference_max_delta']==0,'Archived full reference drift')
                if args.mode != 'oracle' and cid in REPEAT_CASE_IDS:
                    q['controls']['fixed_replay_repeats']={}
                    for condition in FIXED_REPEAT_CONDITIONS:
                        again=generate_condition(tokenizer,model,processor,image,case,base['pixels'],condition,snapshots[condition],seed,archived['measurement_target'])
                        q['controls']['fixed_replay_repeats'][condition]=row_repeat_check(q['arms'][condition],again)
                    q['controls']['fresh_native_repeats']=[]
                    for condition in FRESH_CONDITIONS:
                        for i in range(2):
                            obs,snap=capture_condition(model,base['pixels'],seed,condition,reference,key)
                            feature=write_feature(args,report,cid.replace(':','-')+'-'+condition+f'-fresh-{i+1}.pt',snap)
                            evidence={'condition':condition,'repeat':i+1,'encoding':obs,'feature_file':feature,'comparison':{'status':'running'}}
                            q['controls']['fresh_native_repeats'].append(evidence)
                            write_json(path,q)  # Preserve the actual candidate even if structural comparison fails.
                            fresh=generate_condition(tokenizer,model,processor,image,case,base['pixels'],condition,snap,seed,archived['measurement_target'])
                            evidence['row']=fresh;write_json(path,q)
                            comparison=compare_realizations(observations[condition],snapshots[condition],obs,snap,
                                q['arms'][condition]['reference_score'],fresh['reference_score'])
                            comparison['generated_ids_equal']=fresh['generation_tokens']['generated_token_ids']==q['arms'][condition]['generation_tokens']['generated_token_ids']
                            evidence['comparison']=comparison
                            write_json(path,q)
                            del snap
                if args.mode == 'oracle':
                    run_oracle(args,report,item,loaded,image,base,snapshots)
                    sham=report['oracle']['rows']['oracle-sham'];row_repeat_check(q['arms']['E-merge'],sham)
                q['status']='passed';write_json(path,q)
                for row in q['arms'].values():
                    stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');rows.append(row)
                stream.flush();report['questions'].append({'case_id':cid,'status':'passed'})
                report['question_sha256'][str(path.relative_to(args.output))]=sha256(path)
                report['progress']['completed_questions']=len(report['questions'])
                if args.mode != 'oracle':report['summaries']=summarize(rows)
                write_json(args.output/'report.json',report)
                print('Completed '+cid+': '+' / '.join(k+'='+repr(v['response']) for k,v in q['arms'].items()),flush=True)
                if last_use[key] == position:
                    del bank[key]
                    del cache,base,snapshots
            except BaseException as error:
                q.update(status='failed',error=f'{type(error).__name__}: {error}');write_json(path,q);raise
    report.pop('active_case',None)


def finish_run(output,report,started):
    report.update(wall_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output/'report.json',report)
    rows=[]
    for path in sorted((output/'questions').glob('*.json')):
        q=json.loads(path.read_text())
        for name,row in q.get('arms',{}).items():
            rows.append({'case_id':q['case_id'],'status':q['status'],'condition':name,
                'question':q['case']['question'],'target':q['case']['canonical_target'],
                'response':row['response'],'correct':row['scoring']['correct'],
                'consensus_score':row['scoring'].get('consensus_score'),
                'reference_mean_logprob':row['reference_score']['mean_logprob']})
    if rows:write_csv(output/'answers.csv',rows)
    if report.get('oracle'):
        write_csv(output/'oracle-answers.csv',[{'condition':name,'target':row['oracle']['target'],
            'response':row['response'],'correct':row['scoring']['correct'],
            'reference_mean_logprob':row['reference_score']['mean_logprob']}
            for name,row in report['oracle']['rows'].items()])
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>RQ2 selection and merging</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:1em}pre{white-space:pre-wrap}</style>'
        '<h1>RQ2: E/random selection with and without merging</h1><p>Mode: '+html.escape(report['mode'])+
        '; execution status: '+html.escape(report['status'])+'</p><p>Native FPS source features and spatial basis replacement '
        'slots remain fixed. Random subsets use three prespecified seeds; each subset is reused for its merge/no-merge pair. '
        'Native source is unchanged. This curated panel is not a population accuracy or speed benchmark.</p>'
        '<p><a href="report.json">Provenance, exact controls and seed-wise effects</a>'+
        (' · <a href="answers.csv">Primary answers CSV</a>' if rows else '')+
        (' · <a href="oracle.json">Separate post-hoc Brolio oracle</a>' if report.get('oracle') else '')+'</p>'
        '<pre>'+html.escape(json.dumps(report.get('summaries',report.get('oracle',{})),indent=2))+'</pre>',encoding='utf-8')
    write_json(output/'artifact-sha256.json',{str(p.relative_to(output)):sha256(p) for p in output.rglob('*') if p.is_file()})
    with zipfile.ZipFile(output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob('*')):
            if path.is_file():bundle.write(path,path.relative_to(output))
    print('Status:',report['status'],'Download:',output.with_suffix('.zip'),flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument('--calibration',action='store_true');modes.add_argument('--calibration-report',type=Path)
    modes.add_argument('--oracle-only',action='store_true')
    p.add_argument('--model',default=MODEL_ID);p.add_argument('--vision-model',default=VISION_ID)
    p.add_argument('--cache-dir',type=Path,default=Path('/tmp/apet-original-hf-cache'))
    p.add_argument('--download-budget-gib',type=float,default=0);p.add_argument('--require-cuda',action='store_true')
    p.add_argument('--expected-gpus',type=int,default=2);p.add_argument('--setup-log',type=Path);p.add_argument('--runtime-report',type=Path)
    args=p.parse_args(argv)
    for key,value in vars(args).items():
        if isinstance(value,Path):setattr(args,key,value.resolve())
    args.mode='calibration' if args.calibration else 'oracle' if args.oracle_only else 'full'
    if args.output.exists() or args.output.with_suffix('.zip').exists():p.error('Preserve existing output; choose a new path')
    if args.output==args.inputs or args.inputs in args.output.parents:p.error('Output must be outside accepted inputs')
    if args.expected_gpus<1 or not math.isfinite(args.download_budget_gib) or args.download_budget_gib<0:p.error('Invalid GPU count/download budget')
    os.environ['MPLBACKEND']='Agg';os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES']='0';signal.signal(signal.SIGTERM,interrupt_run)
    args.output.mkdir(parents=True);started=time.perf_counter()
    report={'version':VERSION,'mode':args.mode,'status':'running','diagnostic_only':args.mode=='oracle',
        'calibration_passed':False,'scientific_pilot_passed':False,'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    try:
        report.update(provenance('Crossed E/random x merge/no-merge, with fixed native basis slots and frozen image realizations'))
        report['status']='running';validate_runtime(report);verify_native_reference(report)
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count()==args.expected_gpus,'Enable the expected CUDA GPUs')
        profile,index,manifests,records=validate_bundle(args.inputs)
        report.update(profile=profile,profile_sha256=sha256(PROFILE),measurement_sources_sha256=measurement_hashes(),
            conditions=CONDITIONS,reference_index_sha256=profile['reference_index_sha256'])
        prerequisite=validate_calibration(args.calibration_report,args) if args.mode=='full' else None
        copy_evidence(args,report)
        report['auxiliary_sources_sha256']={name:sha256(REPO/name) for name in AUXILIARY}
        for name in AUXILIARY:
            dest=args.output/'evidence/sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dest)
        if prerequisite:
            dest=args.output/'evidence/prerequisite-calibration';dest.mkdir()
            shutil.copy2(args.calibration_report,dest/'report.json')
            for field,prefix in [('evidence_sha256','evidence/'),('question_sha256',''),('feature_sha256','')]:
                for name in prerequisite[field]:
                    target=dest/prefix/name;target.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copy2(args.calibration_report.parent/prefix/name,target)
            report['calibration_prerequisite']={'passed':True,'report_sha256':sha256(args.calibration_report)}
        report['evidence_sha256']={str(f.relative_to(args.output/'evidence')):sha256(f) for f in (args.output/'evidence').rglob('*') if f.is_file()}
        write_json(args.output/'report.json',report)
        print('Loading LLaVA once for crossed '+args.mode,flush=True);loaded=load_model(args,report)
        report['calibration_identity']=identity(args,report,profile)
        if prerequisite:require(prerequisite.get('calibration_identity')==report['calibration_identity'],'Crossed calibration runtime/measurement/checkpoint identity changed')
        run_panel(args,report,profile,manifests,records,loaded)
        expected=profile['calibration_case_ids'] if args.mode=='calibration' else [ORACLE_CASE] if args.mode=='oracle' else profile['case_ids']
        require(report['expected_case_ids']==expected and [q['case_id'] for q in report['questions']]==expected
                and report['progress']=={'completed_questions':len(expected),'expected_questions':len(expected)},'Incomplete crossed panel')
        report.update(status='passed',calibration_passed=args.mode=='calibration',scientific_pilot_passed=args.mode=='full')
    except KeyboardInterrupt:report.update(status='interrupted',error='Partial evidence preserved')
    except Exception as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc());print(report['traceback'],flush=True)
    finally:finish_run(args.output,report,started)
    return int(report['status']!='passed')


if __name__=='__main__':raise SystemExit(main())
