"""TextVQA100 repeated random masks, staged 96->32; no OCR/causal claims yet."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_original_env import REPO, provenance, validate_runtime, verify_native_reference
from rq1_protocol import sha256, write_json
from rq2_protocol import require
from rq_next_seed_protocol import VERSION, SEEDS, CALIBRATION_SEEDS, adapter_policy, random_plan
from rq2_staged_textvqa_protocol import validate_inputs
from run_rq2_staged_textvqa100 import measurement_hashes as parent_hashes, verify_prerequisites
from run_rq2_staged_calibration import compare_replay
from run_rq1_input import load_model, save_image, case_seed, interrupt_run
from run_rq2_random_calibration import prepare_pixels
from smoke_original_apet import MODEL_ID, VISION_ID

HISTORY = REPO / 'research/protocols/textvqa_seed_history_v1.json'


def measurement_hashes():
    paths = ('scripts/rq_next_seed_protocol.py', 'scripts/run_textvqa_seed_coverage.py',
             'research/protocols/textvqa_seed_history_v1.json',
             'research/tests/test_textvqa_seed_coverage.py')
    return {**parent_hashes(), **{p: sha256(REPO / p) for p in paths}}


def verify_calibration(path, hashes):
    require(path is not None, 'Full run needs the reviewed calibration report')
    r = json.loads(path.read_text())
    require(r.get('version') == VERSION and r.get('mode') == 'calibration'
            and r.get('status') == 'passed' and r.get('source_sha256') == hashes
            and r.get('completed_questions') == 3 and r.get('completed_predictions') == 18,
            'Calibration incomplete or sources changed; rerun calibration')
    return r


def history_comparison(case, row, history):
    old = history['cases'][case['case_id']]
    require(old['image_sha256'] == case['image_sha256'], 'Historical image differs')
    previous = old['arms'].get(row['condition'])
    if previous is None:
        return None
    if 'random_plan' in previous:
        require(row['decoder']['random_plan'] == previous['random_plan'], 'Archived seed mask changed')
    return dict(response_equal=row['response'] == previous['response'],
        generated_ids_equal=row['generation_tokens']['generated_token_ids'] == previous['generated_token_ids'],
        previous_response=previous['response'], current_response=row['response'],
        consensus_delta=row['scoring']['consensus_score'] - previous['scoring']['consensus_score'],
        policy='Fresh-run drift recorded; never retry or replace based on correctness')


def run(args, report, data, loaded):
    import torch
    from PIL import Image
    from rq2_staged_native import capture_input, generate_condition
    from rq2_native import tensor_fingerprint
    tokenizer, model, processor = loaded
    require(len(model.model.layers) == 32, 'Expected original 32-block model')
    cases = data['cases'][:3] if args.mode == 'calibration' else data['cases']
    seeds = CALIBRATION_SEEDS if args.mode == 'calibration' else SEEDS
    conditions = ['baseline576', 'native-96-to-32'] + [f'pure-random-96-to-32-s{s}' for s in seeds]
    report.update(case_ids=[c['case_id'] for c in cases], seeds=list(seeds), conditions=conditions,
                  completed_questions=0, completed_predictions=0, expected_predictions=len(cases)*len(conditions))
    history = json.loads(HISTORY.read_text())
    for directory in ('questions', 'features', 'images'):
        (args.output / directory).mkdir(exist_ok=True)
    with adapter_policy(), (args.output / 'predictions.jsonl').open('x') as stream:
        for index, case in enumerate(cases):
            cid, key = case['case_id'], case['image_sha256']
            seed = case_seed(case, 590)
            args.manifest = args.inputs / 'textvqa/manifest.json'
            image = save_image(args, data, case)
            pixels, canvas = prepare_pixels(model, processor, image)
            Image.fromarray(canvas).save(args.output / 'images' / (key + '-canvas.png'))
            q = dict(case=case, status='running', arms={}, controls={}, historical_comparison={})
            path = args.output / 'questions' / (cid.replace(':','-') + '.json')
            write_json(path, q)
            baseline = None
            full_observation = None
            for condition in conditions:
                print(f'{index+1}/{len(cases)} {cid}: {condition}', flush=True)
                captured = []
                # Actual CLIP output before native basis overwrite/merging; hook is observational.
                def vision_hook(module, inputs, output):
                    captured.append(output[0][:, 1:].detach().cpu().clone())
                hook = model.get_vision_tower().register_forward_hook(vision_hook) if condition == 'native-96-to-32' else None
                try:
                    observation, snapshot = capture_input(model, pixels, seed, condition, key)
                finally:
                    if hook is not None:
                        hook.remove()
                if full_observation is not None:
                    require(observation['source_clip'] == full_observation['source_clip']
                            and observation['source_patches'] == full_observation['source_patches'],
                            'Fresh CLIP source changed within question')
                row, layer = generate_condition(tokenizer, model, processor, image, case, pixels,
                                                 condition, snapshot, seed)
                if condition.startswith('pure-random'):
                    require(observation['full_projected'] == full_observation['encoded']['features'],
                            'Random arm changed full projected source')
                    plan = random_plan(key, row['random_seed'], 96, 32)
                    require(observation['random_plan'] == row['decoder']['random_plan'] == plan,
                            'Input/decoder masks disagree')
                    require(observation['pure_input'] == dict(fps=False,E=False,basis_replacement=False,merging=False),
                            'Pure random unexpectedly ran native compression')
                if condition == 'native-96-to-32':
                    require(len(captured) == 1 and list(captured[0].shape) == [1,576,1024], 'CLIP capture shape changed')
                    require(tensor_fingerprint(captured[0]) == observation['source_patches'], 'Captured features differ')
                    tensor_path = args.output / 'features' / (key + '.pt')
                    torch.save(dict(pixels=pixels.detach().cpu(), clip_patches=captured[0],
                        native_layer16_visual=layer['source_visual'].detach().cpu()), tensor_path)
                    q['feature_dump'] = dict(file=str(tensor_path.relative_to(args.output)), sha256=sha256(tensor_path),
                        image_size=list(image.size), canvas_size=[336,336], grid=[24,24],
                        input_diagnostics=observation, layer16_diagnostics=row['decoder'],
                        note='Native layer16: 96 contextual states; basis slots have substituted content. Positions are not isolated pixel evidence.')
                if condition == 'baseline576':
                    baseline = row
                    full_observation = observation
                q['arms'][condition] = row
                previous = history_comparison(case, row, history)
                if previous is not None:
                    q['historical_comparison'][condition] = previous
                stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
                report['completed_predictions'] += 1
                write_json(path, q)
                if index == 0 and condition == 'pure-random-96-to-32-s404':
                    repeat, _ = generate_condition(tokenizer, model, processor, image, case, pixels,
                        condition, snapshot, seed, decoder_replay=layer)
                    q['controls']['new_seed_fixed_replay'] = compare_replay(row, repeat)
                    probe, _ = generate_condition(tokenizer, model, processor, image, case, pixels,
                        condition, snapshot, seed, probe=True)
                    q['controls']['new_seed_cache_probe'] = probe
                del snapshot, layer
            if index == 0:
                condition = 'pure-random-576-to-576-s101'
                obs, snap = capture_input(model, pixels, seed, condition, key)
                identity, _ = generate_condition(tokenizer, model, processor, image, case, pixels, condition, snap, seed)
                q['controls']['identity576'] = compare_replay(baseline, identity)
                del snap
            q['status'] = 'passed'
            write_json(path, q)
            report['completed_questions'] += 1
            write_json(args.output / 'report.json', report)
    require(report['completed_predictions'] == report['expected_predictions'], 'Incomplete predictions')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('inputs','output','setup-log','runtime-report','prerequisite-report'):
        p.add_argument('--'+field, type=Path, required=True)
    p.add_argument('--calibration-report', type=Path)
    p.add_argument('--mode', choices=('calibration','full'), default='calibration')
    p.add_argument('--model', default=MODEL_ID); p.add_argument('--vision-model', default=VISION_ID)
    p.add_argument('--cache-dir', type=Path, default=Path('/tmp/apet-original-hf-cache'))
    p.add_argument('--download-budget-gib', type=float, default=20)
    p.add_argument('--expected-gpus', type=int, default=2)
    args = p.parse_args(argv)
    for k,v in vars(args).items():
        if isinstance(v,Path): setattr(args,k,v.resolve())
    if args.output.exists() or args.output.with_suffix('.zip').exists(): p.error('Use a fresh output path')
    if args.inputs == args.output or args.inputs in args.output.parents: p.error('Output must be outside inputs')
    args.output.mkdir(parents=True)
    signal.signal(signal.SIGTERM, interrupt_run)
    started=time.perf_counter()
    report=dict(version=VERSION,mode=args.mode,status='running', arguments={k:str(v) for k,v in vars(args).items()},
        coverage_status='Not measured: OCR boxes absent. Masks alone do not establish evidence coverage or mediation.')
    try:
        report.update(provenance('TextVQA100 random-seed extension and native feature dump'))
        report['status']='running'
        validate_runtime(report); verify_native_reference(report)
        _, data, _ = validate_inputs(args.inputs)
        report['prerequisite'] = verify_prerequisites(args)
        hashes = measurement_hashes(); report['source_sha256']=hashes
        if args.mode == 'full': report['calibration']=verify_calibration(args.calibration_report,hashes)
        evidence=args.output/'evidence'
        for name in hashes:
            dest=evidence/'sources'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(REPO/name,dest)
        for source,name in [(args.inputs/'textvqa/manifest.json','manifest.json'),(args.setup_log,'setup.log'),
                            (args.runtime_report,'runtime.json'),(args.prerequisite_report,'prerequisites.json'),
                            (REPO/'environments/original/uv.lock','uv.lock')]:
            shutil.copy2(source,evidence/name)
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count()==args.expected_gpus,'Enable T4 x2')
        run(args,report,data,load_model(args,report))
        report['status']='passed'
        questions=[json.loads(f.read_text()) for f in sorted((args.output/'questions').glob('*.json'))]
        report['historical_drift']={name: [q['case']['case_id'] for q in questions
            if not q['historical_comparison'][name]['generated_ids_equal']]
            for name in questions[0]['historical_comparison']}
        report['summary']={name: dict(questions=len(questions),
            mean_consensus=sum(q['arms'][name]['scoring']['consensus_score'] for q in questions)/len(questions),
            full_credit=sum(bool(q['arms'][name]['scoring']['correct']) for q in questions))
            for name in report['conditions']}
    except (Exception,KeyboardInterrupt) as error:
        report.update(status='failed',error=str(error),traceback=traceback.format_exc()); print(report['traceback'],flush=True)
    finally:
        report['wall_seconds']=time.perf_counter()-started
        report['finished_utc']=datetime.now(timezone.utc).isoformat()
        write_json(args.output/'report.json',report)
        write_json(args.output/'artifact-sha256.json',{str(f.relative_to(args.output)):sha256(f) for f in args.output.rglob('*') if f.is_file()})
        with zipfile.ZipFile(args.output.with_suffix('.zip'),'x',zipfile.ZIP_DEFLATED) as z:
            for f in sorted(args.output.rglob('*')):
                if f.is_file(): z.write(f,f.relative_to(args.output))
        print('Status:',report['status'],'Download:',args.output.with_suffix('.zip'),flush=True)
    return int(report['status']!='passed')

if __name__=='__main__': raise SystemExit(main())
