"""Original staged ApET versus matched pure random on fixed GQA development800."""
import argparse
from collections import defaultdict
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
from check_original_env import REPO, provenance, validate_runtime, verify_native_reference
from rq1_protocol import sha256, write_json
from rq2_protocol import require, source_hashes
from rq2_staged_protocol import CONDITIONS, SCHEDULES, IDENTITY_CONDITION
from rq2_staged_gqa_protocol import VERSION, PROFILE, REPEAT_IDS, TENSOR_IDS, PROBE_CASE, FRESH_CASE, validate_inputs, selected_ids, summarize, load_calibration_reference
from rq2_staged_textvqa_protocol import PROBE_CONDITIONS
from run_rq2_staged_textvqa100 import MEASUREMENT as SHARED_SOURCES, validate_bank, compact_row, save_input_bank, save_decoder, compare_replay, prepare_pixels, load_model, save_image, case_seed, interrupt_run, MODEL_ID, VISION_ID, write_csv
MEASUREMENT = (*SHARED_SOURCES, 'scripts/rq2_staged_gqa_protocol.py', 'scripts/run_rq2_staged_gqa800.py', 'scripts/summarize_rq2_staged_gqa800.py', 'research/protocols/rq2_staged_gqa800_v1.json', 'scripts/eval_original_gqa_subset.py', 'scripts/eval_original_gqa_full.py', 'scripts/prepare_gqa_subsets.py', 'scripts/prepare_gqa_full.py', 'scripts/prepare_gqa_pilot.py', 'scripts/eval_original_mmstar.py')
TESTS = ('research.tests.test_rq2_native', 'research.tests.test_rq2_staged_protocol', 'research.tests.test_rq2_staged_runner', 'research.tests.test_rq2_staged_native', 'research.tests.test_rq2_staged_gqa_protocol', 'research.tests.test_rq2_staged_gqa_runner')
AUXILIARY = tuple((m.replace('.', '/') + '.py' for m in TESTS)) + ('research/notebooks/kaggle_rq2_staged_gqa800.ipynb', 'research/RQ2_STAGED_GQA800.md', 'research/tests/test_rq2_staged_textvqa_runner.py', 'research/tests/test_rq2_random_runner.py', 'research/tests/test_rq2_frozen_native.py')

def measurement_hashes():
    return source_hashes(extra=MEASUREMENT)

def prerequisite_signature():
    return {**measurement_hashes(), **{p: sha256(REPO / p) for p in AUXILIARY}}

def verify_prerequisites(args):
    p = json.loads(args.prerequisite_report.read_text())
    require(p.get('version') == VERSION and p.get('status') == 'passed' and (p.get('test_modules') == list(TESTS)) and (p.get('skipped') == 0) and (type(p.get('tests_run')) is int) and (p['tests_run'] > 26), 'Require all current native/GQA staged prerequisites with zero skips')
    require(p.get('source_sha256') == prerequisite_signature() and p.get('setup_log_sha256') == sha256(args.setup_log) and (p.get('runtime_report_sha256') == sha256(args.runtime_report)), 'Sources/runtime changed since prerequisites')
    return p

def run_panel(args, report, profile, data, loaded):
    from PIL import Image
    from rq2_staged_native import capture_input, generate_condition, generate_raw_native
    tokenizer, model, processor = loaded
    require(len(model.model.layers) == 32, 'Expected original32-block LLaVA1.5-7B')
    ids = selected_ids(profile, args.image_shards)
    accepted = set(ids)
    groups = defaultdict(list)
    for c in data['cases']:
        if c['case_id'] in accepted:
            groups[c['image_id']].append(c)
    require(all((len(g) == 8 for g in groups.values())), 'Keep every question in requested image groups')
    report.update(image_shards=args.image_shards, expected_case_ids=ids, questions=[], question_sha256={}, feature_sha256={}, encoding_sha256={}, progress=dict(completed_questions=0, expected_questions=len(ids), completed_images=0, expected_images=len(groups), completed_primary_predictions=0, expected_primary_predictions=len(ids) * 9))
    for name in ('questions', 'features', 'encodings', 'images'):
        (args.output / name).mkdir(exist_ok=True)
    complete = []
    begin = time.perf_counter()
    with (args.output / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for image_id, cases in groups.items():
            report['active_image'] = image_id
            first = cases[0]
            key = first['image_sha256']
            seeds = {case_seed(c, 590) for c in cases}
            require(len(seeds) == 1, 'Same-image seeds changed')
            seed = seeds.pop()
            args.manifest = args.data_manifest
            image = save_image(args, data, first)
            pixels, canvas = prepare_pixels(model, processor, image)
            Image.fromarray(canvas).save(args.output / 'images' / (key + '-canvas.png'))
            observations, snapshots = ({}, {})
            names = (*CONDITIONS, IDENTITY_CONDITION) if any((c['case_id'] in profile['identity_case_ids'] for c in cases)) else tuple(CONDITIONS)
            print(f'Image {image_id}: capture once for eight questions', flush=True)
            encoding = args.output / 'encodings' / (key + '.json')
            for condition in names:
                observations[condition], snapshots[condition] = capture_input(model, pixels, seed, condition, key)
                write_json(encoding, dict(image_id=image_id, image_sha256=key, seed=seed, status='capturing', encodings=observations))
            checks = validate_bank(observations, key)
            write_json(encoding, dict(image_id=image_id, image_sha256=key, seed=seed, case_ids=[c['case_id'] for c in cases], encodings=observations, controls=checks))
            reference = dict(file=str(encoding.relative_to(args.output)), sha256=sha256(encoding))
            report['encoding_sha256'][reference['file']] = reference['sha256']
            tensors = save_input_bank(args, report, key, snapshots) if any((c['case_id'] in TENSOR_IDS for c in cases)) else None
            for case in cases:
                cid = case['case_id']
                report['active_case'] = cid
                path = args.output / 'questions' / (cid.replace(':', '-') + '.json')
                q = dict(case_id=cid, case=case, seed=seed, status='running', arms={}, controls={'input_bank': checks}, encoding_reference=reference)
                if tensors:
                    q['input_tensor_bank'] = tensors
                write_json(path, q)
                try:
                    layers = {}
                    for condition in CONDITIONS:
                        row, layers[condition] = generate_condition(tokenizer, model, processor, image, case, pixels, condition, snapshots[condition], seed)
                        row = compact_row(row, reference)
                        row.update(image_id=image_id, types=case['types'], known_repeatability_case=case['known_repeatability_case'])
                        if cid in TENSOR_IDS and layers[condition] is not None:
                            row['decoder_tensor'] = save_decoder(args, report, cid, condition, layers[condition])
                        q['arms'][condition] = row
                        write_json(path, q)
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                        stream.flush()
                        report['progress']['completed_primary_predictions'] += 1
                        print(f"{cid}: {condition} = {row['response']!r}; correct={row['scoring']['correct']}", flush=True)
                    baseline = q['arms']['baseline576']
                    require(all((r['generation']['prompt'] == baseline['generation']['prompt'] for r in q['arms'].values())), 'Condition prompts differ')
                    if cid in profile['identity_case_ids']:
                        row, _ = generate_condition(tokenizer, model, processor, image, case, pixels, IDENTITY_CONDITION, snapshots[IDENTITY_CONDITION], seed)
                        q['controls']['identity576'] = compare_replay(baseline, compact_row(row, reference))
                    if cid in REPEAT_IDS:
                        q['controls']['fixed_replays'] = {}
                        for condition in CONDITIONS:
                            row, _ = generate_condition(tokenizer, model, processor, image, case, pixels, condition, snapshots[condition], seed, decoder_replay=layers[condition])
                            row = compact_row(row, reference)
                            row['control_only'] = True
                            q['controls']['fixed_replays'][condition] = compare_replay(q['arms'][condition], row)
                    if cid == PROBE_CASE:
                        q['controls']['cached_decoding_probes'] = []
                        for condition in PROBE_CONDITIONS:
                            row, _ = generate_condition(tokenizer, model, processor, image, case, pixels, condition, snapshots[condition], seed, decoder_replay=layers[condition], probe=True)
                            q['controls']['cached_decoding_probes'].append(compact_row(row, reference))
                    if cid == FRESH_CASE:
                        q['controls']['fresh_raw_native'] = []
                        for a, b in SCHEDULES:
                            condition = f'native-{a}-to-{b}'
                            for repeat in range(2):
                                row = generate_raw_native(tokenizer, model, processor, image, case, condition, seed)
                                q['controls']['fresh_raw_native'].append(dict(repeat=repeat + 1, row=row, generated_ids_equal=row['generation_tokens']['generated_token_ids'] == q['arms'][condition]['generation_tokens']['generated_token_ids'], correct_equal=row['scoring']['correct'] == q['arms'][condition]['scoring']['correct'], policy='Fresh whole-native variation recorded; never replaces first primary realization'))
                    q['status'] = 'passed'
                    write_json(path, q)
                    report['question_sha256'][str(path.relative_to(args.output))] = sha256(path)
                    complete.extend(q['arms'].values())
                    report['questions'].append(dict(case_id=cid, status='passed'))
                    report['progress']['completed_questions'] += 1
                    write_json(args.output / 'report.json', report)
                    n = report['progress']['completed_questions']
                    eta = (time.perf_counter() - begin) / n * (len(ids) - n)
                    print(f'Completed {n}/{len(ids)} questions; rough remaining {eta / 60:.1f} min (includes observed capture/control work)', flush=True)
                    del layers
                except BaseException as error:
                    q.update(status='failed', error=f'{type(error).__name__}: {error}')
                    write_json(path, q)
                    raise
            report['progress']['completed_images'] += 1
            write_json(args.output / 'report.json', report)
            del observations, snapshots, pixels
    report['summaries'] = summarize(complete, expected_ids=ids, intervals=True)
    report.pop('active_case', None)
    report.pop('active_image', None)

def copy_evidence(args, report, study):
    root = args.output / 'evidence'
    files = {**{str(REPO / p): 'sources/' + p for p in (*measurement_hashes(), *AUXILIARY)}, str(args.data_manifest): 'inputs/development/manifest.json', str(study): 'inputs/study.json', str(args.setup_log): 'notebook/setup.log', str(args.runtime_report): 'notebook/runtime.json', str(args.prerequisite_report): 'notebook/prerequisites.json'}
    for source, name in files.items():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    report['evidence_sha256'] = {str(p.relative_to(root)): sha256(p) for p in root.rglob('*') if p.is_file()}

def complete_image_rows(questions):
    """Keep unfinished images in raw records while excluding them from summaries."""
    groups = defaultdict(list)
    for q in questions:
        groups[q['case']['image_id']].append(q)
    return [row for g in groups.values() if len(g) == 8 and all((q['status'] == 'passed' for q in g)) for q in g for row in q['arms'].values()]

def finish_run(output, report, started):
    questions = [json.loads(p.read_text()) for p in sorted((output / 'questions').glob('*.json'))]
    complete = complete_image_rows(questions)
    if complete and 'summaries' not in report:
        report['summaries'] = summarize(complete)
    report['summary_coverage'] = dict(complete_image_questions=len(complete) // 9, passed_questions=sum((q['status'] == 'passed' for q in questions)), policy='Only complete matched eight-question image groups enter summaries; all partial raw arms are preserved')
    report.update(wall_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
    report['question_sha256'] = {str(p.relative_to(output)): sha256(p) for p in (output / 'questions').glob('*.json')}
    write_json(output / 'report.json', report)
    answers = [dict(case_id=q['case_id'], image_id=q['case']['image_id'], case_status=q['status'], question=q['case']['question'], gold=q['case']['canonical_target'], condition=name, response=r['response'], correct=r['scoring']['correct'], seed=r['seed'], random_seed=r['random_seed'], input_tokens=r['input_tokens'], after_block16_tokens=r['decoder_tokens'] or 576) for q in questions for name, r in q.get('arms', {}).items()]
    if answers:
        write_csv(output / 'answers.csv', answers)
    esc = lambda x: html.escape(str(x))
    s = report.get('summaries', {})
    page = ['<!doctype html><meta charset="utf-8"><title>Staged GQA development</title>', '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:1em}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.3em}img{width:336px;max-width:100%}article{border-top:1px solid #bbb;margin-top:1em}</style>', '<h1>Staged GQA development: native ApET and matched pure random</h1><p>Status: ' + esc(report['status']) + '. Full cohort800 questions/100 images; this export covers ' + str(s.get('questions', 0)) + ' complete-image questions. Reserved evaluation is excluded.</p>', '<p><a href="report.json">Report and comparisons</a> · <a href="answers.csv">Answers</a> · <a href="predictions.jsonl">Predictions</a></p>', '<table><tr><th>Condition</th><th>Accuracy</th><th>APR</th></tr>']
    for name, v in s.get('conditions', {}).items():
        apr = format(v['APR'], '.1%') if v['APR'] is not None else 'n/a'
        page.append(f"<tr><td>{esc(name)}</td><td>{v['accuracy']:.1%}</td><td>{apr}</td></tr>")
    page.append('</table>')
    shown = set()
    for q in questions:
        c = q['case']
        key = c['image_sha256']
        if key not in shown:
            shown.add(key)
            page.append('<h2>Image ' + esc(c['image_id']) + '</h2><img loading="lazy" src="images/' + key + '-canvas.png" alt="Actual model canvas">')
        page.append('<article><h3>' + esc(q['case_id']) + '</h3><p>' + esc(c['question']) + ' · Reference: ' + esc(c['canonical_target']) + '</p><table><tr><th>Condition</th><th>Answer</th><th>Correct</th></tr>')
        for name, r in q.get('arms', {}).items():
            page.append('<tr><td>' + esc(name) + '</td><td>' + esc(r['response']) + '</td><td>' + str(r['scoring']['correct']) + '</td></tr>')
        page.append('</table></article>')
    (output / 'index.html').write_text('\n'.join(page), encoding='utf-8')
    write_json(output / 'artifact-sha256.json', {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()})
    with zipfile.ZipFile(output.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                z.write(p, p.relative_to(output))
    print('Status:', report['status'], 'Download:', output.with_suffix('.zip'), flush=True)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('data-manifest', 'output', 'setup-log', 'runtime-report', 'prerequisite-report'):
        parser.add_argument('--' + field, type=Path, required=True)
    parser.add_argument('--run-gqa', action='store_true', required=True)
    parser.add_argument('--image-shards', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--vision-model', default=VISION_ID)
    parser.add_argument('--cache-dir', type=Path, default=Path('/tmp/apet-original-hf-cache'))
    parser.add_argument('--download-budget-gib', type=float, default=0)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--expected-gpus', type=int, default=2)
    args = parser.parse_args(argv)
    for key, v in vars(args).items():
        if isinstance(v, Path):
            setattr(args, key, v.resolve())
    if args.output.exists() or args.output.with_suffix('.zip').exists():
        parser.error('Preserve prior output; choose a new path')
    if args.output == args.data_manifest.parent or args.data_manifest.parent in args.output.parents:
        parser.error('Output must be outside the input data directory')
    if args.expected_gpus < 1 or not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        parser.error('Invalid GPU count/download budget')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
    signal.signal(signal.SIGTERM, interrupt_run)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(version=VERSION, status='running', execution_passed=False, scientific_benchmark_passed=False, arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    try:
        report.update(provenance('Fixed GQA development800; staged native ApET versus matched pure random'))
        report['status'] = 'running'
        validate_runtime(report)
        verify_native_reference(report)
        profile, data, study = validate_inputs(args.data_manifest)
        ids = selected_ids(profile, args.image_shards)
        report.update(profile=profile, profile_sha256=sha256(PROFILE), conditions=CONDITIONS, reviewed_calibration=load_calibration_reference(), measurement_sources_sha256=measurement_hashes(), prerequisite=verify_prerequisites(args))
        copy_evidence(args, report, study)
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == args.expected_gpus, 'Enable expected CUDA GPUs')
        write_json(args.output / 'report.json', report)
        print(f'Validated requested development shards {args.image_shards}: {len(ids)} questions, {len(ids) * 9} primary predictions. Loading once.', flush=True)
        loaded = load_model(args, report)
        run_panel(args, report, profile, data, loaded)
        require([q['case_id'] for q in report['questions']] == ids and report['progress']['completed_primary_predictions'] == len(ids) * 9, 'Incomplete requested GQA shards')
        report.update(status='passed', execution_passed=True)
    except KeyboardInterrupt:
        report.update(status='interrupted', error='Partial raw evidence preserved')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        finish_run(args.output, report, started)
    return int(report['status'] != 'passed')
if __name__ == '__main__':
    raise SystemExit(main())
