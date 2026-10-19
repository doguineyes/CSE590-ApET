"""RQ2 v2: capture the first native realization per image; replay exact tuples.

Seven fixed calibration questions precede the 30-case panel. Independent native
repeats record uncertainty; only within-realization replay must be byte exact.
Upstream compression and the v1 protocol/failed evidence remain unchanged.
"""

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
from check_original_env import REPO, EXPECTED, provenance, validate_runtime, verify_native_reference
from rq1_protocol import sha256, write_json
from rq2_protocol import ARMS, require, source_hashes, paired_summary
from rq2_frozen_protocol import VERSION, PROFILE, load_profile, choose_panel, validate_bundle
from run_rq1_input import interrupt_run, load_model, save_image
from run_rq2_input_merge import copy_evidence
from smoke_original_apet import MODEL_ID, VISION_ID
from summarize_rq1_pilot import write_csv

MEASUREMENT_FILES = (
    'scripts/run_rq2_input_merge.py', 'scripts/rq2_native.py',
    'scripts/rq2_parity_diagnostic.py', 'scripts/rq2_frozen_native.py',
    'scripts/rq2_frozen_protocol.py', 'scripts/run_rq2_input_merge_v2.py',
    'research/protocols/rq2_input_merge_v2.json',
)
AUXILIARY_FILES = (
    'research/tests/test_rq2_native.py', 'research/tests/test_rq2_frozen_native.py',
    'research/tests/test_rq2_frozen_protocol.py', 'research/tests/test_rq2_frozen_runner.py',
    'research/notebooks/kaggle_rq2_input_merge_v2.ipynb', 'research/RQ2_INPUT_MERGE_V2.md',
)


def measurement_hashes():
    return source_hashes(extra=MEASUREMENT_FILES)


def calibration_identity(args, report, profile):
    """Stable measurement identity; download counts, timings and paths excluded."""
    checkpoints = {name: {k: row[k] for k in ('source', 'model_id', 'revision', 'path') if k in row}
                   for name, row in report.get('checkpoints', {}).items()}
    return {'protocol_version': VERSION, 'profile_sha256': sha256(PROFILE),
        'reference_index_sha256': profile['reference_index_sha256'],
        'input_protocol_sha256': sha256(args.inputs / 'protocol.json'),
        'measurement_source_sha256': measurement_hashes(),
        'native_source_sha256': report['implementation']['native_source_sha256'],
        'original_lock_sha256': report['lock_sha256'],
        'runtime_identity': {'python': report['python'], 'expected_packages': EXPECTED,
            'expected_gpus': args.expected_gpus,
            'model': {k: report.get('model', {}).get(k) for k in
                      ('class', 'module', 'attention', 'precision', 'device_map')}},
        'checkpoint_identity': {'arguments': {'model': args.model, 'vision_model': args.vision_model},
                                'resolved': checkpoints},
        'case_ids': profile['calibration']['case_ids']}


def validate_calibration(path, args):
    """Require actual hashed detailed evidence, never only a passed report flag."""
    require(path is not None and path.is_file(), 'Full v2 panel requires a passed v2 calibration report')
    profile = load_profile()
    r = json.loads(path.read_text())
    ids = profile['calibration']['case_ids']
    require(r.get('version') == VERSION and r.get('mode') == 'calibration'
            and r.get('status') == 'passed' and r.get('calibration_passed') is True
            and r.get('scientific_pilot_passed') is False,
            'Expected the completed v2 calibration, not a diagnostic, v1 pilot or full report')
    require(r.get('expected_case_ids') == ids
            and [q['case_id'] for q in r.get('questions', [])] == ids
            and all(q['status'] == 'passed' for q in r['questions'])
            and r.get('progress') == {'completed_questions': len(ids), 'expected_questions': len(ids)},
            'V2 calibration is incomplete')
    require(r.get('measurement_sources_sha256') == measurement_hashes()
            and r.get('profile_sha256') == sha256(PROFILE), 'V2 calibration measurement sources changed')
    require(r.get('arguments', {}).get('model') == args.model
            and r.get('arguments', {}).get('vision_model') == args.vision_model,
            'V2 calibration checkpoint arguments changed')
    require(r.get('implementation', {}).get('native_matches_reference') is True
            and r.get('lock_sha256') == profile['lock_sha256'], 'V2 calibration native/runtime identity differs')
    for field, prefix in [('evidence_sha256', 'evidence/'), ('question_sha256', ''),
                          ('feature_sha256', '')]:
        require(bool(r.get(field)), 'V2 calibration lacks hashed ' + field)
        for relative, digest in r[field].items():
            target = (path.parent / prefix / relative).resolve()
            require(path.parent.resolve() in target.parents and target.is_file()
                    and sha256(target) == digest, 'Missing or changed v2 calibration artifact: ' + relative)
    rows = []
    for cid in ids:
        relative = 'questions/' + cid.replace(':', '-') + '.json'
        require(relative in r['question_sha256'], 'Calibration question is not hashed')
        q = json.loads((path.parent / relative).read_text())
        require(q['case_id'] == cid and q['status'] == 'passed'
                and q['controls']['archived_source_exact'] is True
                and q['controls']['same_image_frozen_reuse_exact'] is True
                and q['controls']['selection_and_basis']['status'] == 'passed'
                and q['controls']['baseline_reference_max_delta'] == 0
                and q['arms']['baseline576']['scoring']['correct'] is True
                and set(q['arms']) == set(ARMS), 'Calibration structural controls failed')
        for arm, row in q['arms'].items():
            require(row['case_id'] == cid and row['arm'] == arm, 'Calibration arm identity differs')
            require(all(row['frozen_replay'][key]['status'] == 'passed'
                        and row['frozen_replay'][key]['image_prefill_calls'] == 1
                        for key in ('generation', 'reference_scoring')), 'Calibration replay was not exact')
            rows.append(row)
        if cid in profile['controls']['fixed_replay_repeats']:
            require(set(q['controls']['fixed_replay_repeats']) == set(ARMS)
                    and all(v['status'] == 'passed' for v in q['controls']['fixed_replay_repeats'].values()),
                    'Calibration fixed-tuple repeats missing or failed')
            fresh = q['controls']['fresh_native_repeats']
            require(fresh['status'] == 'recorded' and len(fresh['repeats']) == 2
                    and all(v['comparison']['identity_controls']['status'] == 'passed'
                            for v in fresh['repeats']), 'Calibration fresh native controls missing')
    require(paired_summary(rows) == r.get('summaries'), 'Calibration summaries differ from detailed predictions')
    return r


def row_repeat_check(first, repeated):
    from rq1_native import max_score_delta
    require(first['generation_tokens']['generated_token_ids'] == repeated['generation_tokens']['generated_token_ids']
            and first['encoding'] == repeated['encoding']
            and max_score_delta(first['reference_score'], repeated['reference_score']) == 0,
            'Fixed-tuple replay changed; preserve evidence for review')
    return {'status': 'passed', 'generated_ids_exact': True, 'frozen_encoding_exact': True,
            'reference_score_max_delta': 0.0, 'repeated': repeated}


def save_bank(args, report, key, seed, observations, snapshots):
    """Store the primary first realization BEFORE inspecting any generated answer."""
    import torch
    relative = 'features/' + key + '.pt'
    def cpu(snapshot):
        return {k: tuple(t.detach().cpu() for t in value) if k == 'encoded'
                else value.detach().cpu() if k in ('pre_merge', 'post_merge') else value
                for k, value in snapshot.items()}
    torch.save({'image_sha256': key, 'seed': seed,
                'snapshots': {arm: cpu(snapshot) for arm, snapshot in snapshots.items()}}, args.output / relative)
    report['feature_sha256'][relative] = sha256(args.output / relative)
    meta = {'image_sha256': key, 'seed': seed, 'feature_file': relative,
        'feature_sha256': report['feature_sha256'][relative], 'encodings': observations,
        'policy': 'First unconditional native realization per arm; run-local bank, shared across questions'}
    write_json(args.output / 'features' / (key + '.json'), meta)
    report['feature_sha256']['features/' + key + '.json'] = sha256(args.output / 'features' / (key + '.json'))
    return meta


def run_panel(args, report, profile, manifests, records, loaded):
    import torch
    from PIL import Image
    from rq1_native import encode_image, max_score_delta
    from rq2_native import assert_selection_equal
    from rq2_frozen_native import capture_encoding, generate_from_snapshot, compare_realizations
    tokenizer, model, processor = loaded
    panel = choose_panel(records, calibration=args.calibration)
    report.update(expected_case_ids=[r['case']['case_id'] for r in panel], questions=[],
        question_sha256={}, feature_sha256={}, image_bank={},
        progress={'completed_questions': 0, 'expected_questions': len(panel)})
    for folder in ('questions', 'features'):
        (args.output / folder).mkdir()
    bank, rows = {}, []
    with (args.output / 'predictions.jsonl').open('x', encoding='utf-8') as stream:
        for item in panel:
            case, archived = item['case'], item['reference']
            cid, key, seed = case['case_id'], case['image_sha256'], archived['seed']
            dataset = cid.split(':')[0]
            args.manifest = args.inputs / dataset / 'manifest.json'
            image = save_image(args, manifests[dataset], case)
            path = args.output / 'questions' / (cid.replace(':', '-') + '.json')
            q = {'case_id': cid, 'status': 'running', 'case': item['rq2'], 'arms': {}, 'controls': {}}
            report['active_case'] = cid
            write_json(path, q)
            try:
                if key not in bank:
                    base = encode_image(model, processor, image, seed)
                    Image.fromarray(base['canvas']).save(args.output / 'images' / (key + '-canvas.png'))
                    observations, snapshots = {}, {}
                    for arm in ARMS:
                        print(f'{cid}: first image capture {arm}', flush=True)
                        observations[arm], snapshots[arm] = capture_encoding(model, base['pixels'], seed, arm)
                    assert_selection_equal(observations['native-input288'], observations['native-input288-no-merge'])
                    require(observations['baseline576']['encoded']['features']['sha256'] == archived['projected_sha256'],
                            'Frozen full-token capture differs from archived baseline features')
                    meta = save_bank(args, report, key, seed, observations, snapshots)
                    bank[key] = {'base': base, 'seed': seed, 'observations': observations, 'snapshots': snapshots, 'meta': meta}
                    report['image_bank'][key] = meta
                    write_json(args.output / 'report.json', report)
                cached = bank[key]
                require(cached['seed'] == seed, 'Same-image archived seed differs')
                base, snapshots, observations = cached['base'], cached['snapshots'], cached['observations']
                require(all(base[k] == archived[k] for k in ('clip_sha256', 'projected_sha256', 'seed')),
                        'Archived source identity changed')
                require(all(row['source_clip']['sha256'] == archived['clip_sha256'] for row in observations.values()),
                        'Captured CLIP features differ from archived source')
                native = observations['native-input288']
                require(native['input']['errors'] == item['feature']['diagnostics']['input']['errors']
                        and native['input']['basis_indices'] == item['feature']['diagnostics']['input']['basis_indices'],
                        'Archived E/FPS changed')
                q['controls']['archived_source_exact'] = True
                q['controls']['selection_and_basis'] = assert_selection_equal(native, observations['native-input288-no-merge'])
                q['feature_bank'] = cached['meta']
                for arm in ARMS:
                    print(f'{cid}: frozen replay {arm}', flush=True)
                    row = generate_from_snapshot(tokenizer, model, processor, image, case, base['pixels'],
                        arm, snapshots[arm], seed, archived['measurement_target'])
                    q['arms'][arm] = row
                    write_json(path, q)
                    require(row['encoding'] == observations[arm], 'Same-image frozen observation changed')
                    require(row['reference_score']['prompt'] == archived['prompt']
                            and row['reference_score']['target_token_ids'] == archived['target_token_ids'], 'Frozen target/prompt changed')
                baseline = q['arms']['baseline576']
                require(baseline['scoring']['correct']
                        and baseline['generation_tokens']['generated_token_ids'][:len(archived['target_token_ids'])]
                            == archived['target_token_ids'], 'Fresh baseline/frozen target prefix changed; do not retarget')
                drift = max_score_delta(baseline['reference_score'], archived['controls']['reference'])
                q['controls']['baseline_reference_max_delta'] = drift
                require(drift == 0, 'Archived baseline reference drift')
                q['controls']['same_image_frozen_reuse_exact'] = True
                q['historical_comparison'] = {arm: {'historical': item['rq2']['historical_predictions'][old],
                    'raw_answer_equal': q['arms'][arm]['response'] == item['rq2']['historical_predictions'][old]['response']}
                    for arm, old in [('baseline576', 'baseline'), ('native-input288', 'input-only')]}
                if cid in profile['controls']['fixed_replay_repeats']:
                    q['controls']['fixed_replay_repeats'] = {}
                    for arm in ARMS:
                        again = generate_from_snapshot(tokenizer, model, processor, image, case, base['pixels'],
                            arm, snapshots[arm], seed, archived['measurement_target'])
                        q['controls']['fixed_replay_repeats'][arm] = row_repeat_check(q['arms'][arm], again)
                        write_json(path, q)
                    q['controls']['fresh_native_repeats'] = {'status': 'running', 'repeats': []}
                    for repeat in range(profile['controls']['fresh_native_captures_per_control']):
                        obs, snap = capture_encoding(model, base['pixels'], seed, 'native-input288')
                        fresh = generate_from_snapshot(tokenizer, model, processor, image, case, base['pixels'],
                            'native-input288', snap, seed, archived['measurement_target'])
                        comparison = compare_realizations(native, snapshots['native-input288'], obs, snap,
                            q['arms']['native-input288']['reference_score'], fresh['reference_score'])
                        comparison['generated_ids_equal'] = (fresh['generation_tokens']['generated_token_ids'] ==
                            q['arms']['native-input288']['generation_tokens']['generated_token_ids'])
                        comparison['scoring_equal'] = fresh['scoring'] == q['arms']['native-input288']['scoring']
                        relative = 'features/' + cid.replace(':', '-') + f'-fresh-native-{repeat+1}.pt'
                        torch.save({'encoded': tuple(t.detach().cpu() for t in snap['encoded']),
                                    'pre_merge': snap['pre_merge'].detach().cpu(),
                                    'post_merge': snap['post_merge'].detach().cpu()}, args.output / relative)
                        report['feature_sha256'][relative] = sha256(args.output / relative)
                        q['controls']['fresh_native_repeats']['repeats'].append({
                            'repeat': repeat+1, 'row': fresh, 'comparison': comparison,
                            'feature_file': relative, 'feature_sha256': report['feature_sha256'][relative]})
                        write_json(path, q)
                        del snap
                    q['controls']['fresh_native_repeats']['status'] = 'recorded'
                q['status'] = 'passed'
                write_json(path, q)
                for row in q['arms'].values():
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    rows.append(row)
                stream.flush()
                report['questions'].append({'case_id': cid, 'status': 'passed'})
                report['question_sha256'][str(path.relative_to(args.output))] = sha256(path)
                report['progress']['completed_questions'] = len(report['questions'])
                report['summaries'] = paired_summary(rows)
                write_json(args.output / 'report.json', report)
                print('Completed ' + cid + ': ' + ' / '.join(arm + '=' + repr(row['response']) for arm, row in q['arms'].items()), flush=True)
            except BaseException as error:
                q['status'], q['error'] = 'failed', f'{type(error).__name__}: {error}'
                write_json(path, q)
                raise
    report.pop('active_case', None)


def finish_run(output, report, started):
    report.update(wall_seconds=time.perf_counter()-started, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / 'report.json', report)
    rows = []
    for p in sorted((output / 'questions').glob('*.json')):
        q = json.loads(p.read_text())
        rows.extend({'case_id': q['case_id'], 'status': q['status'], 'arm': arm,
                     'response': row['response'], 'correct': row['scoring']['correct'],
                     'reference_mean_logprob': row['reference_score']['mean_logprob']}
                    for arm, row in q.get('arms', {}).items())
    if rows:
        write_csv(output / 'answers.csv', rows)
    text = ('<!doctype html><meta charset="utf-8"><title>RQ2 v2 frozen-realization review</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto}pre{white-space:pre-wrap}</style>'
        '<h1>RQ2 v2: first captured realization per image</h1><p>Mode: '+html.escape(report['mode'])+
        '; status: '+html.escape(report['status'])+'</p><p>Exact capture/replay and structural controls; '
        'independent fresh native variability is recorded separately. Native ApET source is unchanged. '
        'This curated panel is not a population accuracy or speed benchmark.</p><p><a href="report.json">Report and controls</a>'+
        (' · <a href="answers.csv">Answers CSV</a>' if rows else '')+'</p><pre>'+
        html.escape(json.dumps(report.get('summaries', {}), indent=2))+'</pre>')
    (output / 'index.html').write_text(text, encoding='utf-8')
    inventory = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'artifact-sha256.json', inventory)
    with zipfile.ZipFile(output.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as bundle:
        for p in sorted(output.rglob('*')):
            if p.is_file(): bundle.write(p, p.relative_to(output))
    print('Status:', report['status'], 'Download:', output.with_suffix('.zip'), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--calibration', action='store_true')
    p.add_argument('--calibration-report', type=Path)
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--vision-model', default=VISION_ID)
    p.add_argument('--cache-dir', type=Path, default=Path('/tmp/apet-original-hf-cache'))
    p.add_argument('--download-budget-gib', type=float, default=0)
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--expected-gpus', type=int, default=2)
    p.add_argument('--setup-log', type=Path)
    p.add_argument('--runtime-report', type=Path)
    args = p.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path): setattr(args, key, value.resolve())
    if args.calibration == bool(args.calibration_report):
        p.error('Choose --calibration OR --calibration-report PATH for the full panel')
    if args.expected_gpus < 1 or not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        p.error('Invalid GPU count or download budget')
    if args.output.exists() or args.output.with_suffix('.zip').exists():
        p.error('Choose a new output path; preserve existing evidence')
    if args.output == args.inputs or args.inputs in args.output.parents:
        p.error('Output must be outside the input bundle')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
    signal.signal(signal.SIGTERM, interrupt_run)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'version': VERSION, 'mode': 'calibration' if args.calibration else 'full',
        'status': 'running', 'calibration_passed': False, 'scientific_pilot_passed': False,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    try:
        report.update(provenance('RQ2 v2 conditional on first captured native realization per image; exact replay, independent numerical controls'))
        report['status'] = 'running'
        validate_runtime(report)
        verify_native_reference(report)
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == args.expected_gpus,
                'Enable the expected CUDA GPUs for the original runtime')
        profile, index, manifests, records = validate_bundle(args.inputs)
        report.update(profile=profile, profile_sha256=sha256(PROFILE),
            measurement_sources_sha256=measurement_hashes(), arms=ARMS,
            reference_index_sha256=profile['reference_index_sha256'],
            timing_note='Capture/hash/replay and controls affect timing; not a speed benchmark')
        prerequisite = validate_calibration(args.calibration_report, args) if not args.calibration else None
        copy_evidence(args, report)
        report['auxiliary_sources_sha256'] = {relative: sha256(REPO / relative) for relative in AUXILIARY_FILES}
        for relative in AUXILIARY_FILES:
            target = args.output / 'evidence' / 'sources' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / relative, target)
        if prerequisite:
            dest = args.output / 'evidence' / 'prerequisite-calibration'
            dest.mkdir()
            shutil.copy2(args.calibration_report, dest / 'report.json')
            for field, prefix in [('evidence_sha256', 'evidence/'),
                                  ('question_sha256', ''), ('feature_sha256', '')]:
                for relative in prerequisite[field]:
                    target = dest / prefix / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(args.calibration_report.parent / prefix / relative, target)
            report['calibration_prerequisite'] = {'passed': True, 'report_sha256': sha256(args.calibration_report)}
        report['evidence_sha256'] = {str(f.relative_to(args.output / 'evidence')): sha256(f)
            for f in (args.output / 'evidence').rglob('*') if f.is_file()}
        write_json(args.output / 'report.json', report)
        print('Validated fixed inputs. Loading LLaVA once for v2 '+report['mode']+'.', flush=True)
        loaded = load_model(args, report)
        report['calibration_identity'] = calibration_identity(args, report, profile)
        if prerequisite:
            require(prerequisite.get('calibration_identity') == report['calibration_identity'],
                    'V2 calibration measurement/runtime/checkpoint identity changed')
        run_panel(args, report, profile, manifests, records, loaded)
        expected = profile['calibration']['case_ids'] if args.calibration else profile['case_ids']
        require(report.get('expected_case_ids') == expected
                and [q['case_id'] for q in report.get('questions', [])] == expected
                and report.get('progress') == {'completed_questions': len(expected), 'expected_questions': len(expected)},
                'V2 run returned without the entire declared panel')
        report.update(status='passed', calibration_passed=args.calibration,
                      scientific_pilot_passed=not args.calibration)
    except KeyboardInterrupt:
        report.update(status='interrupted', error='Partial evidence preserved')
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}', traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        finish_run(args.output, report, started)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
