"""Matched input / decoder-block-16 RQ1 pilot on the original Kaggle runtime.

Default: six frozen questions x 32 random patch positions x two stages. Load the
model once. A full 576-position run requires its passed calibration report.
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
from check_original_env import REPO, provenance, validate_runtime, verify_native_reference
from rq1_layer import (LAYER, PROFILE, VERSION, paired_summary, selected_tokens,
                       validate_calibration, validate_layer_profile)
from rq1_protocol import sha256, spearman, write_json
from run_rq1_input import case_seed, generation_score, interrupt_run, load_model
from run_rq1_validation import copy_evidence, image_for
from smoke_original_apet import MODEL_ID, VISION_ID, generate_native_answer
from summarize_rq1_pilot import write_csv


def layer_evidence(args, report, data):
    copy_evidence(args, report, data)
    evidence = args.output / 'evidence'
    shutil.copy2(PROFILE, evidence / 'layer-profile.json')
    for relative in ['scripts/rq1_layer.py', 'scripts/rq1_layer_native.py', 'scripts/run_rq1_layer.py',
                     'research/tests/test_rq1_layer.py', 'research/tests/test_rq1_layer_native.py',
                     'research/RQ1_LAYER_PILOT.md']:
        target = evidence / 'sources' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, target)
    if args.calibration_report:
        shutil.copy2(args.calibration_report, evidence / 'prerequisite-layer-calibration.json')
    report['evidence_sha256'] = {str(p.relative_to(evidence)): sha256(p)
                                for p in evidence.rglob('*') if p.is_file()}


def run_panel(args, report, profile, data, lookup, loaded):
    import torch
    from PIL import Image
    from rq1_layer_native import measure_layer
    from rq1_native import (encode_image, measure_question, observe_generation_tokens,
                            prepare_question, replaced, tensor_sha256)

    tokenizer, model, processor = loaded
    (args.output / 'questions').mkdir()
    (args.output / 'features').mkdir()
    encoded = None
    current_image = None
    records = []
    report['questions'] = []
    for entry in profile['cases']:
        cid = entry['case_id']
        case = lookup[cid]
        image = image_for(args, data, case)
        seed = case_seed(case, profile['seed'])
        if seed != entry['seed']:
            raise ValueError('Seed differs from the frozen panel')
        order = selected_tokens(entry, args.tokens)
        record = {'case_id': cid, 'image_id': case['image_id'], 'image_sha256': case['image_sha256'],
                  'question': case['question'], 'measurement_target': entry['measurement_target'],
                  'archived_role': entry['archived_role'], 'task_metadata': case['task_metadata'],
                  'seed': seed, 'status': 'running', 'tokens_per_stage': args.tokens}
        path = args.output / 'questions' / (cid.replace(':', '-') + '.json')
        report['active_step'] = {'case_id': cid, 'phase': 'encode'}
        write_json(path, record)
        write_json(args.output / 'report.json', report)
        print(f'Preparing {cid}; {args.tokens} matched positions per stage', flush=True)
        if current_image != case['image_sha256']:
            del encoded
            encoded = encode_image(model, processor, image, seed)
            current_image = case['image_sha256']
            Image.fromarray(encoded['canvas']).save(args.output / 'images' / (current_image + '-canvas.png'))
            tensor = args.output / 'features' / (current_image + '-input.pt')
            torch.save({'clip_with_cls': encoded['clip'].detach().cpu().contiguous(),
                        'projected_patches': encoded['projected'].detach().cpu().contiguous()}, tensor)
            write_json(tensor.with_suffix('.json'), {'image_id': case['image_id'], 'image_sha256': current_image,
                'seed': seed, 'clip_sha256': encoded['clip_sha256'], 'projected_sha256': encoded['projected_sha256'],
                'tensor_file_sha256': sha256(tensor), 'diagnostics': encoded['diagnostics'],
                'processor': processor.to_dict()})
        record['clip_sha256'] = encoded['clip_sha256']
        record['projected_sha256'] = encoded['projected_sha256']
        record['input_diagnostics'] = encoded['diagnostics']
        free, generated = {}, {}
        cached = lambda *a, **kw: (encoded['projected'].clone(), encoded['mask'].clone())
        with replaced(model, 'encode_images', cached), observe_generation_tokens(model, tokenizer, generated):
            generate_native_answer(tokenizer, model, processor, image, case['generation_question'], 'baseline',
                seed=seed, max_new_tokens=16, require_nonempty=False, record=free)
        free['scoring'] = generation_score(cid.split(':')[0], case, free['response'])
        record['free_generation'] = free
        record['generation_tokens'] = generated
        write_json(path, record)
        if not free['scoring']['correct']:
            raise RuntimeError(f'{cid}: fresh baseline changed; no retargeting or case replacement')
        prepared = prepare_question(tokenizer, model, case, encoded)
        if prepared['target_token_ids'] != entry['target_token_ids']:
            raise RuntimeError(f'{cid}: tokenization differs from the frozen reference')
        del prepared

        def progress(stage):
            record[stage.get('stage', 'input')] = stage
            count = len(stage['tokens'])
            phase = stage.get('stage', 'input')
            report['active_step'] = {'case_id': cid, 'phase': phase, 'patches': count,
                                     'expected_patches': args.tokens, 'status': stage['status']}
            write_json(path, record)
            write_json(args.output / 'report.json', report)
            if count % 16 == 0 or stage['status'] != 'running':
                print(f'{cid} {phase}: {count}/{args.tokens}; {stage["status"]}', flush=True)

        input_record = measure_question(tokenizer, model, case, encoded, order,
            profile['control_tolerance'], progress, generation_tokens=generated, baseline_correct=True)
        if input_record['status'] != 'passed':
            raise RuntimeError('Input measurement controls failed')
        for row in input_record['tokens']:
            row['perturbation_l2'] = float(encoded['projected'][:, row['token_index']].float().norm().item())

        def save_boundary(hidden, layer_record):
            tensor = args.output / 'features' / (cid.replace(':', '-') + '-layer16.pt')
            torch.save({'full_block_output': hidden.detach().cpu().contiguous()}, tensor)
            write_json(tensor.with_suffix('.json'), {
                'case_id': cid, 'layer': LAYER, 'module_index': LAYER-1,
                'boundary_shape': list(hidden.shape), 'boundary_hidden_sha256': tensor_sha256(hidden),
                'visual_hidden_sha256': layer_record['visual_hidden_sha256'],
                'image_expanded_start': layer_record['image_expanded_start'],
                'target_expanded_start': layer_record['target_expanded_start'],
                'target_token_ids': layer_record['target_token_ids'],
                'tensor_file_sha256': sha256(tensor), 'diagnostics': layer_record['diagnostics']})
            return str(tensor.relative_to(args.output))

        layer_record = measure_layer(model, tokenizer, case, encoded, input_record, entry, order,
                                     profile['control_tolerance'], progress, save_boundary)
        if layer_record['status'] != 'passed':
            raise RuntimeError('Layer measurement controls failed')
        record['decoder'] = layer_record
        record['comparison'] = paired_summary(input_record, layer_record,
            encoded['diagnostics']['input']['basis_indices'], layer_record['diagnostics']['stage']['basis_indices'])
        record['status'] = 'passed'
        records.append(record)
        write_json(path, record)
        report['questions'].append({'case_id': cid, 'status': 'passed', **record['comparison']})
        report['progress'] = {'completed_questions': len(records), 'expected_questions': len(profile['cases']),
            'completed_patch_stage_scores': len(records)*args.tokens*2,
            'expected_patch_stage_scores': len(profile['cases'])*args.tokens*2}
        write_json(args.output / 'report.json', report)
        print(f'Completed {cid}: input rho={record["comparison"]["input"]["rho_E_I_all"]}; '
              f'layer16 rho={record["comparison"]["layer16"]["rho_E_I_all"]}', flush=True)
    # Causal visual positions precede questions, but numerical full-forward shapes
    # can differ. Record same-image E/feature agreement rather than assume it.
    report['same_image_pairs'] = []
    for i, a in enumerate(records):
        for b in records[i+1:]:
            if a['image_sha256'] != b['image_sha256']:
                continue
            report['same_image_pairs'].append({'case_a': a['case_id'], 'case_b': b['case_id'],
                'input_projected_equal_hash': a['projected_sha256'] == b['projected_sha256'],
                'layer_visual_equal_hash': a['decoder']['visual_hidden_sha256'] == b['decoder']['visual_hidden_sha256'],
                'layer_E_max_abs_difference': max(abs(x-y) for x,y in zip(
                    a['decoder']['diagnostics']['stage']['errors'], b['decoder']['diagnostics']['stage']['errors'])),
                'rho_I_between_questions_layer16': spearman(
                    [t['intervention_score'] for t in a['decoder']['tokens']],
                    [t['intervention_score'] for t in b['decoder']['tokens']])})


def write_review(output, report):
    from render_rq1_gallery import patch_regions, photo_bounds
    summaries, tokens = [], []
    profile = report.get('protocol', {})
    image_meta = {}
    for path in (output / 'evidence').glob('*-manifest.json'):
        image_meta.update(json.loads(path.read_text())['images'])
    for path in sorted((output / 'questions').glob('*.json')):
        record = json.loads(path.read_text())
        regions = patch_regions(photo_bounds(image_meta[record['image_id']]['size'])) if record.get('image_id') in image_meta else {}
        for stage, key in [('input', 'input'), ('layer16', 'decoder')]:
            part = record.get(key, {})
            diag = record.get('input_diagnostics', {}).get('input', {}) if stage == 'input' else part.get('diagnostics', {}).get('stage', {})
            basis = set(diag.get('basis_indices', []))
            for t in part.get('tokens', []):
                tokens.append({'case_id': record['case_id'], 'stage': stage,
                    'token_index': t['token_index'], 'patch_row': t['patch_row'], 'patch_col': t['patch_col'],
                    'region': regions.get(t['token_index'], {}).get('region'),
                    'fps_basis_source': t['token_index'] in basis, 'E': t['reconstruction_error'],
                    'I': t['intervention_score'], 'perturbation_l2': t.get('perturbation_l2'),
                    'reference_mean_logprob': part.get('controls', {}).get('reference', {}).get('mean_logprob'),
                    'perturbed_mean_logprob': t['perturbed_score']['mean_logprob']})
        if record.get('status') == 'passed':
            c = record['comparison']
            summaries.append({'case_id': record['case_id'], 'image_id': record['image_id'],
                'question': record['question'], 'target': record['measurement_target'], 'archived_role': record['archived_role'],
                'patches': c['measured_positions'], 'input_rho_all': c['input']['rho_E_I_all'],
                'layer16_rho_all': c['layer16']['rho_E_I_all'],
                'input_rho_nonbasis': c['input']['rho_E_I_nonbasis'],
                'layer16_rho_nonbasis': c['layer16']['rho_E_I_nonbasis'],
                'input_rho_common_nonbasis': c['input']['rho_E_I_common_nonbasis'],
                'layer16_rho_common_nonbasis': c['layer16']['rho_E_I_common_nonbasis'],
                'rho_E_between_stages': c['rho_E_between_stages'], 'rho_I_between_stages': c['rho_I_between_stages']})
    write_csv(output / 'tokens.csv', tokens)
    write_csv(output / 'questions.csv', summaries)
    parts = ['<!doctype html><meta charset="utf-8"><title>RQ1 input / layer 16</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:1em}table{border-collapse:collapse}td,th{padding:.5em;border:1px solid #ccc}</style>',
        '<h1>RQ1 input / decoder layer 16</h1><p>Status: ' + html.escape(report['status']) + '</p>',
        '<p>Matched positions, fixed answers, uncompressed upstream context. Native reconstruction is observed on protected copies; its compressed output is discarded. No native compression accuracy or population inference is claimed.</p>',
        '<p>Input E is in CLIP space; layer E is in decoder hidden space. Zeroing strengths differ and their L2 norms are recorded. Compare rankings, not calibrated raw E/I magnitudes. Basis sources are reported separately. Padding features can carry contextual information.</p>',
        '<p><a href="tokens.csv">All patch/stage scores</a> · <a href="questions.csv">Question comparisons</a> · <a href="report.json">Full report and controls</a></p>',
        '<table><tr><th>Case</th><th>Patches</th><th>Input E–I rho</th><th>Layer 16 E–I rho</th><th>I rank agreement across stages</th></tr>']
    for row in summaries:
        parts.append('<tr>' + ''.join('<td>' + html.escape(str(row[k])) + '</td>' for k in
            ['case_id', 'patches', 'input_rho_all', 'layer16_rho_all', 'rho_I_between_stages']) + '</tr>')
    parts += ['</table><p>The six questions cover five images and were fixed before the validity-check outcomes. '
              'The 32-position calibration uses random positions independent of E/I. Small-sample correlations are descriptive.</p>',
              '<p>' + html.escape(profile.get('interpretation', '')) + '</p>']
    (output / 'index.html').write_text('\n'.join(parts), encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tokens', type=int, choices=[32, 576], default=32)
    p.add_argument('--calibration-report', type=Path)
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--vision-model', default=VISION_ID)
    p.add_argument('--cache-dir', type=Path, default=Path('/tmp/apet-original-hf-cache'))
    p.add_argument('--download-budget-gib', type=float, default=0)
    args = p.parse_args()
    args.inputs, args.output = args.inputs.resolve(), args.output.resolve()
    if args.calibration_report:
        args.calibration_report = args.calibration_report.resolve()
    if args.output.exists() or args.output.with_suffix('.zip').exists():
        p.error('Choose a new output path; earlier results are preserved')
    if args.inputs == args.output or args.inputs in args.output.parents:
        p.error('Output must be outside the accepted input directory')
    if not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        p.error('Invalid download budget')
    if args.tokens == 32 and args.calibration_report:
        p.error('--calibration-report is only for the 576-position sweep')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
    signal.signal(signal.SIGTERM, interrupt_run)
    args.output.mkdir(parents=True)
    start = time.perf_counter()
    report = {'version': VERSION, 'status': 'running',
        'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}}
    try:
        report.update(provenance('Project RQ1 input / layer-16 matched measurement; original ApET unchanged'))
        report['status'] = 'running'
        validate_runtime(report)
        verify_native_reference(report)
        profile, data, lookup = validate_layer_profile(args.inputs)
        report['protocol'], report['profile_sha256'] = profile, sha256(PROFILE)
        if args.tokens == 576:
            report['calibration_prerequisite'] = validate_calibration(
                args.calibration_report, profile, args.model, args.vision_model)
        layer_evidence(args, report, data)
        loaded = load_model(args, report)
        run_panel(args, report, profile, data, lookup, loaded)
        import torch
        report['peak_allocated_gib_final_segment'] = {str(i): torch.cuda.max_memory_allocated(i)/1024**3
                                                     for i in range(torch.cuda.device_count())}
        report['memory_note'] = 'Generation resets peak counters per question; this is the final segment, not a whole-run peak. Instrumented timings are not a speed benchmark.'
        report['status'] = 'passed'
    except KeyboardInterrupt:
        report['status'], report['error'] = 'interrupted', 'Completed records preserved'
    except Exception as error:
        report['status'], report['error'] = 'failed', f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], flush=True)
    finally:
        report['wall_seconds'] = time.perf_counter()-start
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / 'report.json', report)
        try:
            write_review(args.output, report)
        except Exception as error:
            report['review_error'] = str(error)
            write_json(args.output / 'report.json', report)
        with zipfile.ZipFile(args.output.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as z:
            for path in sorted(args.output.rglob('*')):
                if path.is_file():
                    z.write(path, path.relative_to(args.output))
        print('Status:', report['status'], 'Download:', args.output.with_suffix('.zip'), flush=True)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
