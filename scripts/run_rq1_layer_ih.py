"""Bounded zeroing / released-style IH comparison on the unchanged ApET runtime.

Run with original Python -X utf8 -I. Reuse the existing zeroing panel unchanged,
then add physical retain-one measurements only at layer 16, one model load.
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
from rq1_layer import selected_tokens
from rq1_layer_ih import (PROFILE, SOURCE_FILES, VERSION, comparison_rows, comparison_summary,
                          validate_calibration, validate_profile)
from rq1_protocol import sha256, write_json
from run_rq1_input import interrupt_run, load_model
from run_rq1_layer import layer_evidence, run_panel
from run_rq1_validation import image_for
from smoke_original_apet import MODEL_ID, VISION_ID
from summarize_rq1_pilot import write_csv


def copy_ih_evidence(args, report, data):
    layer_evidence(args, report, data)
    evidence = args.output/'evidence'
    shutil.copy2(PROFILE, evidence/'ih-profile.json')
    for relative in SOURCE_FILES+['research/tests/test_rq1_layer_ih.py',
                                 'research/tests/test_rq1_layer_ih_native.py', 'research/RQ1_LAYER_IH_PILOT.md',
                                 'research/notebooks/kaggle_rq1_layer16_pilot.ipynb']:
        target = evidence/'sources'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO/relative, target)
    report['evidence_sha256'] = {str(p.relative_to(evidence)): sha256(p)
                                for p in evidence.rglob('*') if p.is_file()}


def run_ih_panel(args, report, profile, data, lookup, loaded):
    from rq1_layer_ih_native import measure_ih
    from rq1_native import encode_image
    tokenizer, model, processor = loaded
    current_image, encoded = None, None
    report['ih_questions'] = []
    for entry in profile['cases']:
        cid = entry['case_id']; case = lookup[cid]
        path = args.output/'questions'/(cid.replace(':', '-')+'.json')
        q = json.loads(path.read_text(encoding='utf-8'))
        order = selected_tokens(entry, args.tokens)
        if current_image != entry['image_sha256']:
            del encoded
            encoded = encode_image(model, processor, image_for(args, data, case), entry['seed'])
            current_image = entry['image_sha256']
        for field in ['clip_sha256', 'projected_sha256']:
            if encoded[field] != q[field]:
                raise RuntimeError('IH re-encoding changed the frozen full-path features: '+field)
        q['status'] = 'running_ih'
        print(f'Preparing {cid} IH: {args.tokens} matched singleton positions', flush=True)

        def progress(part):
            q['ih'] = part
            write_json(path, q)
            count = len(part['tokens'])
            report['active_step'] = {'case_id': cid, 'phase': 'ih', 'positions': count,
                                     'expected_positions': args.tokens, 'status': part['status']}
            report['ih_progress'] = {'completed_questions': len(report['ih_questions']),
                'expected_questions': 6, 'completed_retention_scores': len(report['ih_questions'])*args.tokens+count,
                'expected_retention_scores': 6*args.tokens}
            write_json(args.output/'report.json', report)
            if count % 16 == 0 or part['status'] != 'running':
                print(f'{cid} IH: {count}/{args.tokens}; {part["status"]}', flush=True)

        ih = measure_ih(model, tokenizer, case, encoded, q['decoder'], order,
                        profile['control_tolerance'], progress)
        if ih['status'] != 'passed': raise RuntimeError('IH numerical/reference controls failed')
        q['ih'], q['status'] = ih, 'passed'
        q['ih_comparison'] = comparison_summary(q['decoder'], ih, q['decoder']['diagnostics']['stage']['basis_indices'])
        write_json(path, q)
        report['ih_questions'].append({'case_id': cid, 'status': 'passed', 'controls_passed': ih['controls']['passed'],
                                       **q['ih_comparison']})
        report['ih_progress']['completed_questions'] = len(report['ih_questions'])
        write_json(args.output/'report.json', report)
        print(f'Completed {cid}: layer E/zero-first rho={q["ih_comparison"]["all"]["rho_E_zero_first"]}; '
              f'E/IH-first rho={q["ih_comparison"]["all"]["rho_E_ih_first"]}; '
              f'zero/IH rho={q["ih_comparison"]["all"]["rho_zero_ih_first"]}', flush=True)
    report['total_patch_method_scores'] = 6*args.tokens*3


def write_review(output, report):
    from render_rq1_gallery import patch_regions, photo_bounds
    images = {}
    for path in (output/'evidence').glob('*-manifest.json'):
        images.update(json.loads(path.read_text(encoding='utf-8'))['images'])
    comparisons, summaries, zeros = [], [], []
    for path in sorted((output/'questions').glob('*.json')):
        q = json.loads(path.read_text(encoding='utf-8'))
        for stage, key in [('input', 'input'), ('layer16', 'decoder')]:
            for t in q.get(key, {}).get('tokens', []):
                zeros.append({'case_id': q['case_id'], 'stage': stage, 'token_index': t['token_index'],
                              'E': t['reconstruction_error'], 'I_whole_log': t['intervention_score']})
        if q.get('ih', {}).get('status') != 'passed': continue
        regions = patch_regions(photo_bounds(images[q['image_id']]['size']))
        basis = set(q['decoder']['diagnostics']['stage']['basis_indices'])
        comparisons += [{'case_id': q['case_id'], **r, **regions[r['token_index']],
                          'fps_basis_source': r['token_index'] in basis}
                        for r in comparison_rows(q['decoder'], q['ih'])]
        s = q['ih_comparison']
        summaries.append({'case_id': q['case_id'], 'target': q['measurement_target'], 'positions': s['positions'],
            'first_target_piece': q['ih']['first_reference_token_string'],
            'input_E_zero_whole': q['comparison']['input']['rho_E_I_all'],
            'layer_E_zero_whole': q['comparison']['layer16']['rho_E_I_all'],
            **{'all_'+k: v for k, v in s['all'].items()},
            **{'nonbasis_'+k: v for k, v in s['nonbasis'].items()},
            'full_probability': q['ih']['controls']['reference']['token_logprobs'][0],
            'remove_all_probability': q['ih']['remove_all_probability']})
        summaries[-1]['full_probability'] = math.exp(summaries[-1]['full_probability'])
    write_csv(output/'zero-tokens.csv', zeros)
    write_csv(output/'ih-comparison.csv', comparisons)
    write_csv(output/'questions.csv', summaries)
    columns = ['case_id', 'positions', 'layer_E_zero_whole', 'all_rho_E_zero_first',
               'all_rho_E_ih_first', 'all_rho_zero_ih_first']
    body = ['<!doctype html><meta charset="utf-8"><title>RQ1 zeroing / IH layer pilot</title>',
        '<style>body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:1em}table{border-collapse:collapse}td,th{padding:.5em;border:1px solid #ccc}</style>',
        '<h1>RQ1 layer 16: zeroing and IH retain-one</h1><p>Status: '+html.escape(report['status'])+'</p>',
        '<p>Same six questions, native E and randomized singleton positions. Zeroing is unchanged; IH physically removes every other visual state after block 16, preserves all text states and original retained positions, and compares first-reference-token probability with the remove-all-at-layer reference.</p>',
        '<p>The official scripts default to 2×2 groups; this comparison uses their supported 1×1 unit. Native ApET/SDPA and our gold rendering are retained. This is a documented adaptation, not a complete IH benchmark reproduction. Read <a href="evidence/ih-profile.json">the source-pinned protocol and differences</a>.</p>',
        '<p>Compare zero-first and IH-first using the same probability units; zero-whole remains the historical primary log score. A first tokenizer piece is not always a complete answer. Text already contains earlier image information; this remove-all reference is not question-only. Differences also include deletion versus zeroing, so causal explanations are limited.</p>',
        '<p><a href="questions.csv">Case and nonbasis summaries</a> · <a href="ih-comparison.csv">All comparable positions</a> · <a href="zero-tokens.csv">Unchanged whole-score zeroing</a> · <a href="report.json">Raw report</a></p>',
        '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in columns)+'</tr>']
    for row in summaries:
        body.append('<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in columns)+'</tr>')
    body.append('</table><p>Six curated questions on five images. Basis and padding remain labeled. Small-sample correlations and counterfactual responses are descriptive; no intrinsic token importance, population effect or information horizon is established.</p>')
    (output/'index.html').write_text('\n'.join(body), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, choices=[32, 576], default=32)
    parser.add_argument('--calibration-report', type=Path)
    parser.add_argument('--model', default=MODEL_ID)
    parser.add_argument('--vision-model', default=VISION_ID)
    parser.add_argument('--cache-dir', type=Path, default=Path('/tmp/apet-original-hf-cache'))
    parser.add_argument('--download-budget-gib', type=float, default=0)
    args = parser.parse_args()
    args.inputs, args.output = args.inputs.resolve(), args.output.resolve()
    if args.calibration_report: args.calibration_report = args.calibration_report.resolve()
    if args.output.exists() or args.output.with_suffix('.zip').exists(): parser.error('Choose a new output path')
    if args.inputs == args.output or args.inputs in args.output.parents: parser.error('Output must be outside inputs')
    if not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0: parser.error('Invalid download budget')
    if args.tokens == 32 and args.calibration_report: parser.error('Calibration report is only for the full sweep')
    os.environ['MPLBACKEND'] = 'Agg'; os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
    signal.signal(signal.SIGTERM, interrupt_run)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'version': VERSION, 'status': 'running',
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    try:
        report.update(provenance('Project RQ1 unchanged zeroing + released-style IH layer-16 comparison'))
        report['status'] = 'running'
        validate_runtime(report); verify_native_reference(report)
        ih, parent, data, lookup = validate_profile(args.inputs)
        report.update(protocol=parent, profile_sha256=ih['parent_profile_sha256'],
                      ih_protocol=ih, ih_profile_sha256=sha256(PROFILE),
                      expected_patch_method_scores=6*args.tokens*3)
        if args.tokens == 576:
            report['calibration_prerequisite'] = validate_calibration(args.calibration_report, args.model, args.vision_model)
        copy_ih_evidence(args, report, data)
        loaded = load_model(args, report)
        tick = time.perf_counter()
        run_panel(args, report, parent, data, lookup, loaded)
        report['phase_seconds'] = {'zeroing': time.perf_counter()-tick}
        import torch
        for index in range(torch.cuda.device_count()): torch.cuda.reset_peak_memory_stats(index)
        tick = time.perf_counter()
        run_ih_panel(args, report, parent, data, lookup, loaded)
        report['phase_seconds']['ih'] = time.perf_counter()-tick
        report['ih_phase_peak_allocated_gib'] = {str(i): torch.cuda.max_memory_allocated(i)/1024**3
                                                for i in range(torch.cuda.device_count())}
        report['memory_note'] = 'Peak counters reset at IH phase entry; this is not a whole-run memory or speed benchmark.'
        report['status'] = 'passed'
    except KeyboardInterrupt:
        report['status'], report['error'] = 'interrupted', 'Completed records preserved'
    except Exception as error:
        report['status'], report['error'] = 'failed', f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc(); print(report['traceback'], flush=True)
    finally:
        report['wall_seconds'] = time.perf_counter()-started
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output/'report.json', report)
        try: write_review(args.output, report)
        except Exception as error:
            report['review_error'] = str(error); write_json(args.output/'report.json', report)
        with zipfile.ZipFile(args.output.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(args.output.rglob('*')):
                if path.is_file(): archive.write(path, path.relative_to(args.output))
        print('Status:', report['status'], 'Download:', args.output.with_suffix('.zip'), flush=True)
    return int(report['status'] != 'passed')


if __name__ == '__main__': raise SystemExit(main())
