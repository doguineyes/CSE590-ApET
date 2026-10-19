"""First RQ2 pilot: native input ApET versus the same selection/basis without merge.

Run in the original isolated Linux environment. All native source files remain
unchanged. A failed gate writes evidence and a ZIP; no case is silently dropped.
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
from rq1_protocol import sha256, write_json
from rq2_protocol import (ARMS, PROFILE, VERSION, case_panel, paired_summary,
                          require, score_response, source_hashes, validate_bundle)
from run_rq1_input import interrupt_run, load_model, save_image
from smoke_original_apet import MODEL_ID, VISION_ID, configure, generate_native_answer
from summarize_rq1_pilot import write_csv


def score_fixed_reference(tokenizer, model, case, pixels, encoded, arm, target_text):
    """Teacher force the same frozen answer through native multimodal preparation.

    The supplied tuple is the actual native full-576 features plus its selection
    mask, captured during this arm's generation. It is replayed unchanged; this
    function does not implement selection, token positioning or compression.
    """
    import torch
    from rq1_native import reference_tokens, replaced, score_embeddings
    configure(model, 'baseline' if arm == 'baseline576' else 'input-only')
    prompt, ids, labels, prefix_length = reference_tokens(tokenizer, case, target_text)
    device = model.get_input_embeddings().weight.device
    ids, labels = ids.to(device), labels.to(device)
    positions = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    attention = torch.ones_like(ids, dtype=torch.bool)
    cached = lambda *a, **kw: tuple(t.clone() for t in encoded)
    with replaced(model, 'encode_images', cached), torch.inference_mode():
        _, new_positions, new_attention, _, embeddings, expanded_labels = \
            model.prepare_inputs_labels_for_multimodal_x(
                ids, positions, attention, None, labels, pixels, image_sizes=None)
    keep = ARMS[arm]['input_tokens']
    target_ids = labels[labels != -100].tolist()
    target_start = prefix_length - 1 + keep
    require(embeddings.shape[1] == ids.shape[1] - 1 + keep,
            'Native reference expanded length differs from arm budget')
    require(expanded_labels[0, target_start:].tolist() == target_ids
            and bool((expanded_labels[0, :target_start] == -100).all())
            and bool(new_attention.all()), 'Native reference label/attention mapping differs')
    require(torch.equal(new_positions[0], torch.arange(embeddings.shape[1], device=new_positions.device)),
            'Native reference output positions changed')
    result = score_embeddings(model, {'embeddings': embeddings, 'position_ids': new_positions,
        'attention_mask': new_attention, 'labels': expanded_labels})
    return {'target': target_text, 'target_token_ids': target_ids, 'prompt': prompt,
            'expanded_sequence_length': embeddings.shape[1], **result,
            'definition': 'Whole fixed reference mean token log probability, teacher forced, no EOS; secondary diagnostic'}


def copy_evidence(args, report):
    root = args.output / 'evidence'
    root.mkdir()
    shutil.copytree(args.inputs / 'references', root / 'references')
    shutil.copy2(args.inputs / 'protocol.json', root / 'accepted-input-protocol.json')
    for dataset in ('gqa', 'mmstar', 'textvqa'):
        dest = root / 'inputs' / dataset / 'manifest.json'
        dest.parent.mkdir(parents=True)
        shutil.copy2(args.inputs / dataset / 'manifest.json', dest)
    for relative in report['measurement_sources_sha256']:
        dest = root / 'sources' / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / relative, dest)
    for argument, name in [('setup_log', 'setup.log'), ('runtime_report', 'runtime.json')]:
        path = getattr(args, argument, None)
        if path is not None:
            require(path.is_file(), 'Supplied notebook evidence missing: ' + str(path))
            dest = root / 'notebook' / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, dest)
    report['evidence_sha256'] = {str(p.relative_to(root)): sha256(p)
                               for p in root.rglob('*') if p.is_file()}


def raw_native_control(tokenizer, model, processor, image, case, pixels, seed):
    """Unmodified compression output and uninstrumented native generation control."""
    import torch
    from rq1_native import observe_generation_tokens
    from rq2_native import snapshot_encoded
    configure(model, 'input-only')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    with torch.inference_mode():
        original = snapshot_encoded(model.encode_images(pixels))
    generation, tokens = {}, {}
    with observe_generation_tokens(model, tokenizer, tokens):
        generate_native_answer(tokenizer, model, processor, image, case['generation_question'],
            'input-only', seed=seed, max_new_tokens=16, require_nonempty=False, record=generation)
    return {'encoded': original, 'generation': generation, 'generation_tokens': tokens}


def run_arm(tokenizer, model, processor, image, case, pixels, arm, target, seed):
    from rq1_native import observe_generation_tokens
    from rq2_native import native_input_variant
    name = 'baseline' if arm == 'baseline576' else 'input-only'
    configure(model, name)
    generation, tokens = {}, {}
    variant = 'no-merge' if arm == 'native-input288-no-merge' else 'native'
    with native_input_variant(model, variant, capture_tensors=True) as observed:
        with observe_generation_tokens(model, tokenizer, tokens):
            generate_native_answer(tokenizer, model, processor, image, case['generation_question'],
                name, seed=seed, max_new_tokens=16, require_nonempty=False, record=generation)
    snapshots = observed.pop('snapshots')
    require(len(observed['calls']) == len(snapshots) == 1, 'Expected one visual prefill per generation')
    row = {'case_id': case['case_id'], 'arm': arm, 'response': generation['response'],
           'scoring': score_response(case, generation['response']), 'generation': generation,
           'generation_tokens': tokens, 'encoding': observed['calls'][0]}
    row['reference_score'] = score_fixed_reference(
        tokenizer, model, case, pixels, snapshots[0]['encoded'], arm, target)
    return row, snapshots[0]


def same_image_identity(native, variant):
    return {'native': {k: native[k] for k in ('source_clip', 'source_patches', 'pre_merge',
                  'post_merge', 'input', 'input_merge', 'basis_slot_to_source', 'encoded')},
            'no_merge': {k: variant[k] for k in ('pre_merge', 'post_merge', 'encoded')}}


def run_panel(args, report, profile, manifests, records, loaded):
    import torch
    from PIL import Image
    from rq1_native import encode_image, max_score_delta
    from rq2_native import assert_native_parity, assert_selection_equal
    tokenizer, model, processor = loaded
    panel = case_panel(records, args.questions)
    report['expected_case_ids'] = [r['case']['case_id'] for r in panel]
    report['questions'], report['question_sha256'], report['feature_sha256'] = [], {}, {}
    report['progress'] = {'completed_questions': 0, 'expected_questions': len(panel)}
    (args.output / 'questions').mkdir()
    (args.output / 'features').mkdir()
    current_image, encoded = None, None
    identities, rows = {}, []
    with (args.output / 'predictions.jsonl').open('x', encoding='utf-8') as jsonl:
        for item in panel:
            case, archived = item['case'], item['reference']
            cid, seed = case['case_id'], item['rq2']['seed']
            dataset = cid.split(':')[0]
            q = {'case_id': cid, 'status': 'running', 'case': item['rq2'], 'arms': {}, 'controls': {}}
            path = args.output / 'questions' / (cid.replace(':', '-') + '.json')
            write_json(path, q)
            report['active_case'] = cid
            write_json(args.output / 'report.json', report)
            args.manifest = args.inputs / dataset / 'manifest.json'
            image = save_image(args, manifests[dataset], case)
            try:
                if current_image != case['image_sha256']:
                    encoded = encode_image(model, processor, image, seed)
                    current_image = case['image_sha256']
                    Image.fromarray(encoded['canvas']).save(args.output / 'images' / (current_image + '-canvas.png'))
                for key in ('clip_sha256', 'projected_sha256', 'seed'):
                    require(encoded[key] == archived[key], 'Archived input identity changed: ' + key)
                q['controls']['archived_source_exact'] = True
                q['native_control'] = raw_native_control(
                    tokenizer, model, processor, image, case, encoded['pixels'], seed)
                write_json(path, q)
                snapshots = {}
                for arm in ARMS:
                    print(f'{cid}: {arm}', flush=True)
                    row, snapshot = run_arm(tokenizer, model, processor, image, case,
                        encoded['pixels'], arm, archived['measurement_target'], seed)
                    q['arms'][arm] = row
                    snapshots[arm] = snapshot
                    write_json(path, q)
                    require(row['encoding']['source_clip']['sha256'] == archived['clip_sha256'],
                            'Generation CLIP features differ from archived source')
                    require(row['reference_score']['target_token_ids'] == archived['target_token_ids']
                            and row['reference_score']['prompt'] == archived['prompt'], 'Fixed target/prompt changed')
                baseline = q['arms']['baseline576']
                require(baseline['scoring']['correct'], 'Fresh baseline no longer correct; preserve, do not retarget')
                require(baseline['generation_tokens']['generated_token_ids'][:len(archived['target_token_ids'])]
                        == archived['target_token_ids'], 'Fresh baseline frozen-target prefix changed')
                q['controls']['baseline_reference_max_delta'] = max_score_delta(
                    baseline['reference_score'], archived['controls']['reference'])
                require(q['controls']['baseline_reference_max_delta'] == 0, 'Archived baseline reference drift')
                native, variant = [q['arms'][arm] for arm in ('native-input288', 'native-input288-no-merge')]
                control = q['native_control']
                q['controls']['native_parity'] = assert_native_parity(control['encoded'], native['encoding']['encoded'],
                    control['generation_tokens']['generated_token_ids'], native['generation_tokens']['generated_token_ids'])
                q['controls']['selection_and_basis'] = assert_selection_equal(native['encoding'], variant['encoding'])
                require(native['encoding']['input']['errors'] == item['feature']['diagnostics']['input']['errors']
                        and native['encoding']['input']['basis_indices'] == item['feature']['diagnostics']['input']['basis_indices'],
                        'Archived reconstruction/FPS differs')
                identity = same_image_identity(native['encoding'], variant['encoding'])
                previous = identities.setdefault(current_image, identity)
                require(identity == previous, 'Same-image compression changed across questions')
                q['controls']['same_image_exact'] = True
                q['historical_comparison'] = {}
                for arm, prior in [('baseline576', 'baseline'), ('native-input288', 'input-only')]:
                    old = item['rq2']['historical_predictions'][prior]
                    q['historical_comparison'][arm] = {'historical': old,
                        'raw_answer_equal': q['arms'][arm]['response'] == old['response'],
                        'scoring_equal': q['arms'][arm]['scoring']['correct'] == old['correct']}
                # Repeats are controls, never additional evaluated questions.
                q['controls']['repeats'] = {}
                if cid in profile['controls']['reference_repeats']:
                    for arm in ARMS:
                        repeated, _ = run_arm(tokenizer, model, processor, image, case,
                            encoded['pixels'], arm, archived['measurement_target'], seed)
                        q['controls']['repeats'][arm] = repeated
                        write_json(path, q)
                        require(repeated['generation_tokens']['generated_token_ids'] == q['arms'][arm]['generation_tokens']['generated_token_ids']
                                and repeated['encoding'] == q['arms'][arm]['encoding']
                                and max_score_delta(repeated['reference_score'], q['arms'][arm]['reference_score']) == 0,
                                'Repeated arm changed: ' + cid + '/' + arm)
                # Detailed tensor evidence is bounded to the six TextVQA images.
                # Other cases retain native E/selection/merge records and hashes.
                if dataset == 'textvqa':
                    feature_path = args.output / 'features' / (current_image + '.pt')
                    torch.save({'case_id': cid, 'image_sha256': current_image,
                        'native_pre_merge': snapshots['native-input288']['pre_merge'].cpu(),
                        'native_post_merge': snapshots['native-input288']['post_merge'].cpu(),
                        'native_encoded': tuple(t.cpu() for t in snapshots['native-input288']['encoded']),
                        'no_merge_encoded': tuple(t.cpu() for t in snapshots['native-input288-no-merge']['encoded'])}, feature_path)
                    relative = str(feature_path.relative_to(args.output))
                    q['feature_file'], q['feature_sha256'] = relative, sha256(feature_path)
                    report['feature_sha256'][relative] = q['feature_sha256']
                q['status'] = 'passed'
                write_json(path, q)
                for row in q['arms'].values():
                    jsonl.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    rows.append(row)
                jsonl.flush()
                report['questions'].append({'case_id': cid, 'status': q['status']})
                report['question_sha256'][str(path.relative_to(args.output))] = sha256(path)
                report['summaries'] = paired_summary(rows)
                report['progress']['completed_questions'] = len(report['questions'])
                write_json(args.output / 'report.json', report)
                print(f'Completed {cid}: '+ ' / '.join(f'{a}={r["response"]!r}' for a, r in q['arms'].items()), flush=True)
                del snapshots
            except BaseException as error:
                q['status'], q['error'] = 'failed', f'{type(error).__name__}: {error}'
                write_json(path, q)
                raise
    report.pop('active_case', None)


def write_review(output, report):
    if report.get('diagnostic_only'):
        detail = html.escape(json.dumps(report.get('diagnostic', {}), indent=2))
        (output / 'index.html').write_text(
            '<!doctype html><meta charset="utf-8"><title>RQ2 native parity diagnostic</title>'
            '<style>body{font:16px system-ui;max-width:1200px;margin:2em auto;padding:1em}pre{white-space:pre-wrap}</style>'
            '<h1>One-question native parity diagnostic</h1><p>Execution status: '+html.escape(report['status'])+'</p>'
            '<p>This records native numerical differences. A completed diagnostic does not establish a passed RQ2 pilot '
            'or authorize changing its feature-parity tolerance. Original gates and native code remain unchanged.</p>'
            '<p><a href="report.json">Report/provenance</a>'+ \
            (' · <a href="parity-diagnostic.json">Stage comparisons</a>' if (output/'parity-diagnostic.json').is_file() else '')+'</p>'
            '<pre>'+detail+'</pre>', encoding='utf-8')
        return
    rows = []
    for path in sorted((output / 'questions').glob('*.json')):
        q = json.loads(path.read_text())
        for arm, r in q.get('arms', {}).items():
            rows.append({'case_id': q['case_id'], 'question_status': q['status'], 'arm': arm,
                'question': q['case']['question'], 'target': q['case']['canonical_target'],
                'response': r['response'], 'correct': r['scoring']['correct'],
                'consensus_score': r['scoring'].get('consensus_score'),
                'reference_mean_logprob': r['reference_score']['mean_logprob'],
                'mmstar_grounding_flag': q['case']['flags']['mmstar_grounding'],
                'mmstar_prior_quality_flag': q['case']['flags']['mmstar_prior_quality']})
    write_csv(output / 'answers.csv', rows)
    esc = lambda value: html.escape(str(value))
    parts = ['<!doctype html><meta charset="utf-8"><title>RQ2 input merge pilot</title>',
        '<style>body{font:16px system-ui;max-width:1300px;margin:2em auto;padding:1em}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:.5em}</style>',
        '<h1>Native selection/basis: merge versus no merge</h1><p>Status: '+esc(report['status'])+'</p>',
        '<p>Fixed curated RQ1 panel; generated answers and whole frozen-reference scores. Baseline576, native input288, same selection/basis with averaging disabled. Decoder compression off. Repeats are controls, not extra samples. A passed run verifies exact native parity and same selected sources/basis assignment before comparing answers.</p>',
        '<p><a href="report.json">Provenance, summaries and controls</a>'+ \
            (' · <a href="answers.csv">Answers CSV</a>' if rows else '')+ \
            (' · <a href="predictions.jsonl">Passed predictions</a>' if (output/'predictions.jsonl').is_file() else '')+'</p>',
        '<p>Primary TextVQA examples: Brolio, Cookery school, Give up; preserved controls: Dominion, Sky mall, Laughter. Existing MMStar flags remain visible. This diagnostic sample does not estimate population accuracy. Instrumented timing is not a speed benchmark.</p>',
        '<table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in ['case_id','status','arm','question','target','answer','correct','consensus','reference mean log p','grounding flag','quality flag'])+'</tr>']
    for r in rows:
        parts.append('<tr>'+''.join('<td>'+esc(r[k])+'</td>' for k in ['case_id','question_status','arm','question','target','response','correct','consensus_score','reference_mean_logprob','mmstar_grounding_flag','mmstar_prior_quality_flag'])+'</tr>')
    (output / 'index.html').write_text('\n'.join(parts)+'</table>', encoding='utf-8')


def finish_run(output, report, started):
    report['wall_seconds'] = time.perf_counter() - started
    report['finished_utc'] = datetime.now(timezone.utc).isoformat()
    write_json(output / 'report.json', report)
    try:
        write_review(output, report)
    except Exception as error:
        report['review_error'] = str(error)
        write_json(output / 'report.json', report)
    # Hash all evidence, including partial case files; original ZIP never replaced.
    inventory = {str(p.relative_to(output)): sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output / 'artifact-sha256.json', inventory)
    with zipfile.ZipFile(output.with_suffix('.zip'), 'x', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(output.rglob('*')):
            if path.is_file():
                z.write(path, path.relative_to(output))
    print('Status:', report['status'], 'Download:', output.with_suffix('.zip'), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--questions', type=int, default=30)
    p.add_argument('--model', default=MODEL_ID)
    p.add_argument('--vision-model', default=VISION_ID)
    p.add_argument('--cache-dir', type=Path, default=Path('/tmp/apet-original-hf-cache'))
    p.add_argument('--download-budget-gib', type=float, default=0)
    p.add_argument('--require-cuda', action='store_true')
    p.add_argument('--expected-gpus', type=int, default=2)
    p.add_argument('--setup-log', type=Path)
    p.add_argument('--runtime-report', type=Path)
    p.add_argument('--diagnose-native-parity', action='store_true',
                   help='Only first frozen question; measure raw/observed encoding drift, not the 30-case pilot')
    args = p.parse_args(argv)
    for field in ('inputs', 'output', 'cache_dir', 'setup_log', 'runtime_report'):
        value = getattr(args, field)
        if value is not None:
            setattr(args, field, value.resolve())
    if not 1 <= args.questions <= 30:
        p.error('--questions must be 1..30; prefixes are execution checks only')
    if args.expected_gpus < 1 or not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        p.error('Invalid GPU count or download budget')
    if args.output.exists() or args.output.with_suffix('.zip').exists():
        p.error('Choose a new output directory; preserve existing results')
    if args.inputs == args.output or args.inputs in args.output.parents:
        p.error('Output must be outside the input bundle')
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ['HF_XET_CHUNK_CACHE_SIZE_BYTES'] = '0'
    signal.signal(signal.SIGTERM, interrupt_run)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = {'version': 'rq2-native-parity-diagnostic-v1' if args.diagnose_native_parity else VERSION,
        'status': 'running', 'arguments': {
        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    if args.diagnose_native_parity:
        report.update(diagnostic_only=True, scientific_pilot_passed=False)
    try:
        report.update(provenance('RQ2 native input merge/no-merge generation pilot; selection+basis fixed'))
        report['status'] = 'running'
        validate_runtime(report)
        verify_native_reference(report)
        import torch
        require(torch.cuda.is_available(), 'Pretrained original LLaVA pilot requires CUDA')
        require(torch.cuda.device_count() == args.expected_gpus, 'Unexpected GPU count')
        profile, index, manifests, records = validate_bundle(args.inputs)
        report.update(profile=profile, profile_sha256=sha256(PROFILE), arms=ARMS,
            measurement_sources_sha256=source_hashes(extra=[
                'scripts/run_rq2_input_merge.py', 'scripts/rq2_native.py', 'scripts/rq2_parity_diagnostic.py',
                'research/tests/test_rq2_protocol.py', 'research/tests/test_rq2_native.py',
                'research/tests/test_rq2_runner.py', 'research/tests/test_rq2_parity_diagnostic.py',
                'research/notebooks/kaggle_rq2_input_merge_pilot.ipynb',
                'research/RQ2_INPUT_MERGE_PILOT.md']),
            scope='Outcome-enriched diagnostic panel, not benchmark accuracy; only merging omitted in the variant',
            timing_note='Hashing, controls and snapshots affect runtime; not a performance benchmark')
        copy_evidence(args, report)
        write_json(args.output / 'report.json', report)
        if args.diagnose_native_parity:
            report.update(diagnostic_only=True, scientific_pilot_passed=False,
                scope='One-question numerical diagnostic; no RQ2 pilot outcome or gate relaxation')
        print('Validated fixed inputs. Loading LLaVA once for '+
              ('one-question parity diagnostic.' if args.diagnose_native_parity else f'{args.questions} questions.'), flush=True)
        loaded = load_model(args, report)
        if args.diagnose_native_parity:
            from rq2_parity_diagnostic import run_diagnostic
            run_diagnostic(args, report, profile, manifests, records, loaded)
        else:
            run_panel(args, report, profile, manifests, records, loaded)
        report['status'] = 'passed'
    except KeyboardInterrupt:
        report['status'], report['error'] = 'interrupted', 'Partial evidence preserved'
    except Exception as error:
        report['status'], report['error'] = 'failed', f'{type(error).__name__}: {error}'
        report['traceback'] = traceback.format_exc()
        print(report['traceback'], flush=True)
    finally:
        finish_run(args.output, report, started)
    return int(report['status'] != 'passed')


if __name__ == '__main__':
    raise SystemExit(main())
