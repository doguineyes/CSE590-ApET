"""Verify an archived layer calibration and rebuild its review without inference.

Use the existing editor Python. Read ZIPs directly, hash tensor files without
loading them, preserve originals, and write a new derived directory only.
"""

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import statistics
import tempfile
import zipfile

from analyze_rq1_scores import PLAN, effects, load_sources
from analyze_rq1_validity import assert_close, verify_score
from rq1_layer import CASE_IDS, PROFILE, VERSION, paired_summary, selected_tokens, validate_calibration
from rq1_protocol import SOURCES, sha256, spearman, write_json
from run_rq1_input import generation_score
from run_rq1_layer import write_review
from summarize_rq1_pilot import write_csv


def require(condition, message):
    if not condition:
        raise ValueError(message)


def control_drift(part, keys, target_count):
    controls = part['controls']
    require(controls['passed'] and controls['tolerance_per_answer_token'] == .001,
            'Control gate or frozen tolerance differs')
    reference = controls['reference']
    verify_score(reference)
    require(len(reference['token_logprobs']) == target_count, 'Reference answer length differs')
    drift = 0.0
    for key in keys:
        score = controls[key]
        verify_score(score)
        require(len(score['token_logprobs']) == target_count, 'Control answer length differs')
        drift = max(drift, *(abs(a-b) for a, b in zip(reference['token_logprobs'], score['token_logprobs'])))
    require(drift <= .001, 'Full-forward control drift exceeds tolerance')
    return drift


def verify_question(q, entry, case, count):
    require(q['status'] == 'passed', 'Question did not pass')
    for key in ('case_id', 'image_id', 'image_sha256', 'measurement_target', 'seed', 'archived_role'):
        assert_close(q[key], entry[key], 'Frozen case.'+key)
    assert_close(q['question'], case['question'], 'Accepted question')
    assert_close(q['free_generation']['scoring'],
                 generation_score(q['case_id'].split(':')[0], case, q['free_generation']['response']),
                 'Fresh answer scoring')
    require(q['free_generation']['scoring']['correct'], 'Fresh baseline is not correct')
    a, b = q['input'], q['decoder']
    require(a['prompt'] == q['free_generation']['prompt'], 'Scoring/generation prompt mismatch')
    require(a['target_alignment']['passed'] and a['target_alignment']['target_matches_generated_prefix'],
            'Target prefix is not aligned')
    ids = entry['target_token_ids']
    require(q['generation_tokens']['generated_token_ids'][:len(ids)] == ids, 'Generated target IDs differ')
    require((b['layer'], b['module_index'], b['boundary']) == (16, 15, 'after_block_16_before_block_17'),
            'Wrong decoder boundary')
    require(b['controls']['boundary_repeat_exact'] and b['controls']['final_boundary_exact'],
            'Repeated boundary states changed within a question')
    native = b['diagnostics']
    require(native['compressed_output_discarded'] and native['rng_restored']
            and native['keep_tokens_observation_only'] == 288
            and native['native_method'] == 'llava.model.modeling_llama_x.LlamaModel.layer_prune',
            'Native decoder observation policy differs')
    require(native['live_hidden_sha256_before_after'] == b['boundary_hidden_sha256'],
            'Observed boundary identity differs')
    require(b['boundary_shape'] == [1, b['expanded_sequence_length'], 4096], 'Boundary geometry differs')
    order = selected_tokens(entry, count)
    drifts, rows, timing = [], [], []
    for label, part, diag, controls in [
        ('input', a, q['input_diagnostics']['input'],
         ['reference_repeat', 'noop', 'native_multimodal', 'reference_after']),
        ('layer16', b, native['stage'],
         ['observation_only', 'reference_after_E_observation', 'reference_repeat',
          'noop_at_boundary', 'boundary_repeat', 'reference_after'])]:
        require(part['status'] == 'passed' and part['target_token_ids'] == ids, 'Stage target/status differs')
        for field in ('image_expanded_start', 'target_expanded_start', 'expanded_sequence_length'):
            require(part[field] == a[field], 'Stage position mapping differs')
        require(part['target_expanded_start']+len(ids) == part['expanded_sequence_length']
                and part['image_expanded_start']+576 <= part['target_expanded_start'], 'Invalid visual/answer span')
        assert_close(part['controls']['reference'], a['controls']['reference'], 'Matched reference score')
        drifts.append(control_drift(part, controls, len(ids)))
        require(part['token_execution_order'] == order
                and [t['token_index'] for t in part['tokens']] == order, 'Patch order or completion differs')
        require(len(diag['errors']) == 576 and len(set(diag['basis_indices'])) == 10
                and len(diag['selected_indices']) == 288 and diag['source_tokens'] == 576,
                'Native reconstruction geometry differs')
        if label == 'layer16':
            require(diag['layer'] == 16 and diag['feature_dimension'] == 4096, 'Native observation stage differs')
        for t in part['tokens']:
            i = t['token_index']
            require((t['patch_row'], t['patch_col']) == divmod(i, 24), 'Patch coordinates differ')
            assert_close(t['reconstruction_error'], diag['errors'][i], 'Native E')
            score = effects(part['controls']['reference'], t['perturbed_score'])
            assert_close(t['intervention_score'], score['whole_log'], 'Signed sensitivity')
            require(math.isfinite(t['perturbation_l2']) and t['perturbation_l2'] > 0, 'Invalid zeroing norm')
            require(math.isfinite(t['seconds']) and t['seconds'] > 0, 'Invalid intervention timing')
            timing.append(t['seconds'])
            rows.append({'case_id': q['case_id'], 'stage': label, 'token_index': i,
                         'E': t['reconstruction_error'], 'fps_basis_source': i in diag['basis_indices'],
                         'zeroing_l2': t['perturbation_l2'], **score})
    summary = paired_summary(a, b, q['input_diagnostics']['input']['basis_indices'], native['stage']['basis_indices'])
    assert_close(q['comparison'], summary, 'Paired statistics')
    return summary, rows, timing, max(drifts)


def analyze(archive, output, log=None, expected_sha256=None):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    require(not output.exists(), 'Choose a new derived output folder; original results are preserved')
    archive_hash = sha256(archive)
    require(expected_sha256 is None or expected_sha256 == archive_hash, 'Archive SHA256 differs')
    canonical, canonical_audits = load_sources(archive.parent, json.loads(PLAN.read_text(encoding='utf-8')))
    history = {r['case']['case_id']: r for r in canonical}
    records, flat, comparisons, timing, drifts = [], [], [], [], []
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        require(len(names) == len(set(names)) and z.testzip() is None, 'ZIP CRC/duplicate entry failure')
        for name in names:
            require(output in (output/name).resolve().parents, 'Unsafe archive path')
        report = json.loads(z.read('report.json'))
        count = report['arguments']['tokens']
        require(report['status'] == 'passed' and report['version'] == VERSION and count in (32, 576),
                'Expected a complete passed layer run')
        require(report['implementation']['native_matches_reference'], 'Native reference differs')
        for name, digest in report['evidence_sha256'].items():
            require(hashlib.sha256(z.read('evidence/'+name)).hexdigest() == digest, 'Evidence hash mismatch: '+name)
        for name, digest in report['implementation']['native_source_sha256'].items():
            require(hashlib.sha256(z.read('evidence/sources/'+name)).hexdigest() == digest, 'Native source mismatch')
        require(hashlib.sha256(z.read('evidence/sources/environments/original/uv.lock')).hexdigest()
                == report['lock_sha256'], 'Original lock mismatch')
        profile_raw = z.read('evidence/layer-profile.json')
        require(hashlib.sha256(profile_raw).hexdigest() == report['profile_sha256'] == sha256(PROFILE), 'Layer profile differs')
        profile = json.loads(profile_raw)
        assert_close(report['protocol'], profile, 'Report protocol')
        lookup = {}
        for dataset, (_, digest) in SOURCES.items():
            raw = z.read('evidence/'+dataset+'-manifest.json')
            require(hashlib.sha256(raw).hexdigest() == digest, 'Accepted manifest hash differs')
            lookup.update({c['case_id']: c for c in json.loads(raw)['cases']})
        features = {}
        for name in names:
            if name.startswith('features/') and name.endswith('.json'):
                meta = json.loads(z.read(name))
                tensor = str(Path(name).with_suffix('.pt'))
                require(hashlib.sha256(z.read(tensor)).hexdigest() == meta['tensor_file_sha256'], 'Feature file bytes differ')
                features[tensor] = meta
        require(len(features) == 11, 'Expected five input and six boundary feature snapshots')
        require([q['case_id'] for q in report['questions']] == CASE_IDS, 'Report panel differs')
        for entry, reported in zip(profile['cases'], report['questions']):
            cid = entry['case_id']
            q = json.loads(z.read('questions/'+cid.replace(':', '-')+'.json'))
            comparison, tokens, seconds, drift = verify_question(q, entry, lookup[cid], count)
            assert_close(reported, {'case_id': cid, 'status': 'passed', **comparison}, 'Report summary')
            original = history[cid]['case']
            for field in ('clip_sha256', 'projected_sha256', 'seed'):
                assert_close(q[field], original[field], 'Archived input repeat.'+field)
            assert_close(q['input']['controls']['reference'], original['controls']['reference'], 'Archived baseline repeat')
            assert_close(q['input_diagnostics']['input']['errors'], history[cid]['native']['input']['errors'], 'Archived E repeat')
            old = {t['token_index']: t for t in original['tokens']}
            for t in q['input']['tokens']:
                assert_close(t['intervention_score'], old[t['token_index']]['intervention_score'], 'Archived I repeat')
            meta = features['features/'+q['image_sha256']+'-input.pt']
            for field in ('image_id', 'image_sha256', 'seed', 'clip_sha256', 'projected_sha256'):
                assert_close(q[field], meta[field], 'Input feature identity')
            boundary = features[q['decoder']['boundary_feature_file']]
            for field in ('boundary_shape', 'boundary_hidden_sha256', 'visual_hidden_sha256',
                          'target_token_ids', 'image_expanded_start', 'target_expanded_start'):
                assert_close(q['decoder'][field], boundary[field], 'Boundary feature identity')
            images = [n for n in names if n.startswith('images/'+q['image_sha256']+'.')]
            require(len(images) == 1 and hashlib.sha256(z.read(images[0])).hexdigest() == q['image_sha256'],
                    'Original image bytes differ')
            records.append(q); flat.extend(tokens); timing.extend(seconds); drifts.append(drift)
            comparisons.append({'case_id': cid, 'target': q['measurement_target'],
                'input_rho': comparison['input']['rho_E_I_all'], 'layer16_rho': comparison['layer16']['rho_E_I_all'],
                'cross_stage_E_rho': comparison['rho_E_between_stages'],
                'cross_stage_I_rho': comparison['rho_I_between_stages']})
        require(report['progress']['completed_patch_stage_scores'] == len(flat) == 6*count*2, 'Completion count differs')
        pair_a, pair_b = records[:2]
        da, db = [q['decoder']['diagnostics']['stage'] for q in (pair_a, pair_b)]
        pair = {'case_a': pair_a['case_id'], 'case_b': pair_b['case_id'],
            'input_projected_equal_hash': pair_a['projected_sha256'] == pair_b['projected_sha256'],
            'layer_visual_equal_hash': pair_a['decoder']['visual_hidden_sha256'] == pair_b['decoder']['visual_hidden_sha256'],
            'layer_E_max_abs_difference': max(abs(a-b) for a, b in zip(da['errors'], db['errors'])),
            'rho_I_between_questions_layer16': spearman([t['intervention_score'] for t in pair_a['decoder']['tokens']],
                                                        [t['intervention_score'] for t in pair_b['decoder']['tokens']])}
        assert_close(report['same_image_pairs'], [pair], 'Same-image comparison')
        pair.update({'rho_E_all_576': spearman(da['errors'], db['errors']),
                     'basis_indices_equal': da['basis_indices'] == db['basis_indices'],
                     'selected_indices_equal': da['selected_indices'] == db['selected_indices'],
                     'input_I_between_questions': spearman([t['intervention_score'] for t in pair_a['input']['tokens']],
                                                          [t['intervention_score'] for t in pair_b['input']['tokens']])})
        eligibility = {'passed': False, 'reason': 'This review is not a 32-position calibration'}
        if count == 32:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'report.json'; path.write_bytes(z.read('report.json'))
                try:
                    eligibility = validate_calibration(path, profile, report['arguments']['model'], report['arguments']['vision_model'])
                except ValueError as error:
                    eligibility = {'passed': False, 'reason': str(error)}
        output.mkdir(parents=True)
        z.extractall(output)  # Separate derived copy; the source ZIP is unchanged.
    write_review(output, report)
    analysis_sources = {}
    for relative in ['scripts/analyze_rq1_layer.py', 'scripts/analyze_rq1_scores.py',
                     'scripts/analyze_rq1_validity.py', 'scripts/analyze_rq1_saved.py',
                     'scripts/run_rq1_layer.py', 'scripts/rq1_layer.py',
                     'scripts/rq1_protocol.py', 'scripts/render_rq1_gallery.py',
                     'scripts/summarize_rq1_pilot.py', 'research/protocols/rq1_score_reanalysis_v1.json']:
        source = Path(__file__).resolve().parents[1]/relative
        target = output/'analysis-evidence'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        analysis_sources[relative] = sha256(target)
    score_rows = []
    for q in records:
        for stage in ('input', 'layer16'):
            rows = [t for t in flat if t['case_id'] == q['case_id'] and t['stage'] == stage]
            nonbasis = [t for t in rows if not t['fps_basis_source']]
            score_rows.append({'case_id': q['case_id'], 'stage': stage,
                'rho_whole_all': spearman([t['E'] for t in rows], [t['whole_log'] for t in rows]),
                'rho_first_all': spearman([t['E'] for t in rows], [t['first_log'] for t in rows]),
                'rho_whole_nonbasis': spearman([t['E'] for t in nonbasis], [t['whole_log'] for t in nonbasis]),
                'median_absolute_I': statistics.median(abs(t['whole_log']) for t in rows),
                'median_zeroing_l2': statistics.median(t['zeroing_l2'] for t in rows),
                'exact_zero_I_positions': sum(t['whole_log'] == 0 for t in rows)})
    verification = {'version': 'rq1-layer-saved-review-v1', 'source_archive': str(archive),
        'archive_sha256': archive_hash, 'source_log': str(log) if log else None,
        'log_sha256': sha256(log) if log else None, 'source_commit': report['commit'],
        'source_status': report['status'], 'original_review_error': report.get('review_error'),
        'verified_evidence_files': len(report['evidence_sha256']), 'verified_feature_file_bytes': len(features),
        'tensor_contents_loaded': False, 'inference_performed': False, 'questions': len(records),
        'analysis_sources_sha256': analysis_sources,
        'positions_per_stage_per_question': count, 'patch_stage_scores': len(flat),
        'max_full_forward_control_drift': max(drifts), 'repeated_boundary_controls_exact': True,
        'historical_input_scores_repeated_exactly': count*len(records), 'canonical_source_audits': canonical_audits,
        'wall_seconds': report['wall_seconds'], 'intervention_seconds': sum(timing),
        'mean_seconds_per_intervention': statistics.mean(timing),
        'full_6912_interventions_linear_minutes_excluding_overhead': statistics.mean(timing)*6912/60,
        'full_run_source_eligibility': eligibility, 'same_image': pair,
        'interpretation': 'Curated six-question calibration. Stage-specific zeroing sensitivity, not intrinsic importance. Different spaces and norms; no causal or population depth conclusion. Full-run eligibility concerns source/controls, not scientific approval.'}
    write_json(output/'verification.json', verification)
    write_json(output/'stage-score-summary.json', score_rows)
    write_csv(output/'stage-score-summary.csv', score_rows)
    write_csv(output/'token-score-reanalysis.csv', flat)
    write_json(output/'calibration-comparison.json', comparisons)
    banner = '<p><strong>Verified local review.</strong> Original ZIP/log unchanged. The original summary export failed while reading UTF-8 text under an ASCII locale; this derived page is rebuilt from saved records. <a href="verification.json">Verification</a> · <a href="stage-score-summary.csv">Score and norm summary</a> · <a href="token-score-reanalysis.csv">First/whole token scores</a>.</p>'
    index = output/'index.html'
    index.write_text(index.read_text(encoding='utf-8')+banner+
        '<p>Same-image layer states are not bit-identical across the two question forwards. Native E ranking agreement over all 576 positions: '+
        html.escape(str(pair['rho_E_all_576']))+'. Basis and selected sets match. No pure question-only causal attribution is established.</p>', encoding='utf-8')
    return verification


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-sha256')
    args = parser.parse_args()
    result = analyze(args.archive, args.output, args.log, args.expected_sha256)
    print('Verified', result['patch_stage_scores'], 'scores. Review:', args.output/'index.html')
