"""Verify and review an archived RQ1 validity run without inference or unpickling.

Write a NEW derived folder. Original ZIP/log are never edited. Tensor file bytes
are hash checked; tensor contents require Torch and are not loaded by this tool.
"""

import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
import zipfile

from rq1_protocol import sha256, write_json
from rq1_validation import (CONDITIONS, METHODS, PERTURBATION_CASE_IDS, SHORTCUT_CASE_IDS,
                            VERSION, perturbation_summary, rotate_options, score_rotated, shortcut_summary)
from run_rq1_input import generation_score
from run_rq1_validation import write_review
from summarize_rq1_pilot import write_csv


def assert_close(actual, expected, label):
    if isinstance(expected, dict):
        if actual.keys() != expected.keys(): raise ValueError(label + ': keys differ')
        for k in expected: assert_close(actual[k], expected[k], label + '.' + str(k))
    elif isinstance(expected, (list, tuple)):
        if len(actual) != len(expected): raise ValueError(label + ': lengths differ')
        for i, (a, e) in enumerate(zip(actual, expected)): assert_close(a, e, label + f'[{i}]')
    elif isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(label + ': numeric mismatch')
    elif actual != expected:
        raise ValueError(label + ': mismatch')


def verify_score(score):
    values = score['token_logprobs']
    if not values or not all(math.isfinite(v) for v in values): raise ValueError('Missing/nonfinite score')
    assert_close(score['sum_logprob'], sum(values), 'Score sum')
    assert_close(score['mean_logprob'], sum(values)/len(values), 'Score mean')


def verify_question(q, entry, case):
    if q['status'] != 'passed' or q['target_token_ids'] != entry['target_token_ids']:
        raise ValueError('Question status/target mismatch')
    if q['image_sha256'] != entry['image_sha256'] or q['measurement_target'] != entry['measurement_target']:
        raise ValueError('Question image/reference mismatch')
    if not q['target_alignment']['passed'] or not q['target_alignment']['target_matches_generated_prefix']:
        raise ValueError('Reference alignment failed')
    assert_close(q['free_generation']['scoring'], generation_score(q['case_id'].split(':')[0], case, q['free_generation']['response']), 'Free-answer scoring')
    if not q['free_generation']['scoring']['correct']: raise ValueError('Fresh baseline is not correct')
    if q['generation_tokens']['generated_token_ids'][:len(entry['target_token_ids'])] != entry['target_token_ids']:
        raise ValueError('Generated reference token IDs differ')
    controls, alternatives = q['controls'], q['alternative_controls']
    if not controls['passed'] or not alternatives['passed']: raise ValueError('Controls failed')
    reference = controls['reference']
    verify_score(reference)
    max_control_delta = 0.0
    for score in [controls[k] for k in ['reference_repeat', 'noop', 'native_multimodal', 'reference_after']] + [
            alternatives['reference_before'], alternatives['reference_after'], *alternatives['shams'].values()]:
        verify_score(score)
        if len(score['token_logprobs']) != len(reference['token_logprobs']): raise ValueError('Control token length differs')
        max_control_delta = max(max_control_delta, *(abs(a-b) for a,b in zip(score['token_logprobs'], reference['token_logprobs'])))
    if max_control_delta > .001: raise ValueError('Control drift exceeded the frozen tolerance')
    frozen = {p['token_index']: p for p in entry['patches']}
    zeros = {t['token_index']: t for t in q['tokens']}
    if len(zeros) != 32 or len(q['tokens']) != 32 or set(zeros) != set(frozen): raise ValueError('Zeroing patch set differs')
    rows = q['alternatives']
    if len(rows) != 32 or [r['token_index'] for r in rows] != [r['token_index'] for r in entry['patches']]:
        raise ValueError('Alternative patch ordering differs')
    for row in rows:
        i = row['token_index']
        for key, value in frozen[i].items(): assert_close(row[key], value, 'Frozen patch.' + key)
        assert_close(row['current_E'], zeros[i]['reconstruction_error'], 'Native E')
        if set(row['effects']) != set(METHODS): raise ValueError('Missing intervention method')
        for method, effect in row['effects'].items():
            verify_score(effect['score'])
            assert_close(effect['I'], reference['mean_logprob']-effect['score']['mean_logprob'], 'Reference-minus-perturbed I')
            if not math.isfinite(effect['delta_l2']) or effect['delta_l2'] < 0: raise ValueError('Invalid perturbation norm')
        assert_close(row['effects']['zero']['I'], zeros[i]['intervention_score'], 'Repeated zero I')
    assert_close(q['comparison_by_stratum'], perturbation_summary(rows), 'Stratum statistics')
    return max_control_delta


def analyze(archive, output, log=None):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    if output.exists(): raise ValueError('Choose a new derived output folder; original results are preserved')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or z.testzip() is not None: raise ValueError('Archive CRC/duplicate-entry failure')
        for name in names:
            if output not in (output/name).resolve().parents: raise ValueError('Unsafe archive path')
        report = json.loads(z.read('report.json'))
        if report['status'] != 'passed' or report['version'] != VERSION or report['mode'] != 'all':
            raise ValueError('Expected a passed complete v1 validity run')
        for relative, digest in report['evidence_sha256'].items():
            if hashlib.sha256(z.read('evidence/'+relative)).hexdigest() != digest:
                raise ValueError('Evidence hash mismatch: '+relative)
        profile = json.loads(z.read('evidence/validation-profile.json'))
        if hashlib.sha256(z.read('evidence/validation-profile.json')).hexdigest() != report['profile_sha256']:
            raise ValueError('Profile hash mismatch')
        lookup = {}
        for dataset, meta in profile['inputs'].items():
            raw = z.read('evidence/'+dataset+'-manifest.json')
            if hashlib.sha256(raw).hexdigest() != meta['manifest_sha256']: raise ValueError('Accepted manifest differs')
            lookup.update({c['case_id']:c for c in json.loads(raw)['cases']})
        native = report['implementation']['native_source_sha256']
        if not report['implementation']['native_matches_reference']: raise ValueError('Native source changed')
        for name, digest in native.items():
            if hashlib.sha256(z.read('evidence/sources/'+name)).hexdigest() != digest: raise ValueError('Native source hash mismatch')
        if hashlib.sha256(z.read('evidence/sources/environments/original/uv.lock')).hexdigest() != report['lock_sha256']:
            raise ValueError('Original lock hash mismatch')
        feature_files = []
        for name in names:
            if name.startswith('features/') and name.endswith('.json'):
                meta = json.loads(z.read(name)); tensor = str(Path(name).with_suffix('.pt'))
                if hashlib.sha256(z.read(tensor)).hexdigest() != meta['tensor_file_sha256']:
                    raise ValueError('Feature file hash mismatch')
                feature_files.append(tensor)
        trials = [json.loads(line) for line in z.read('shortcuts.jsonl').decode('utf-8').splitlines()]
        expected = Counter((cid, condition, shift, 'trial') for cid in SHORTCUT_CASE_IDS for condition in CONDITIONS for shift in range(4))
        expected.update((cid, 'real', 0, 'reference_repeat') for cid in SHORTCUT_CASE_IDS)
        actual = Counter((r['case_id'],r['condition'],r['shift'],r['kind']) for r in trials)
        if actual != expected: raise ValueError('Shortcut trial/repeat identities differ')
        for row in trials:
            case = lookup[row['case_id']]
            variant = rotate_options(case['question'], case['canonical_target'], row['shift'])
            assert_close(row['variant'], variant, 'Option rotation')
            assert_close(row['scoring'], score_rotated(row['response'], variant), 'Semantic choice scoring')
        assert_close(report['shortcut_summary'], shortcut_summary(trials), 'Shortcut summary')
        questions = [json.loads(z.read('questions/'+cid.replace(':','-')+'.json')) for cid in PERTURBATION_CASE_IDS]
        max_drift = []
        for q, entry in zip(questions, profile['perturbations']):
            if q['case_id'] != entry['case_id']: raise ValueError('Question identity mismatch')
            max_drift.append(verify_question(q, entry, lookup[q['case_id']]))
        output.mkdir(parents=True)
        z.extractall(output)  # Derived copy only. No write to the original ZIP.
    write_review(output, report)  # Regenerate the empty HTML locally in UTF-8.
    rows = []
    for q in questions:
        for stratum, stats in q['comparison_by_stratum'].items():
            selected = [r for r in q['alternatives'] if stratum == 'all_selected' or r['stratum'] == stratum]
            for method, summary in stats.items():
                rows.append({'case_id':q['case_id'], 'stratum':stratum, 'method':method, **summary,
                    'median_absolute_I':statistics.median(abs(r['effects'][method]['I']) for r in selected),
                    'median_perturbation_l2':statistics.median(r['effects'][method]['delta_l2'] for r in selected)})
    write_csv(output/'method-summary.csv', rows)
    write_csv(output/'shortcut-trials.csv', [{'case_id':r['case_id'],'kind':r['kind'],'condition':r['condition'],
        'shift':r['shift'],'response':r['response'],'gold_display':r['variant']['gold_display'],
        'chosen_original_label':r['scoring']['selected_original_label'],
        'chosen_original_content':r['scoring']['selected_content'],'correct':r['scoring']['correct']} for r in trials])
    evidence = {'version':'rq1-validity-saved-review-v1','source_archive':str(archive),
        'source_zip_sha256':sha256(archive), 'source_log':str(log) if log else None,
        'source_log_sha256':sha256(log) if log else None, 'source_commit':report['commit'],
        'source_status':report['status'],'original_export_error':report.get('review_error'),
        'derived_html_regenerated_utf8':True,'verified_evidence_files':len(report['evidence_sha256']),
        'verified_feature_file_bytes':len(feature_files),'tensor_contents_loaded':False,
        'semantic_trials':72,'separate_reference_repeats':6,'questions':6,'patch_method_scores':576,
        'max_full_forward_control_drift':max(max_drift),
        'historical':{q['case_id']:q['historical_comparison'] for q in questions},
        'interpretation':'Curated cases, random/high-I/low-E-high-I strata kept separate. Intervention-specific sensitivity, not ground-truth importance. Half has lower strength; neighbors may preserve information. No inference rerun or automatic cohort exclusion.'}
    write_json(output/'verification.json', evidence)
    banner = '<p><strong>Local reconstructed review.</strong> Original ZIP/log are unchanged. The original HTML export failed under ASCII; this UTF-8 copy is rebuilt entirely from saved records. <a href="verification.json">Verification</a> · <a href="method-summary.csv">Method/stratum statistics</a> · <a href="shortcut-trials.csv">All choice trials</a>.</p>'
    index = output/'index.html'
    index.write_text(index.read_text(encoding='utf-8')+banner, encoding='utf-8')
    return evidence


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--log',type=Path)
    args = p.parse_args()
    result = analyze(args.archive,args.output,args.log)
    print('Verified',result['semantic_trials'],'trials +',result['separate_reference_repeats'],
          'repeats;',result['patch_method_scores'],'patch/method scores. Review:',args.output/'index.html')
