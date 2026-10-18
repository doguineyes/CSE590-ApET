"""Verify saved IH calibration, preserve originals, and summarize counterfactuals.

CPU only: no inference, tensor loading, new reference selection or score changes.
Run with the existing editor Python and choose a new derived output directory.
"""

import argparse
import hashlib
import html
import json
from pathlib import Path
import shutil
import statistics
import tempfile
import zipfile

from analyze_rq1_layer import verify_question as verify_zeroing
from analyze_rq1_validity import assert_close, verify_score
from rq1_layer import CASE_IDS
from rq1_layer_ih import (PROFILE, VERSION, comparison_rows, comparison_summary, kept_positions,
                          probability, validate_calibration)
from rq1_protocol import SOURCES, sha256, spearman, write_json
from summarize_rq1_pilot import write_csv

PREVIOUS = 'rq1-layer16-calibration-v1-20261017T230335470061Z.zip'
PREVIOUS_SHA256 = '8758b31819ef9f73992dcd46776247b7015107e583ca2615f691b64dd8249fd3'


def require(value, message):
    if not value: raise ValueError(message)


def delta(a, b):
    verify_score(a); verify_score(b)
    require(len(a['token_logprobs']) == len(b['token_logprobs']), 'Reference length differs')
    return max(abs(x-y) for x, y in zip(a['token_logprobs'], b['token_logprobs']))


def verify_ih(q, entry, count):
    ih, layer = q['ih'], q['decoder']
    require(ih['status'] == 'passed' and ih['controls']['passed'], 'IH controls did not pass')
    require((ih['layer'], ih['window_size'], ih['operator']) ==
            (16, 1, 'physical-retain-one-original-positions'), 'IH boundary/unit/operator differs')
    require(ih['target_token_ids'] == entry['target_token_ids']
            and ih['token_execution_order'] == entry['token_order'][:count]
            and [t['token_index'] for t in ih['tokens']] == entry['token_order'][:count], 'IH target/positions differ')
    require(ih['pre_retention_boundary_sha256'] == layer['boundary_hidden_sha256']
            and ih['controls']['final_boundary_exact'], 'IH uncompressed boundary identity differs')
    c = ih['controls']; require(c['tolerance_per_answer_token'] == .001, 'IH tolerance changed')
    assert_close(c['reference'], layer['controls']['reference'], 'Shared full reference')
    drifts = [delta(c['reference'], c[k]) for k in
              ['observation_reference', 'retain_all_noop', 'reference_before', 'reference_after']]
    drifts += [delta(c['remove_all_before'], c[k]) for k in ['remove_all_repeat', 'remove_all_after']]
    drifts += [delta(ih['tokens'][0]['retained_score'], c['first_retained_repeat'])]
    require(max(drifts) <= .001, 'IH repeated/no-op/reference drift')
    assert_close(c['max_reference_delta'], max(drifts[:3]), 'Pre-reference delta')
    for field, actual in [('final_reference_delta', drifts[3]), ('remove_all_repeat_delta', drifts[4]),
                          ('remove_all_after_delta', drifts[5]), ('first_retained_repeat_delta', drifts[6])]:
        assert_close(c[field], actual, field)
    length, start = layer['expanded_sequence_length'], layer['image_expanded_start']
    assert_close(ih['remove_all_kept_positions'], kept_positions(length, start, []), 'Null position map')
    require(ih['remove_all_shortened_length'] == length-576, 'Null sequence length differs')
    assert_close(ih['remove_all_probability'], probability(c['remove_all_before']), 'Null probability')
    for t in ih['tokens']:
        i = t['token_index']; verify_score(t['retained_score'])
        require(len(t['retained_score']['token_logprobs']) == len(entry['target_token_ids']), 'Retained answer length differs')
        require((t['patch_row'], t['patch_col']) == divmod(i, 24), 'IH patch coordinates differ')
        require(t['pre_retention_boundary_exact'], 'IH pre-deletion boundary check failed')
        assert_close(t['kept_positions'], kept_positions(length, start, [i]), 'Retain-one position map')
        require(t['shortened_length'] == length-575, 'Retain-one sequence length differs')
        assert_close(t['reconstruction_error'], layer['diagnostics']['stage']['errors'][i], 'Shared native E')
        assert_close(t['retained_probability'], probability(t['retained_score']), 'Retained probability')
        assert_close(t['ih_first_probability'], t['retained_probability']-ih['remove_all_probability'], 'IH signed probability difference')
    summary = comparison_summary(layer, ih, layer['diagnostics']['stage']['basis_indices'])
    assert_close(q['ih_comparison'], summary, 'IH comparison statistics')
    return max(drifts)


def analyze(archive, output, log=None, expected_sha256=None):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    require(not output.exists(), 'Choose a new derived folder; preserve original artifacts')
    digest = sha256(archive)
    require(expected_sha256 is None or expected_sha256 == digest, 'Archive SHA256 differs')
    previous = archive.parent/PREVIOUS
    require(sha256(previous) == PREVIOUS_SHA256, 'Previous zeroing calibration hash differs')
    questions, summaries, tokens, timing, drifts = [], [], [], [], []
    with zipfile.ZipFile(archive) as z, zipfile.ZipFile(previous) as old:
        names = z.namelist()
        require(z.testzip() is None and len(names) == len(set(names)), 'ZIP CRC/duplicate-member failure')
        require(old.testzip() is None, 'Previous ZIP CRC failure')
        for name in names: require(output in (output/name).resolve().parents, 'Unsafe ZIP path')
        report = json.loads(z.read('report.json')); count = report['arguments']['tokens']
        require(report['version'] == VERSION and report['status'] == 'passed' and count == 32,
                'Expected the complete 32-position IH calibration')
        require(report['implementation']['native_matches_reference'], 'Native reference differs')
        for name, h in report['evidence_sha256'].items():
            require(hashlib.sha256(z.read('evidence/'+name)).hexdigest() == h, 'Evidence hash mismatch: '+name)
        for name, h in report['implementation']['native_source_sha256'].items():
            require(hashlib.sha256(z.read('evidence/sources/'+name)).hexdigest() == h, 'Native source hash mismatch')
        require(hashlib.sha256(z.read('evidence/sources/environments/original/uv.lock')).hexdigest()
                == report['lock_sha256'], 'Original lock hash mismatch')
        for filename, field in [('layer-profile.json', 'profile_sha256'), ('ih-profile.json', 'ih_profile_sha256')]:
            require(hashlib.sha256(z.read('evidence/'+filename)).hexdigest() == report[field], 'Profile hash mismatch')
        require(report['ih_profile_sha256'] == sha256(PROFILE), 'Current IH profile differs')
        parent = json.loads(z.read('evidence/layer-profile.json'))
        assert_close(report['protocol'], parent, 'Report parent protocol')
        assert_close(report['ih_protocol'], json.loads(z.read('evidence/ih-profile.json')), 'Report IH protocol')
        lookup = {}
        for dataset, (_, h) in SOURCES.items():
            raw = z.read('evidence/'+dataset+'-manifest.json')
            require(hashlib.sha256(raw).hexdigest() == h, 'Accepted manifest hash mismatch')
            lookup.update({c['case_id']: c for c in json.loads(raw)['cases']})
        features = 0
        for name in names:
            if name.startswith('features/') and name.endswith('.json'):
                metadata = json.loads(z.read(name))
                require(hashlib.sha256(z.read(str(Path(name).with_suffix('.pt')))).hexdigest()
                        == metadata['tensor_file_sha256'], 'Feature file byte hash mismatch')
                features += 1
        require(features == 11, 'Expected five input and six boundary feature snapshots')
        require([q['case_id'] for q in report['ih_questions']] == CASE_IDS, 'IH panel differs')
        for entry, reported in zip(parent['cases'], report['ih_questions']):
            cid = entry['case_id']; name = 'questions/'+cid.replace(':', '-')+'.json'
            q, prior = json.loads(z.read(name)), json.loads(old.read(name))
            comparison, _, _, drift = verify_zeroing(q, entry, lookup[cid], count)
            drifts.append(drift); drifts.append(verify_ih(q, entry, count))
            for stage in ['input', 'decoder']:
                assert_close(q[stage]['controls']['reference'], prior[stage]['controls']['reference'], 'Repeated full reference')
                for a, b in zip(q[stage]['tokens'], prior[stage]['tokens']):
                    for key in ['token_index','reconstruction_error','intervention_score','perturbed_score','perturbation_l2']:
                        assert_close(a[key], b[key], 'Previous zeroing repeat.'+key)
                timing += [t['seconds'] for t in q[stage]['tokens']]
            for key in ['clip_sha256','projected_sha256']:
                require(q[key] == prior[key], 'Input feature identity changed')
            require(q['decoder']['boundary_hidden_sha256'] == prior['decoder']['boundary_hidden_sha256'], 'Old layer boundary changed')
            assert_close(reported, {'case_id':cid, 'status':'passed','controls_passed':True, **q['ih_comparison']}, 'Report IH summary')
            rows = comparison_rows(q['decoder'], q['ih']); vals = [t['ih_first_probability'] for t in rows]
            full, null = rows[0]['full_probability'], rows[0]['remove_all_probability']
            summaries.append({'case_id':cid, 'target':q['measurement_target'], 'first_piece':q['ih']['first_reference_token_string'],
                'positions':len(rows), 'rho_E_zero_first':q['ih_comparison']['all']['rho_E_zero_first'],
                'rho_E_ih_first':q['ih_comparison']['all']['rho_E_ih_first'],
                'rho_zero_ih_first':q['ih_comparison']['all']['rho_zero_ih_first'],
                'rho_E_absolute_ih_auxiliary':spearman([t['E'] for t in rows], [abs(v) for v in vals]),
                'full_probability':full, 'remove_all_probability':null, 'full_minus_remove_all':full-null,
                'best_sampled_retained_probability':max(t['retained_probability'] for t in rows),
                'min_signed_ih':min(vals), 'median_signed_ih':statistics.median(vals), 'max_signed_ih':max(vals),
                'median_absolute_ih':statistics.median(map(abs,vals)), 'positive':sum(v>0 for v in vals),
                'zero':sum(v==0 for v in vals), 'negative':sum(v<0 for v in vals), 'unique_ih_scores':len(set(vals))})
            tokens += [{'case_id':cid, **row} for row in rows]
            timing += [t['seconds'] for t in q['ih']['tokens']]
            questions.append(q)
        require(report['total_patch_method_scores'] == 576 and report['ih_progress']['completed_retention_scores'] == 192,
                'Completed score count differs')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'report.json'; path.write_bytes(z.read('report.json'))
            eligibility = validate_calibration(path, report['arguments']['model'], report['arguments']['vision_model'])
        output.mkdir(parents=True); z.extractall(output)
    verification = {'version':'rq1-layer-ih-saved-review-v1','archive':str(archive),'archive_sha256':digest,
        'log':str(log) if log else None,'log_sha256':sha256(log) if log else None,'source_commit':report['commit'],
        'verified_evidence_files':len(report['evidence_sha256']),'verified_feature_file_bytes':features,
        'tensor_contents_loaded':False,'inference_performed':False,'questions':6,'positions_per_question':32,
        'zeroing_scores_repeated_exactly':384,'new_ih_scores':192,'total_patch_method_scores':576,
        'max_control_drift':max(drifts),'wall_seconds':report['wall_seconds'],'phase_seconds':report['phase_seconds'],
        'full_run_source_eligibility':eligibility,'prior_calibration_sha256':PREVIOUS_SHA256,
        'loop_seconds':sum(timing),'full_loop_linear_minutes_before_overhead':sum(timing)*18/60,
        'same_image_ih_rank_agreement':spearman([t['ih_first_probability'] for t in questions[0]['ih']['tokens']],
                                               [t['ih_first_probability'] for t in questions[1]['ih']['tokens']]),
        'scope':'Source-informed 1x1 physical-retention adaptation at layer16. First target tokenizer piece; no generated-answer accuracy or population conclusion. The best retained score is only over sampled positions, not all 576.'}
    source = Path(__file__).resolve(); target = output/'analysis-evidence'/source.name
    target.parent.mkdir(); shutil.copy2(source,target)
    verification['analysis_script_sha256'] = sha256(target)
    write_json(output/'verification.json',verification)
    write_json(output/'effect-summary.json',summaries)
    write_csv(output/'effect-summary.csv',summaries)
    write_csv(output/'verified-comparison.csv',tokens)
    banner = '<p><strong>CPU-verified review.</strong> All controls have zero drift and all 384 zeroing effects repeat exactly. Original ZIP/log preserved. <a href="verification.json">Verification</a> · <a href="effect-summary.csv">Null/full/retained probabilities, signs and effect sizes</a>.</p>'
    columns = ['case_id','full_probability','remove_all_probability','best_sampled_retained_probability','median_absolute_ih']
    table = '<h2>First-target-token probabilities</h2><table><tr>'+''.join('<th>'+k+'</th>' for k in columns)+'</tr>'
    for row in summaries:
        table += '<tr>'+''.join('<td>'+html.escape(str(row[k]))+'</td>' for k in columns)+'</tr>'
    table += '</table><p>Best retained value is over 32 sampled positions. A strong unsampled singleton may still exist. Small singleton scores do not establish small collective effects. Sequence deletion and retained context differ from zeroing; these results do not identify an intrinsic token importance.</p>'
    index = output/'index.html'
    index.write_text(index.read_text(encoding='utf-8')+banner+table,encoding='utf-8')
    return verification


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--log',type=Path);p.add_argument('--expected-sha256')
    args=p.parse_args()
    result=analyze(args.archive,args.output,args.log,args.expected_sha256)
    print('Verified',result['new_ih_scores'],'IH scores and',result['zeroing_scores_repeated_exactly'],
          'repeated zeroing scores. Review:',args.output/'index.html')
