"""CPU review of the frozen six-case, 576-position layer/IH archive.

Verify raw scores before summarizing. Preserve ZIP/log, never load tensors or
run inference, and write only to a new derived directory. The gallery uses saved
CLIP canvases and HTML patch grids; no local model or plotting install is needed.
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

from analyze_rq1_layer_ih import (PREVIOUS, PREVIOUS_SHA256, require,
                                  verify_ih, verify_zeroing)
from analyze_rq1_scores import PLAN, load_sources
from analyze_rq1_validity import assert_close
from render_rq1_gallery import patch_regions, photo_bounds
from rq1_layer import CASE_IDS, PROFILE as LAYER_PROFILE
from rq1_layer_ih import PROFILE, VERSION, comparison_rows, validate_calibration
from rq1_protocol import SOURCES, sha256, spearman, write_json
from summarize_rq1_pilot import write_csv

REPO = Path(__file__).resolve().parents[1]
IH_CALIBRATION = 'rq1-layer16-ih-calibration-v1-20261018T002129035074Z.zip'
IH_CALIBRATION_SHA256 = '84aa8999e9593f9cd2b436cbce794c205f088083ff2a104b511989e201a40e84'


def verify_prerequisite(z, report):
    """Accept the actual embedded rerun by content, independently of its name."""
    raw = z.read('evidence/prerequisite-layer-calibration.json')
    digest = hashlib.sha256(raw).hexdigest()
    require(report['calibration_prerequisite']['passed']
            and digest == report['calibration_prerequisite']['report_sha256']
            == report['evidence_sha256']['prerequisite-layer-calibration.json'],
            'Embedded calibration hash/status differs')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'report.json'
        path.write_bytes(raw)
        gate = validate_calibration(path, report['arguments']['model'], report['arguments']['vision_model'])
    assert_close(report['calibration_prerequisite'], gate, 'Calibration source gate')
    return {**gate, 'original_output': json.loads(raw)['arguments']['output']}


def exact_repeat(current, prior, fields, label):
    previous = {t['token_index']: t for t in prior}
    require(len(previous) == len(prior), label+': duplicate positions')
    current = {t['token_index']: t for t in current}
    require(previous.keys() <= current.keys(), label+': missing repeated positions')
    for i, old in previous.items():
        for field in fields:
            require(current[i][field] == old[field], label+': differs at '+str(i)+'.'+field)
    return len(previous)


def e_percentile(value, values):
    """Average rank, scaled 0..100; basis excluded by the caller."""
    return 100*(sum(x < value for x in values)+(sum(x == value for x in values)-1)/2)/(len(values)-1)


def summarize(q, previous, regions):
    rows = comparison_rows(q['decoder'], q['ih'])
    basis = set(q['decoder']['diagnostics']['stage']['basis_indices'])
    rows = [{'case_id': q['case_id'], **r, 'fps_basis_source': r['token_index'] in basis,
             **regions[r['token_index']]} for r in rows]
    nonbasis = [r for r in rows if not r['fps_basis_source']]
    photo = [r for r in nonbasis if r['region'] == 'photo']
    best = [dict(r) for r in sorted(rows, key=lambda r: (-r['retained_probability'], r['token_index']))[:5]]
    for rank, row in enumerate(best, 1):
        row.update({'retention_rank': rank, 'E_percentile_nonbasis': None if row['fps_basis_source']
                    else e_percentile(row['E'], [r['E'] for r in nonbasis]),
                    'in_previous_32': row['token_index'] in previous['ih']['token_execution_order']})
    vals = [r['ih_first_probability'] for r in rows]
    summary = {'case_id': q['case_id'], 'target': q['measurement_target'],
        'first_piece': q['ih']['first_reference_token_string'], 'positions': len(rows),
        **q['ih_comparison']['all'],
        **{'nonbasis_'+k: v for k, v in q['ih_comparison']['nonbasis'].items()},
        'rho_E_ih_photo_nonbasis': spearman([r['E'] for r in photo], [r['ih_first_probability'] for r in photo]),
        'rho_E_absolute_ih_auxiliary': spearman([r['E'] for r in rows], [abs(v) for v in vals]),
        'full_probability': rows[0]['full_probability'], 'remove_all_probability': rows[0]['remove_all_probability'],
        'best_retained_probability': best[0]['retained_probability'], 'best_token_index': best[0]['token_index'],
        'best_E_percentile_nonbasis': best[0]['E_percentile_nonbasis'],
        'best_in_previous_32': best[0]['in_previous_32'],
        'previous_32_best_retained_probability': max(t['retained_probability'] for t in previous['ih']['tokens']),
        'median_absolute_ih': statistics.median(map(abs, vals)), 'min_signed_ih': min(vals), 'max_signed_ih': max(vals),
        'positive': sum(v > 0 for v in vals), 'zero': sum(v == 0 for v in vals), 'negative': sum(v < 0 for v in vals),
        # Post-hoc probability-unit effect sizes, not significance or precision thresholds.
        'posthoc_IH_above_0_1': sum(v > .1 for v in vals), 'posthoc_IH_above_0_5': sum(v > .5 for v in vals)}
    return summary, rows, best


def analyze(archive, output, log=None, expected_sha256=None):
    archive, output = Path(archive).resolve(), Path(output).resolve()
    require(not output.exists(), 'Choose a new derived folder; preserve original artifacts')
    digest = sha256(archive)
    require(expected_sha256 is None or expected_sha256 == digest, 'Archive SHA256 differs')
    zero_path, ih_path = archive.parent/PREVIOUS, archive.parent/IH_CALIBRATION
    require(sha256(zero_path) == PREVIOUS_SHA256, 'Previous zero calibration hash differs')
    require(sha256(ih_path) == IH_CALIBRATION_SHA256, 'Previous IH calibration hash differs')
    canonical, audits = load_sources(archive.parent, json.loads(PLAN.read_text(encoding='utf-8')))
    history = {r['case']['case_id']: r for r in canonical}
    questions, summaries, flat, top, stages, all_zero = [], [], [], [], [], []
    zero_repeats = ih_repeats = canonical_repeats = features = 0
    drifts = []
    with zipfile.ZipFile(archive) as z, zipfile.ZipFile(zero_path) as old_zero, zipfile.ZipFile(ih_path) as old_ih:
        names = z.namelist()
        require(z.testzip() is None and len(names) == len(set(names)), 'ZIP CRC/duplicate member failure')
        for name in names:
            require(output in (output/name).resolve().parents, 'Unsafe ZIP path')
        report = json.loads(z.read('report.json'))
        require(report['version'] == VERSION and report['status'] == 'passed'
                and report['arguments']['tokens'] == 576, 'Expected a complete full-grid IH run')
        require(report['implementation']['native_matches_reference'], 'Native implementation differs')
        for name, h in report['evidence_sha256'].items():
            require(hashlib.sha256(z.read('evidence/'+name)).hexdigest() == h, 'Evidence hash differs: '+name)
        for name, h in report['implementation']['native_source_sha256'].items():
            require(hashlib.sha256(z.read('evidence/sources/'+name)).hexdigest() == h, 'Native source differs')
        require(hashlib.sha256(z.read('evidence/sources/environments/original/uv.lock')).hexdigest()
                == report['lock_sha256'], 'Lock hash differs')
        for name, field, current in [('layer-profile.json', 'profile_sha256', LAYER_PROFILE),
                                      ('ih-profile.json', 'ih_profile_sha256', PROFILE)]:
            require(hashlib.sha256(z.read('evidence/'+name)).hexdigest() == report[field] == sha256(current),
                    'Frozen profile differs')
        parent = json.loads(z.read('evidence/layer-profile.json'))
        assert_close(report['protocol'], parent, 'Report layer protocol')
        assert_close(report['ih_protocol'], json.loads(z.read('evidence/ih-profile.json')), 'Report IH protocol')
        prerequisite = verify_prerequisite(z, report)
        lookup, images = {}, {}
        for dataset, (_, h) in SOURCES.items():
            raw = z.read('evidence/'+dataset+'-manifest.json')
            require(hashlib.sha256(raw).hexdigest() == h, 'Accepted manifest differs')
            manifest = json.loads(raw)
            lookup.update({c['case_id']: c for c in manifest['cases']}); images.update(manifest['images'])
        for name in names:
            if name.startswith('features/') and name.endswith('.json'):
                meta = json.loads(z.read(name))
                require(hashlib.sha256(z.read(str(Path(name).with_suffix('.pt')))).hexdigest()
                        == meta['tensor_file_sha256'], 'Feature file byte hash differs')
                features += 1
        require(features == 11, 'Expected five input and six boundary snapshots')
        require([q['case_id'] for q in report['questions']] == CASE_IDS
                == [q['case_id'] for q in report['ih_questions']], 'Panel differs')
        for entry, zero_summary, ih_summary in zip(parent['cases'], report['questions'], report['ih_questions']):
            cid = entry['case_id']; name = 'questions/'+cid.replace(':', '-')+'.json'
            q = json.loads(z.read(name)); prior = json.loads(old_zero.read(name)); prior_ih = json.loads(old_ih.read(name))
            comparison, zero_rows, _, drift = verify_zeroing(q, entry, lookup[cid], 576)
            drifts += [drift, verify_ih(q, entry, 576)]
            assert_close(zero_summary, {'case_id': cid, 'status': 'passed', **comparison}, 'Report zero summary')
            assert_close(ih_summary, {'case_id': cid, 'status': 'passed', 'controls_passed': True,
                                     **q['ih_comparison']}, 'Report IH summary')
            for stage in ['input', 'decoder']:
                require(q[stage]['controls']['reference'] == prior[stage]['controls']['reference'], 'Old reference differs')
                zero_repeats += exact_repeat(q[stage]['tokens'], prior[stage]['tokens'],
                    ['reconstruction_error', 'intervention_score', 'perturbed_score', 'perturbation_l2'], 'Old zero repeat')
            ih_repeats += exact_repeat(q['ih']['tokens'], prior_ih['ih']['tokens'],
                ['reconstruction_error', 'ih_first_probability', 'retained_probability', 'retained_score'], 'Old IH repeat')
            original = history[cid]['case']
            for key in ['clip_sha256', 'projected_sha256', 'seed']:
                require(q[key] == original[key], 'Canonical feature/seed differs')
            require(q['decoder']['boundary_hidden_sha256'] == prior['decoder']['boundary_hidden_sha256'],
                    'Old layer boundary identity differs')
            require(q['input']['controls']['reference'] == original['controls']['reference'], 'Canonical reference differs')
            require(q['input_diagnostics']['input']['errors'] == history[cid]['native']['input']['errors'], 'Canonical E differs')
            canonical_repeats += exact_repeat(q['input']['tokens'], original['tokens'],
                ['reconstruction_error', 'intervention_score', 'perturbed_score'], 'Canonical input repeat')
            original_image = [n for n in names if n.startswith('images/'+q['image_sha256']+'.')]
            require(len(original_image) == 1 and hashlib.sha256(z.read(original_image[0])).hexdigest()
                    == q['image_sha256'], 'Encoded image hash differs')
            regions = patch_regions(photo_bounds(images[q['image_id']]['size']))
            summary, rows, best = summarize(q, prior_ih, regions)
            summaries.append(summary); flat.extend(rows); top.extend(best); all_zero.extend(zero_rows)
            stages.append({'case_id': cid, 'target': q['measurement_target'],
                'input_E_zero_whole_rho': comparison['input']['rho_E_I_all'],
                'layer16_E_zero_whole_rho': comparison['layer16']['rho_E_I_all'],
                'input_E_zero_whole_nonbasis_rho': comparison['input']['rho_E_I_nonbasis'],
                'layer16_E_zero_whole_nonbasis_rho': comparison['layer16']['rho_E_I_nonbasis'],
                'cross_stage_E_rho': comparison['rho_E_between_stages'],
                'cross_stage_zero_whole_rho': comparison['rho_I_between_stages']})
            questions.append(q)
        require(report['progress']['completed_questions'] == report['ih_progress']['completed_questions'] == 6
                and report['progress']['completed_patch_stage_scores'] == len(all_zero) == 6912
                and report['ih_progress']['completed_retention_scores'] == len(flat) == 3456
                and report['total_patch_method_scores'] == report['expected_patch_method_scores'] == 10368,
                'Full score counts differ')
        output.mkdir(parents=True); z.extractall(output)
    a, b = questions[:2]
    pair = {'case_a': a['case_id'], 'case_b': b['case_id'],
        'layer_E_rank_agreement': spearman(a['decoder']['diagnostics']['stage']['errors'], b['decoder']['diagnostics']['stage']['errors']),
        'input_zero_whole_rank_agreement': spearman([t['intervention_score'] for t in a['input']['tokens']], [t['intervention_score'] for t in b['input']['tokens']]),
        'layer_zero_whole_rank_agreement': spearman([t['intervention_score'] for t in a['decoder']['tokens']], [t['intervention_score'] for t in b['decoder']['tokens']]),
        'layer_IH_first_rank_agreement': spearman([t['ih_first_probability'] for t in a['ih']['tokens']], [t['ih_first_probability'] for t in b['ih']['tokens']]),
        'input_projected_hash_equal': a['projected_sha256'] == b['projected_sha256'],
        'decoder_visual_hash_equal': a['decoder']['visual_hidden_sha256'] == b['decoder']['visual_hidden_sha256']}
    verification = {'version': 'rq1-layer-ih-full-saved-review-v1', 'archive': str(archive), 'archive_sha256': digest,
        'log': str(log) if log else None, 'log_sha256': sha256(log) if log else None, 'source_commit': report['commit'],
        'questions': 6, 'images': 5, 'positions_per_question': 576, 'verified_zeroing_scores': len(all_zero),
        'verified_IH_scores': len(flat), 'total_patch_method_scores': 10368,
        'exact_zero_calibration_repeats': zero_repeats, 'exact_IH_calibration_repeats': ih_repeats,
        'exact_canonical_input_repeats': canonical_repeats, 'canonical_source_audits': audits,
        'verified_evidence_files': len(report['evidence_sha256']), 'verified_feature_file_bytes': features,
        'embedded_calibration': prerequisite, 'max_control_drift': max(drifts), 'same_image': pair,
        'wall_seconds': report['wall_seconds'], 'phase_seconds': report['phase_seconds'],
        'inference_performed': False, 'tensor_contents_loaded': False,
        'scope': 'Six curated questions, five images, fixed input/block16 boundaries. First tokenizer-piece counterfactual probabilities are not generated-answer accuracy or intrinsic importance. No population or full-horizon conclusion.',
        'standalone_rerun_calibration_gap': 'The prerequisite rerun report is embedded and verified. Its separately timestamped ZIP/log were not supplied; no rerun needed.'}
    sources = ['scripts/analyze_rq1_layer_ih_full.py', 'scripts/analyze_rq1_layer_ih.py',
        'scripts/analyze_rq1_layer.py', 'scripts/analyze_rq1_scores.py', 'scripts/analyze_rq1_validity.py',
        'scripts/analyze_rq1_saved.py', 'scripts/rq1_layer_ih.py', 'scripts/rq1_layer.py',
        'scripts/rq1_protocol.py', 'scripts/render_rq1_gallery.py', 'scripts/summarize_rq1_pilot.py',
        'research/protocols/rq1_score_reanalysis_v1.json']
    verification['analysis_sources_sha256'] = {}
    for relative in sources:
        target = output/'analysis-evidence'/relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(REPO/relative, target)
        verification['analysis_sources_sha256'][relative] = sha256(target)
    write_json(output/'verification.json', verification)
    write_json(output/'effect-summary.json', summaries); write_csv(output/'effect-summary.csv', summaries)
    write_csv(output/'verified-comparison.csv', flat); write_csv(output/'top-retained-tokens.csv', top)
    write_csv(output/'stage-comparison.csv', stages); write_csv(output/'verified-zeroing.csv', all_zero)
    render(output, questions, summaries, flat, top, images, verification)
    return verification


def render(output, questions, summaries, flat, top, images, verification):
    """Exploratory HTML maps, with hover values and exact padded-image footprint."""
    esc = lambda value: html.escape(str(value), quote=True)
    def table(rows, columns):
        return '<table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in columns)+'</tr>'+''.join(
            '<tr>'+''.join('<td>'+esc(r[k] if not isinstance(r[k], float) else f'{r[k]:.6g}')+'</td>' for k in columns)+'</tr>' for r in rows)+'</table>'
    body = ['<!doctype html><meta charset="utf-8"><title>RQ1 full layer / IH review</title>',
        '<style>body{font:16px system-ui;max-width:1500px;margin:2em auto;padding:1em}table{border-collapse:collapse;font-size:14px}td,th{padding:.5em;border:1px solid #ccc}section{margin:3em 0}h3{font-size:16px}.panels{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}.patches,.photo{position:relative;aspect-ratio:1}.patches{display:grid;grid-template-columns:repeat(24,1fr)}.photo img{width:100%;display:block}.footprint{position:absolute;border:2px solid #00bdcf;pointer-events:none;box-sizing:border-box}.mark{position:absolute;box-sizing:border-box;border:2px solid #f4ef00;color:white;font:bold 11px system-ui;background:#0007;text-align:center}.basis{outline:1px dotted #777;outline-offset:-1px}.panel p{font-size:13px}.scroll{overflow:auto}</style>',
        '<h1>Full input / layer-16 / retain-one review</h1>',
        '<p>All 10,368 raw measurements verified; controls have zero drift. No new inference. Original ZIP/log unchanged. ',
        '<a href="verification.json">Verification</a> · <a href="effect-summary.csv">Effects</a> · <a href="top-retained-tokens.csv">Top retained states</a> · <a href="stage-comparison.csv">Input / layer16</a> · <a href="report.json">Raw report</a></p>',
        '<p><strong>The full grid finds strong singletons missed by the 32-position sample.</strong> A low best value in that sample did not establish distributed support or synergy. These are probabilities of the first frozen gold tokenizer piece, not full-answer accuracy; retained deep states and text have already seen the image.</p>',
        '<div class="scroll">'+table(summaries, ['case_id', 'target', 'rho_E_zero_first', 'rho_E_ih_first', 'rho_zero_ih_first', 'full_probability', 'remove_all_probability', 'previous_32_best_retained_probability', 'best_retained_probability', 'best_token_index'])+'</div>',
        '<p>Six curated questions on five images. No population inference. Maps use each panel\'s own linear scale; compare the printed ranges and effect table before comparing colors. Red is positive support under that counterfactual; blue is negative. Cyan outlines the photo footprint within the actual padded 336×336 model input. Padding values remain included. Dotted cells are native FPS basis sources. Hover any cell for its index and raw value. Markers 1–5 show retention rank, with exact indices in the adjacent table.</p>']
    for q, summary in zip(questions, summaries):
        cid = q['case_id']; rows = sorted((r for r in flat if r['case_id'] == cid), key=lambda r: r['token_index'])
        best = [r for r in top if r['case_id'] == cid]
        bounds = photo_bounds(images[q['image_id']]['size'])
        footprint = '<div class="footprint" style="left:%.6f%%;top:%.6f%%;width:%.6f%%;height:%.6f%%"></div>' % (
            bounds[0]/3.36, bounds[1]/3.36, (bounds[2]-bounds[0])/3.36, (bounds[3]-bounds[1])/3.36)
        body += ['<section><h2>'+esc(cid)+' — '+esc(q['measurement_target'])+'</h2><p>'+esc(q['question'])+'</p>', '<div class="panels"><div class="panel"><h3>Actual padded CLIP input</h3><div class="photo"><img src="images/'+q['image_sha256']+'-canvas.png" alt="Actual saved CLIP canvas">']
        for r in best:
            i = r['token_index']
            body.append('<span class="mark" style="left:%.6f%%;top:%.6f%%;width:4.166667%%;height:4.166667%%" title="%s">%d</span>' % (
                (i%24)*100/24, (i//24)*100/24, esc('Retain rank '+str(r['retention_rank'])+', token '+str(i)+', '+r['region']), r['retention_rank']))
        body += [footprint+'</div><p>24×24 cells; token i = 24×row + column. Markers stay inside their cells. Deep states can encode context beyond the marked raw-image patch.</p></div>']
        for field, title, signed in [('E', 'Native decoder reconstruction E', False),
                                      ('zero_first_probability', 'Zero-one first-token sensitivity', True),
                                      ('ih_first_probability', 'Retain-one minus null sensitivity', True)]:
            limit = max(abs(r[field]) for r in rows) or 1
            body += ['<div class="panel"><h3>'+title+'</h3><div class="patches">']
            for r in rows:
                v = r[field]/limit
                if signed:
                    end = (181, 24, 45) if v >= 0 else (33, 102, 172)
                    rgb = tuple(round(248+(c-248)*abs(v)) for c in end)
                else:
                    rgb = tuple(round(a+(b-a)*v) for a,b in zip((30,65,125),(245,204,65)))
                body.append('<span class="%s" style="background:rgb%s" title="%s"></span>' % (
                    'basis' if r['fps_basis_source'] else '', str(rgb),
                    esc(f"i={r['token_index']}; row,col={divmod(r['token_index'],24)}; {field}={r[field]:.10g}; {r['region']}; FPS={r['fps_basis_source']}")))
            body += [footprint+'</div><p>Linear range '+(f'−{limit:.6g} to +{limit:.6g}' if signed else f'0 to {limit:.6g}')+'.</p></div>']
        body += ['</div><p>All values below are probability units. E percentile is average rank among 566 nonbasis states (0–100). It is absent for a basis source. Top-five selection is a descriptive review, not a new cohort.</p><div class="scroll">'+table(best, ['retention_rank','token_index','retained_probability','ih_first_probability','E','E_percentile_nonbasis','in_previous_32','region'])+'</div></section>']
    body += ['<h2>Same-image pair</h2><pre>'+esc(json.dumps(verification['same_image'], indent=2))+'</pre>']
    (output/'full-review.html').write_text('\n'.join(body), encoding='utf-8')
    # Preserve the runner's original export as index.html; link the new derived review.
    index = output/'index.html'
    index.write_text(index.read_text(encoding='utf-8')+'<p><a href="full-review.html">Open the independently verified full-grid review and maps</a></p>', encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--log', type=Path); p.add_argument('--expected-sha256')
    args = p.parse_args()
    result = analyze(args.archive, args.output, args.log, args.expected_sha256)
    print('Verified', result['total_patch_method_scores'], 'scores; review:', args.output/'full-review.html')
