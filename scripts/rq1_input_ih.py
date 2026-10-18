"""Frozen input-stage IH extension: reference reuse, pairing and CPU checks."""

import hashlib
import json
from pathlib import Path
import zipfile

from analyze_rq1_validity import assert_close, verify_score
from rq1_layer_ih import kept_positions, probability
from rq1_protocol import REPO, SOURCES, load_manifest, sha256, spearman, token_order

VERSION = 'rq1-input-ih-v1'
PROFILE = REPO/'research/protocols/rq1_input_ih_v1.json'
SOURCE_FILES = ['scripts/rq1_input_ih.py', 'scripts/rq1_input_ih_native.py',
    'scripts/run_rq1_input_ih.py', 'scripts/rq1_native.py', 'scripts/rq1_protocol.py',
    'scripts/rq1_layer_ih.py', 'scripts/rq1_layer.py', 'scripts/analyze_rq1_validity.py',
    'scripts/run_rq1_input.py', 'scripts/smoke_original_apet.py', 'scripts/original_diagnostics.py',
    'scripts/check_original_env.py', 'environments/original/uv.lock',
    'research/provenance/original_llava_reference.json', 'research/protocols/rq1_input_ih_v1.json',
    'llava/eval/m4c_evaluator.py', 'scripts/mmstar_scoring.py', 'scripts/gqa_scoring.py']


def require(value, message):
    if not value: raise ValueError(message)


def file_at(root, relative):
    root = Path(root).resolve(); path = (root/relative).resolve()
    require(root in path.parents, 'Input path escapes bundle')
    return path


def source_hashes():
    native = json.loads((REPO/'research/provenance/original_llava_reference.json').read_text())['native_source_sha256']
    return {name: sha256(REPO/name) for name in sorted(set(SOURCE_FILES+list(native)))}


def locate_inputs(roots, working, explicit=None):
    """Accept Kaggle's unpacked private dataset or the named ZIP, without kernels changes."""
    working = Path(working)
    if explicit:
        chosen = Path(explicit)
    elif (working/'references/index.json').is_file():
        chosen = working
    else:
        folders, archives = [], []
        for root in map(Path, roots):
            if not root.exists(): continue
            folders += [p.parent.parent for p in root.rglob('index.json')
                        if p.parent.name == 'references' and (p.parent.parent/'protocol.json').is_file()]
            archives += list(root.rglob('rq1-ih-inputs-v1.zip'))
        candidates = sorted(set(folders or archives))
        require(len(candidates) == 1,
                'Attach rq1-ih-inputs-v1.zip (or its extracted private dataset); set INPUT_ZIP for ambiguity. Found: '+str(candidates))
        chosen = candidates[0]
    if chosen.is_dir():
        require((chosen/'references/index.json').is_file(), 'This is the old inputs bundle; IH requires rq1-ih-inputs-v1')
        return chosen.resolve()
    require(chosen.is_file() and not working.exists(), 'ZIP missing or extraction directory already exists; choose a fresh directory')
    with zipfile.ZipFile(chosen) as z:
        names = z.namelist()
        require(z.testzip() is None and len(names) == len(set(names)), 'Invalid input ZIP')
        for name in names: file_at(working, name)
        profile = json.loads(PROFILE.read_text())
        require(hashlib.sha256(z.read('references/index.json')).hexdigest() == profile['reference_index_sha256'],
                'Wrong IH reference bundle')
        working.mkdir(parents=True); z.extractall(working)
    return working.resolve()


def validate_bundle(inputs):
    inputs = Path(inputs).resolve(); profile = json.loads(PROFILE.read_text())
    require(profile['version'] == VERSION and profile['boundary'] == 'before_block_1'
            and profile['window_size'] == 1 and profile['control_tolerance'] == .001
            and profile['strict_archived_reuse'] is True, 'Input IH profile changed')
    require(profile['method_reference_profile_sha256'] == sha256(REPO/profile['method_reference_profile'])
            and profile['source_score_plan_sha256'] == sha256(REPO/'research/protocols/rq1_score_reanalysis_v1.json'),
            'Pinned method/source plan differs')
    index_path = inputs/'references/index.json'
    require(sha256(index_path) == profile['reference_index_sha256'], 'Reference index hash differs')
    index = json.loads(index_path.read_text())
    require(all(a['lock_sha256'] == sha256(REPO/'environments/original/uv.lock')
                for a in index['source_audits']), 'Historical runtime lock differs')
    for name, digest in index['files_sha256'].items():
        require(sha256(file_at(inputs, name)) == digest, 'Reference evidence hash differs: '+name)
    cases, manifests = {}, {}
    for dataset in SOURCES:
        data = load_manifest(inputs/dataset/'manifest.json', dataset)
        manifests[dataset] = data
        cases.update({c['case_id']: c for c in data['cases']})
    require([e['case_id'] for e in index['cases']] == profile['case_ids'], 'Frozen panel differs')
    records = []
    for entry in index['cases']:
        cid = entry['case_id']; case = cases[cid]
        q = json.loads(file_at(inputs, entry['question_file']).read_text())
        feature = json.loads(file_at(inputs, entry['feature_file']).read_text())
        require(q['case_id'] == cid and q['status'] == 'passed' and q['controls']['passed'], 'Failed reference')
        require(q['image_sha256'] == case['image_sha256'] and q['question'] == case['question'], 'Reference identity differs')
        require(q['token_execution_order'] == token_order(case['image_sha256'], 590)
                and [t['token_index'] for t in q['tokens']] == q['token_execution_order'], 'Incomplete/reordered archived grid')
        require(len(feature['diagnostics']['input']['errors']) == 576
                and len(set(feature['diagnostics']['input']['basis_indices'])) == 10, 'Archived native geometry differs')
        for key in ['clip_sha256', 'projected_sha256', 'seed']:
            require(q[key] == feature[key], 'Archived feature identity differs')
        verify_score(q['controls']['reference'])
        for t in q['tokens']:
            verify_score(t['perturbed_score'])
            assert_close(t['intervention_score'], q['controls']['reference']['mean_logprob']-t['perturbed_score']['mean_logprob'], 'Archived zero effect')
            assert_close(t['reconstruction_error'], feature['diagnostics']['input']['errors'][t['token_index']], 'Archived native E')
        records.append({'entry': entry, 'case': case, 'reference': q, 'feature': feature})
    require(len(records) == 30 and len({r['case']['image_sha256'] for r in records}) == 24, 'Expected 30 questions / 24 images')
    return profile, index, manifests, records


def verify_reuse_identity(encoded, prepared, archived, feature):
    for key in ['clip_sha256', 'projected_sha256', 'seed']:
        require(encoded[key] == archived[key], 'Archived reuse blocked: '+key+' differs')
    require(encoded['diagnostics']['input']['errors'] == feature['diagnostics']['input']['errors']
            and encoded['diagnostics']['input']['basis_indices'] == feature['diagnostics']['input']['basis_indices'],
            'Archived reuse blocked: native E/basis differs')
    for field, source in [('prompt','prompt'), ('target_token_ids','target_token_ids'),
                          ('image_start','image_expanded_start'), ('target_start','target_expanded_start')]:
        require(prepared[field] == archived[source], 'Archived reuse blocked: '+field+' differs')
    require(prepared['embeddings'].shape[1] == archived['expanded_sequence_length'], 'Archived sequence shape differs')


def exact_score(actual, archived, label):
    verify_score(actual); verify_score(archived)
    require(actual['token_logprobs'] == archived['token_logprobs'],
            'Archived reuse blocked: '+label+' differs (including tiny drift); preserve evidence and investigate')


def comparison(archived, ih, basis):
    zero = {t['token_index']: t for t in archived['tokens']}
    full = probability(archived['controls']['reference'])
    rows = [{'token_index': t['token_index'], 'E': zero[t['token_index']]['reconstruction_error'],
             'zero_whole_log': zero[t['token_index']]['intervention_score'],
             'zero_first_probability': full-probability(zero[t['token_index']]['perturbed_score']),
             'ih_first_probability': t['ih_first_probability'], 'retained_probability': t['retained_probability'],
             'full_probability': full, 'remove_all_probability': ih['remove_all_probability'],
             'fps_basis_source': t['token_index'] in basis} for t in ih['tokens']]
    summary = {'positions': len(rows)}
    for name, selected in [('all',rows), ('nonbasis',[r for r in rows if not r['fps_basis_source']])]:
        summary[name] = {'positions': len(selected),
            'rho_E_zero_first': spearman([r['E'] for r in selected], [r['zero_first_probability'] for r in selected]),
            'rho_E_ih_first': spearman([r['E'] for r in selected], [r['ih_first_probability'] for r in selected]),
            'rho_zero_ih_first': spearman([r['zero_first_probability'] for r in selected], [r['ih_first_probability'] for r in selected])}
    return rows, summary


def verify_record(q, archived, feature):
    """Recompute saved arithmetic/geometry and strict-reuse controls on CPU."""
    ih=q['ih'];c=ih['controls'];order=ih['token_execution_order'];count=len(order)
    require(q['case_id'] == archived['case_id'] and q['status'] == ih['status'] == 'passed'
            and c['passed'] and c['archived_reuse_exact'] and c['final_input_exact'], 'Question/reuse controls failed')
    require((ih['layer'],ih['boundary'],ih['window_size'],ih['operator']) ==
            (0,'before_block_1',1,'physical-retain-one-original-positions'), 'Wrong input IH boundary/operator')
    require(count in (32,576) and order == archived['token_execution_order'][:count]
            and [t['token_index'] for t in ih['tokens']] == order, 'Missing/reordered IH positions')
    require(ih['target_token_ids'] == archived['target_token_ids']
            == q['generation_tokens']['generated_token_ids'][:len(archived['target_token_ids'])]
            and q['free_generation']['scoring']['correct'], 'Fresh generated prefix/target differs')
    require(c['tolerance_per_answer_token'] == .001, 'Tolerance changed')
    for key in ['archived_reference','reference','reference_repeat','native_multimodal','clone_noop','reference_after']:
        exact_score(c[key],archived['controls']['reference'],key)
    old={t['token_index']:t for t in archived['tokens']}
    require([r['token_index'] for r in c['zero_audit_repeats']] == archived['token_execution_order'][:3], 'Zero audit positions differ')
    for t in c['zero_audit_repeats']:exact_score(t['score'],old[t['token_index']]['perturbed_score'],'zero audit')
    for key in ['retain_all_noop','remove_all_before','remove_all_repeat','remove_all_after','first_retained_repeat']:
        verify_score(c[key])
        require(len(c[key]['token_logprobs']) == len(archived['target_token_ids']), 'Control target length differs')
    from rq1_native import max_score_delta
    drifts=[max_score_delta(c['reference'],c['retain_all_noop']),
            max_score_delta(c['remove_all_before'],c['remove_all_repeat']),
            max_score_delta(c['remove_all_before'],c['remove_all_after']),
            max_score_delta(ih['tokens'][0]['retained_score'],c['first_retained_repeat'])]
    require(max(drifts) <= .001,'IH repeat/no-op drift exceeds tolerance')
    for field,v in zip(['max_reference_delta','remove_all_repeat_delta','remove_all_after_delta','first_retained_repeat_delta'],drifts):
        assert_close(c[field],v,'Control '+field)
    length,start=archived['expanded_sequence_length'],archived['image_expanded_start']
    require(ih['expanded_sequence_length'] == length and ih['image_expanded_start'] == start
            and ih['target_expanded_start'] == archived['target_expanded_start'], 'Input position mapping differs')
    require(ih['remove_all_kept_positions'] == kept_positions(length,start,[])
            and ih['remove_all_shortened_length'] == length-576, 'Null retained geometry differs')
    assert_close(ih['remove_all_probability'],probability(c['remove_all_before']),'Null probability')
    for t in ih['tokens']:
        i=t['token_index'];verify_score(t['retained_score'])
        require(len(t['retained_score']['token_logprobs']) == len(archived['target_token_ids'])
                and (t['patch_row'],t['patch_col']) == divmod(i,24)
                and t['kept_positions'] == kept_positions(length,start,[i])
                and t['shortened_length'] == length-575,'Retained target/geometry differs')
        assert_close(t['reconstruction_error'],feature['diagnostics']['input']['errors'][i],'Reused E')
        assert_close(t['retained_probability'],probability(t['retained_score']),'Retained probability')
        assert_close(t['ih_first_probability'],t['retained_probability']-ih['remove_all_probability'],'IH arithmetic')
    _,summary=comparison(archived,ih,feature['diagnostics']['input']['basis_indices'])
    assert_close(q['comparison'],summary,'Comparison summary')
    return max(drifts)


def validate_calibration(path, model, vision):
    require(path is not None, 'Full input IH sweep needs the new passed input IH calibration')
    profile = json.loads(PROFILE.read_text()); r = json.loads(Path(path).read_text())
    require(r.get('version') == VERSION and r.get('status') == 'passed'
            and r.get('arguments',{}).get('tokens') == 32
            and r.get('profile_sha256') == sha256(PROFILE)
            and r.get('reference_index_sha256') == profile['reference_index_sha256']
            and r.get('measurement_sources_sha256') == source_hashes()
            and r.get('arguments',{}).get('model') == model
            and r.get('arguments',{}).get('vision_model') == vision
            and r.get('lock_sha256') == sha256(REPO/'environments/original/uv.lock')
            and r.get('implementation',{}).get('native_matches_reference'), 'Calibration sources/runtime/profile differ')
    entries = r.get('questions',[])
    require([e['case_id'] for e in entries] == profile['calibration_case_ids']
            and len(entries) == 6
            and all(e['status'] == 'passed' and e['positions'] == 32 and e['archived_reuse_exact'] for e in entries)
            and r.get('progress',{}).get('completed_retention_scores') == 192,
            'Calibration is incomplete or failed exact archived reuse')
    # Preserve and verify the actual detailed calibration, not a report-only gate.
    details = {}
    reference_root=Path(path).parent/'evidence'
    index_path=reference_root/'references/index.json'
    require(sha256(index_path) == profile['reference_index_sha256'],'Calibration embedded references differ')
    index=json.loads(index_path.read_text());lookup={e['case_id']:e for e in index['cases']}
    for entry in entries:
        relative = 'questions/'+entry['case_id'].replace(':','-')+'.json'
        detail = Path(path).parent/relative
        require(sha256(detail) == r.get('question_sha256',{}).get(relative), 'Calibration detail hash differs')
        q = json.loads(detail.read_text())
        ref_entry=lookup[entry['case_id']]
        for name in [ref_entry['question_file'],ref_entry['feature_file']]:
            require(sha256(file_at(reference_root,name)) == index['files_sha256'][name], 'Calibration archived reference differs')
        archived=json.loads(file_at(reference_root,ref_entry['question_file']).read_text())
        feature=json.loads(file_at(reference_root,ref_entry['feature_file']).read_text())
        verify_record(q,archived,feature)
        require(len(q['ih']['tokens']) == 32, 'Calibration detail is not 32 positions')
        details[entry['case_id']] = q
    return {'passed': True, 'report_sha256': sha256(path), 'case_ids': profile['calibration_case_ids']}, details
