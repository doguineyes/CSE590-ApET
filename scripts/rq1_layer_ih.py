"""CPU protocol and comparable first-token metrics for the IH layer extension."""

import json
import math
from pathlib import Path

from rq1_layer import CASE_IDS, PROFILE as PARENT_PROFILE, validate_layer_profile
from rq1_protocol import sha256, spearman

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO/'research/protocols/rq1_layer16_ih_v1.json'
VERSION = 'rq1-layer16-ih-v1'
SOURCE_FILES = [
    'scripts/rq1_layer_ih.py', 'scripts/rq1_layer_ih_native.py', 'scripts/run_rq1_layer_ih.py',
    'scripts/rq1_layer.py', 'scripts/rq1_layer_native.py', 'scripts/run_rq1_layer.py',
    'scripts/rq1_native.py', 'scripts/rq1_protocol.py', 'scripts/rq1_validation.py',
    'scripts/run_rq1_input.py', 'scripts/smoke_original_apet.py', 'scripts/original_diagnostics.py',
    'environments/original/uv.lock', 'research/provenance/original_llava_reference.json',
    'research/protocols/rq1_layer16_v1.json', 'research/protocols/rq1_layer16_ih_v1.json']


def validate_profile(inputs):
    profile = json.loads(PROFILE.read_text(encoding='utf-8'))
    if (profile['version'] != VERSION or profile['parent_profile_sha256'] != sha256(PARENT_PROFILE)
            or profile['layer'] != 16 or profile['next_module_index'] != 16
            or profile['window_size'] != 1 or profile['control_tolerance'] != .001
            or profile['operator'] != 'physical-retain-one-original-positions'):
        raise ValueError('IH measurement profile changed')
    parent, data, lookup = validate_layer_profile(inputs)
    return profile, parent, data, lookup


def kept_positions(length, image_start, retained):
    """Keep every nonvisual position and the specified visual positions, sorted."""
    retained = list(retained)
    if (not isinstance(length, int) or not isinstance(image_start, int)
            or image_start < 0 or image_start+576 >= length
            or any(type(i) is not int or not 0 <= i < 576 for i in retained)
            or len(set(retained)) != len(retained)):
        raise ValueError('Invalid physical-retention geometry or duplicate visual index')
    return list(range(image_start))+[image_start+i for i in sorted(retained)]+list(range(image_start+576, length))


def probability(score):
    values = score['token_logprobs']
    if not values or any(not math.isfinite(v) or v > 1e-6 for v in values):
        raise ValueError('Invalid reference-token log probability')
    return math.exp(values[0])


def comparison_rows(layer_record, ih_record):
    zero = {t['token_index']: t for t in layer_record['tokens']}
    if (ih_record['target_token_ids'] != layer_record['target_token_ids']
            or [t['token_index'] for t in ih_record['tokens']] != layer_record['token_execution_order']):
        raise ValueError('IH and zeroing targets/positions differ')
    full = probability(layer_record['controls']['reference'])
    null = probability(ih_record['controls']['remove_all_before'])
    rows = []
    for token in ih_record['tokens']:
        i = token['token_index']; original = zero[i]
        retained = probability(token['retained_score'])
        rows.append({'token_index': i, 'E': original['reconstruction_error'],
            'zero_mean_log': original['intervention_score'],
            'zero_first_probability': full-probability(original['perturbed_score']),
            'ih_first_probability': retained-null, 'retained_probability': retained,
            'full_probability': full, 'remove_all_probability': null})
    return rows


def comparison_summary(layer_record, ih_record, basis):
    rows = comparison_rows(layer_record, ih_record)
    result = {'positions': len(rows), 'metric': 'first reference-token probability differences',
              'first_reference_token_id': ih_record['target_token_ids'][0]}
    for name, selected in [('all', rows), ('nonbasis', [r for r in rows if r['token_index'] not in basis])]:
        result[name] = {'positions': len(selected),
            'rho_E_zero_first': spearman([r['E'] for r in selected], [r['zero_first_probability'] for r in selected]),
            'rho_E_ih_first': spearman([r['E'] for r in selected], [r['ih_first_probability'] for r in selected]),
            'rho_zero_ih_first': spearman([r['zero_first_probability'] for r in selected], [r['ih_first_probability'] for r in selected]),
            'unique_ih_scores': len({r['ih_first_probability'] for r in selected})}
    return result


def validate_calibration(path, model, vision_model):
    if path is None:
        raise ValueError('IH full sweep requires the new passed IH calibration report')
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    args = report.get('arguments', {})
    entries = report.get('ih_questions', [])
    native = report.get('implementation', {}).get('native_source_sha256', {})
    reference_native = json.loads((REPO/'research/provenance/original_llava_reference.json').read_text(encoding='utf-8'))['native_source_sha256']
    if (report.get('version') != VERSION or report.get('status') != 'passed'
            or args.get('tokens') != 32 or args.get('model') != model or args.get('vision_model') != vision_model
            or report.get('ih_profile_sha256') != sha256(PROFILE)
            or report.get('profile_sha256') != sha256(PARENT_PROFILE)
            or not report.get('implementation', {}).get('native_matches_reference')
            or native.keys() != reference_native.keys()
            or report.get('ih_progress', {}).get('completed_questions') != 6
            or report.get('ih_progress', {}).get('completed_retention_scores') != 192
            or report.get('progress', {}).get('completed_patch_stage_scores') != 384
            or report.get('total_patch_method_scores') != 576
            or [q['case_id'] for q in report.get('questions', [])] != CASE_IDS
            or [q['case_id'] for q in entries] != CASE_IDS
            or any(q.get('status') != 'passed' for q in report.get('questions', []))
            or any(q.get('status') != 'passed' or q.get('positions') != 32
                   or not q.get('controls_passed') for q in entries)):
        raise ValueError('Full IH sweep requires the matching six-case 32-position IH calibration')
    required = SOURCE_FILES+list(native)
    for relative in required:
        if report.get('evidence_sha256', {}).get('sources/'+relative) != sha256(REPO/relative):
            raise ValueError('IH calibration measurement source/environment changed: '+relative)
    return {'passed': True, 'report_sha256': sha256(path), 'questions': len(entries),
            'ih_profile_sha256': sha256(PROFILE)}
