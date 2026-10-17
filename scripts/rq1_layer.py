"""CPU protocol and paired summaries for the first RQ1 decoder-boundary pilot."""

import json
from pathlib import Path

from rq1_protocol import sha256, spearman, token_order
from rq1_validation import PROFILE as VALIDITY_PROFILE, validate_profile

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / 'research/protocols/rq1_layer16_v1.json'
VERSION = 'rq1-layer16-v1'
CASE_IDS = ['gqa:202218829', 'gqa:202218593', 'mmstar:895', 'mmstar:136',
            'textvqa:35962', 'textvqa:35670']
LAYER = 16  # One based: after module index 15, before module index 16.


def validate_layer_profile(inputs, path=PROFILE):
    profile = json.loads(Path(path).read_text())
    if (profile['version'] != VERSION or profile['layer'] != LAYER
            or profile['module_index'] != LAYER - 1 or profile['seed'] != 590
            or profile['control_tolerance'] != .001
            or profile['observation_keep_tokens'] != 288 or profile['basis_tokens'] != 10
            or profile['validity_profile_sha256'] != sha256(VALIDITY_PROFILE)):
        raise ValueError('Layer protocol or its frozen source changed')
    parent, data, lookup = validate_profile(inputs)
    sources = {e['case_id']: e for e in parent['perturbations']}
    if [e['case_id'] for e in profile['cases']] != CASE_IDS:
        raise ValueError('Expected the fixed six-question layer panel')
    for entry in profile['cases']:
        source = sources[entry['case_id']]
        for key in ['image_id', 'image_sha256', 'canonical_target', 'measurement_target',
                    'target_token_ids', 'seed', 'archived_role']:
            if entry[key] != source[key]:
                raise ValueError('Layer case identity differs: ' + key)
        expected = token_order(entry['image_sha256'], profile['seed'])
        if entry['token_order'] != expected:
            raise ValueError('Layer patch order must be independent of observed E/I')
    return profile, data, lookup


def selected_tokens(entry, count):
    if count not in (32, 576):
        raise ValueError('Use the 32-position calibration or explicit 576-position sweep')
    order = entry['token_order']
    if len(order) != 576 or set(order) != set(range(576)):
        raise ValueError('Expected a complete, unique patch permutation')
    return order[:count]


def paired_summary(input_record, layer_record, input_basis, layer_basis):
    """Compare matching positions; never pool tokens from different questions."""
    a = {t['token_index']: t for t in input_record['tokens']}
    b = {t['token_index']: t for t in layer_record['tokens']}
    if len(a) != len(input_record['tokens']) or len(b) != len(layer_record['tokens']) or a.keys() != b.keys():
        raise ValueError('Stage comparison requires identical, unique patch positions')
    if input_record['target_token_ids'] != layer_record['target_token_ids']:
        raise ValueError('Stage comparison requires identical reference token IDs')
    input_basis, layer_basis = set(input_basis), set(layer_basis)
    indices = sorted(a)
    common_nonbasis = [i for i in indices if i not in input_basis | layer_basis]

    def rho(rows, ids, absolute=False):
        return spearman([rows[i]['reconstruction_error'] for i in ids],
                        [abs(rows[i]['intervention_score']) if absolute else rows[i]['intervention_score'] for i in ids])

    result = {'measured_positions': len(indices), 'common_nonbasis_positions': len(common_nonbasis),
              'rho_E_between_stages': spearman([a[i]['reconstruction_error'] for i in indices],
                                              [b[i]['reconstruction_error'] for i in indices]),
              'rho_I_between_stages': spearman([a[i]['intervention_score'] for i in indices],
                                              [b[i]['intervention_score'] for i in indices])}
    for name, rows, basis in [('input', a, input_basis), ('layer16', b, layer_basis)]:
        result[name] = {'rho_E_I_all': rho(rows, indices),
                        'rho_E_I_nonbasis': rho(rows, [i for i in indices if i not in basis]),
                        'rho_E_I_common_nonbasis': rho(rows, common_nonbasis),
                        'rho_E_abs_I_auxiliary': rho(rows, indices, absolute=True)}
    return result


def validate_calibration(path, profile, model, vision_model):
    """A full sweep must identify a passed calibration of this exact measurement."""
    if path is None:
        raise ValueError('The 576-position sweep requires --calibration-report from the passed six-case 32-position run')
    report = json.loads(Path(path).read_text())
    args = report.get('arguments', {})
    if (report.get('status') != 'passed' or report.get('version') != VERSION
            or args.get('tokens') != 32 or args.get('model') != model or args.get('vision_model') != vision_model
            or report.get('profile_sha256') != sha256(PROFILE)
            or [q['case_id'] for q in report.get('questions', [])] != CASE_IDS
            or any(q.get('status') != 'passed' for q in report.get('questions', []))):
        raise ValueError('Full sweep requires the matching passed six-question calibration')
    required = ['scripts/rq1_layer.py', 'scripts/rq1_layer_native.py', 'scripts/run_rq1_layer.py',
                'scripts/rq1_native.py', 'scripts/rq1_protocol.py', 'scripts/rq1_validation.py',
                'scripts/run_rq1_input.py', 'scripts/smoke_original_apet.py', 'scripts/original_diagnostics.py',
                'environments/original/uv.lock', 'research/provenance/original_llava_reference.json']
    required += list(report.get('implementation', {}).get('native_source_sha256', {}))
    for relative in required:
        if report.get('evidence_sha256', {}).get('sources/' + relative) != sha256(REPO / relative):
            raise ValueError('Calibration measurement source/environment changed: ' + relative)
    return {'report_sha256': sha256(path), 'passed': True, 'questions': len(CASE_IDS),
            'profile_sha256': sha256(PROFILE)}
