"""Frozen E/random selection x merge/no-merge experiment, with FPS slots fixed."""
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from rq1_protocol import REPO, sha256
from rq2_protocol import require, _credit
from rq2_frozen_protocol import (PROFILE as PARENT_PROFILE, CALIBRATION_IDS, REPEAT_CASE_IDS,
                                validate_bundle as validate_parent_bundle, choose_panel)

VERSION = 'rq2-crossed-input-v1'
PROFILE = REPO / 'research/protocols/rq2_crossed_input_v1.json'
RANDOM_SEEDS = (101, 202, 303)
ORACLE_CASE = 'textvqa:37533'
ORACLE_TARGET = 317
FIXED_REPEAT_CONDITIONS = ('baseline576', 'E-merge', 'E-no-merge', 'random-101-merge', 'random-101-no-merge')
FRESH_CONDITIONS = ('E-merge', 'random-101-merge')
CONDITIONS = {'baseline576': {'selection': 'full', 'merge': False, 'random_seed': None,
                               'replay_arm': 'baseline576'}}
for selection, seeds in [('E', (None,)), ('random', RANDOM_SEEDS)]:
    for seed in seeds:
        for merge in (True, False):
            key = ('E' if selection == 'E' else f'random-{seed}') + ('-merge' if merge else '-no-merge')
            CONDITIONS[key] = {'selection': selection, 'merge': merge, 'random_seed': seed,
                'replay_arm': 'native-input288' if merge else 'native-input288-no-merge'}


def load_profile(path=PROFILE):
    data = json.loads(Path(path).read_text())
    parent = json.loads(PARENT_PROFILE.read_text())
    require(data.get('version') == VERSION and data.get('parent_profile_sha256') == sha256(PARENT_PROFILE),
            'Crossed protocol version or parent profile changed')
    require(data.get('conditions') == CONDITIONS and data.get('random_seeds') == list(RANDOM_SEEDS),
            'Crossed conditions or prespecified seeds changed')
    require(data.get('case_ids') == parent['case_ids'] and data.get('calibration_case_ids') == list(CALIBRATION_IDS)
            and data.get('reference_index_sha256') == parent['reference_index_sha256'], 'Crossed cohort changed')
    require(data.get('input_tokens') == 288 and data.get('basis_tokens') == 10
            and data.get('ordinary_tokens') == 278 and data.get('decoder_layers') == [], 'Crossed budget/stage changed')
    require(data.get('controls') == {'fixed_replay_repeats': list(REPEAT_CASE_IDS),
        'fresh_native_repeats': list(REPEAT_CASE_IDS), 'fresh_native_captures_per_control': 2,
        'fixed_replay_conditions': list(FIXED_REPEAT_CONDITIONS),
        'fresh_conditions': list(FRESH_CONDITIONS)}, 'Crossed repeat controls changed')
    require(data.get('oracle') == {'case_id': ORACLE_CASE, 'target': ORACLE_TARGET,
        'kind': 'post-hoc oracle, not a deployable protection rule',
        'matched_control': 'Other native merge target with closest incoming count; index tie-break'},
        'Crossed oracle policy changed')
    require(data.get('random_policy') == {
        'pool': 'All 576 source positions except the ten native E-selected basis replacement slots',
        'sample': 'Uniform 278 positions without replacement using isolated Python Random',
        'ordering': 'Descending native E, then source index; append the unchanged native ranked basis-slot tail',
        'basis_source_duplicates': 'Allowed at distinct ordinary coordinates, as in native E selection',
        'derived_seed': 'SHA256 of version|image_sha256|archived_seed|random_seed, full digest integer',
        'pairing': 'One identical random selected set for merge/no-merge and every same-image question'},
        'Crossed random selection policy changed')
    return data


def validate_bundle(inputs):
    data = load_profile()
    _, index, manifests, records = validate_parent_bundle(inputs)
    require([r['case']['case_id'] for r in records] == data['case_ids'], 'Crossed accepted input identities changed')
    return data, index, manifests, records


def panel(records, mode):
    require(mode in ('calibration', 'full', 'oracle'), 'Unknown crossed mode')
    rows = choose_panel(records, calibration=mode == 'calibration')
    return [r for r in rows if r['case']['case_id'] == ORACLE_CASE] if mode == 'oracle' else rows


def random_indices(reference, image_sha256, archived_seed, random_seed):
    """Select coordinates independently of Torch RNG, answers and generation."""
    require(random_seed in RANDOM_SEEDS, 'Unknown frozen random seed')
    stage = reference['input']; errors, ranked = stage['errors'], stage['selected_indices']
    require(len(errors) == 576 and len(ranked) == len(set(ranked)) == 288
            and len(stage['basis_indices']) == 10 and all(math.isfinite(x) for x in errors),
            'Invalid native selector reference')
    tail = ranked[-10:]; slots = set(tail)
    material = f'{VERSION}|{image_sha256}|{archived_seed}|{random_seed}'
    digest = hashlib.sha256(material.encode()).hexdigest()
    rng = random.Random(int(digest, 16))
    ordinary = rng.sample([i for i in range(576) if i not in slots], 278)
    ordinary.sort(key=lambda i: (-errors[i], i))
    return ordinary + tail, {'seed_id': random_seed, 'derived_seed_sha256': digest,
        'ordinary_indices_ranked': ordinary, 'basis_slots_ranked': tail,
        'pool_size': 566, 'selected_count': 288, 'selection_uses_answers': False}


def assert_selector_identity(reference, observed, ranked=None):
    """Across selection rules, compact positions differ; basis spatial slots stay fixed."""
    checks = {key: reference[key] == observed[key] for key in ('source_clip', 'source_patches', 'pre_merge')}
    for key in ('errors', 'basis_indices'):
        checks[key] = reference['input'][key] == observed['input'][key]
    # compact_output_position changes under a new selected set and is NOT a basis
    # source/assignment change; compare the original 576-coordinate slot mapping.
    mapping = lambda row: [(b['source_index'], b['assignment_slot']) for b in row['basis_slot_to_source']]
    checks['basis_source_to_spatial_slot'] = mapping(reference) == mapping(observed)
    checks['basis_ranked_tail'] = reference['input']['selected_indices'][-10:] == observed['input']['selected_indices'][-10:]
    if ranked is not None:
        checks['actual_random_indices'] = observed['input']['selected_indices'] == ranked
    require(all(checks.values()), 'Crossed selector identity changed: '+', '.join(k for k,v in checks.items() if not v))
    return {'status': 'passed', **checks}


def summarize(rows):
    groups = defaultdict(lambda: defaultdict(dict)); allowed = set(load_profile()['case_ids'])
    for row in rows:
        cid, condition = row['case_id'], row['condition']
        require(cid in allowed and condition in CONDITIONS, 'Unknown crossed row')
        target = groups[cid.split(':')[0]][condition]
        require(cid not in target, 'Repeated crossed primary row; repeats are controls')
        target[cid] = row
    result = {}
    for dataset, conditions in groups.items():
        require(set(conditions) == set(CONDITIONS), 'Incomplete crossed conditions')
        baseline = conditions['baseline576']; ids = set(baseline)
        require(all(set(values) == ids for values in conditions.values()), 'Crossed conditions have different question IDs')
        correct_base = {cid for cid,r in baseline.items() if r['scoring']['correct']}
        summary = {'unique_questions': len(ids), 'baseline_correct': len(correct_base), 'conditions': {}}
        for name, values in conditions.items():
            correct = {cid for cid,r in values.items() if r['scoring']['correct']}
            scores = [_credit(r['scoring'], dataset) for r in values.values()]
            require(all(math.isfinite(x) and 0 <= x <= 1 for x in scores), 'Invalid crossed scores')
            summary['conditions'][name] = {'correct': len(correct), 'accuracy': len(correct)/len(ids),
                'APR': len(correct & correct_base)/len(correct_base) if correct_base else None,
                'mean_score': sum(scores)/len(scores), 'losses': sorted(correct_base-correct),
                'gains': sorted(correct-correct_base)}
        effects = {}
        for metric in ('accuracy', 'mean_score'):
            get = lambda name: summary['conditions'][name][metric]
            random_no = [get(f'random-{s}-no-merge') for s in RANDOM_SEEDS]
            random_yes = [get(f'random-{s}-merge') for s in RANDOM_SEEDS]
            avg = lambda x: sum(x)/len(x)
            effects[metric] = {'random_no_merge_seed_values': random_no, 'random_merge_seed_values': random_yes,
                'random_no_merge_mean_over_seeds': avg(random_no), 'random_merge_mean_over_seeds': avg(random_yes),
                'E_minus_random_no_merge': get('E-no-merge')-avg(random_no),
                'E_minus_random_merge': get('E-merge')-avg(random_yes),
                'merge_effect_E': get('E-merge')-get('E-no-merge'),
                'merge_effect_random_mean': avg([a-b for a,b in zip(random_yes,random_no)]),
                'interaction': get('E-merge')-get('E-no-merge')-avg([a-b for a,b in zip(random_yes,random_no)])}
        summary['effects'] = effects
        pairs = {}
        for prefix in ('E', *(f'random-{s}' for s in RANDOM_SEEDS)):
            merge, no_merge = conditions[prefix+'-merge'], conditions[prefix+'-no-merge']
            pairs[prefix] = {
                'rescues_by_merging': sorted(cid for cid in ids if merge[cid]['scoring']['correct'] and not no_merge[cid]['scoring']['correct']),
                'losses_caused_by_merging': sorted(cid for cid in ids if no_merge[cid]['scoring']['correct'] and not merge[cid]['scoring']['correct']),
                'both_correct': sorted(cid for cid in ids if merge[cid]['scoring']['correct'] and no_merge[cid]['scoring']['correct']),
                'both_wrong': sorted(cid for cid in ids if not merge[cid]['scoring']['correct'] and not no_merge[cid]['scoring']['correct']),
                'raw_answer_changes': sorted(cid for cid in ids if merge[cid].get('response') != no_merge[cid].get('response'))}
        summary['paired_merge_outcomes'] = pairs
        summary['limits'] = 'Three seed realizations of the same questions; not three independent cohorts or a population accuracy estimate.'
        result[dataset] = summary
    return result
