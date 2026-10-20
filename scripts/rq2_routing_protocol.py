"""Frozen input-stage routing/protection controls; no model dependencies."""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random

from rq1_protocol import REPO, sha256
from rq2_protocol import require, _credit
from rq2_frozen_protocol import (PROFILE as PARENT_PROFILE, CALIBRATION_IDS,
    REPEAT_CASE_IDS, validate_bundle as validate_parent_bundle, choose_panel)

VERSION = 'rq2-routing-input-v1'
PROFILE = REPO / 'research/protocols/rq2_routing_input_v1.json'
RANDOM_SEEDS = (101, 202, 303)
PROTECTED_COUNT = 28
CONDITIONS = {
    'baseline576': {'routing': 'full', 'seed': None, 'replay_arm': 'baseline576'},
    'native-nearest': {'routing': 'native', 'seed': None, 'replay_arm': 'native-input288'},
    'no-merge': {'routing': 'none', 'seed': None, 'replay_arm': 'native-input288-no-merge'},
}
for seed in RANDOM_SEEDS:
    CONDITIONS[f'shuffled-{seed}'] = {'routing': 'shuffle', 'seed': seed, 'replay_arm': 'native-input288'}
CONDITIONS['protect-high-E'] = {'routing': 'protect-E', 'seed': None, 'replay_arm': 'native-input288'}
for seed in RANDOM_SEEDS:
    CONDITIONS[f'protect-random-{seed}'] = {'routing': 'protect-random', 'seed': seed, 'replay_arm': 'native-input288'}
FIXED_REPEAT_CONDITIONS = ('baseline576', 'native-nearest', 'no-merge', 'shuffled-101',
                           'protect-high-E', 'protect-random-101')
FRESH_CONDITIONS = ('native-nearest', 'shuffled-101', 'protect-high-E', 'protect-random-101')
POLICY = {
    'selection': 'Native E top288, FPS basis sources and spatial replacement slots fixed',
    'shuffle': 'Permute the native destination list across spatially ordered dropped sources; target incoming counts unchanged',
    'protect_E': '28 highest-E ordinary targets; descending E, ascending coordinate tie-break',
    'protect_random': 'Uniform 28 ordinary targets without replacement; same count as E protection',
    'rerouting': 'Native cosine argmax over unprotected ordinary targets; original target order breaks ties',
    'averaging': 'Original native FP16 scatter_add and division by 1 + actual incoming count',
    'rng': 'Isolated Python Random, full SHA256 integer of version|image_sha256|archived_seed|routing|seed',
    'pairing': 'First unconditional realization per image/condition, shared by all same-image questions',
}


def load_profile(path=PROFILE):
    data = json.loads(Path(path).read_text()); parent = json.loads(PARENT_PROFILE.read_text())
    require(data.get('version') == VERSION and data.get('parent_profile_sha256') == sha256(PARENT_PROFILE),
            'Routing protocol/parent changed')
    require(data.get('case_ids') == parent['case_ids'] and data.get('calibration_case_ids') == list(CALIBRATION_IDS)
            and data.get('reference_index_sha256') == parent['reference_index_sha256'], 'Routing cohort changed')
    require(data.get('conditions') == CONDITIONS and data.get('random_seeds') == list(RANDOM_SEEDS)
            and data.get('protected_count') == PROTECTED_COUNT and data.get('policy') == POLICY,
            'Routing/protection policy changed')
    require(data.get('input_tokens') == 288 and data.get('basis_tokens') == 10
            and data.get('decoder_layers') == [], 'Routing budget/stage changed')
    require(data.get('controls') == {'repeat_case_ids': list(REPEAT_CASE_IDS),
        'fixed_replay_conditions': list(FIXED_REPEAT_CONDITIONS), 'fresh_conditions': list(FRESH_CONDITIONS),
        'fresh_captures_per_condition': 2}, 'Routing repeat controls changed')
    return data


def validate_bundle(inputs):
    data = load_profile()
    _, index, manifests, records = validate_parent_bundle(inputs)
    require([r['case']['case_id'] for r in records] == data['case_ids'], 'Routing input order changed')
    return data, index, manifests, records


def panel(records, mode):
    require(mode in ('calibration', 'full'), 'Unknown routing mode')
    return choose_panel(records, calibration=mode == 'calibration')


def routing_plan(reference, image_sha256, archived_seed, condition):
    spec = CONDITIONS[condition]; kind = spec['routing']
    require(kind in ('shuffle', 'protect-E', 'protect-random'), 'Condition needs no routing override')
    stage, graph = reference['input'], reference['input_merge']
    targets, sources = graph['target_indices'], graph['dropped_indices']
    destinations = graph['assigned_target_indices']; ranked = stage['selected_indices']
    require(len(targets) == len(set(targets)) == 278 and targets == ranked[:-10]
            and len(ranked) == len(set(ranked)) == 288 and len(stage['errors']) == 576
            and all(math.isfinite(x) for x in stage['errors']), 'Invalid native routing reference')
    require(sources == sorted(set(range(576))-set(ranked)) and len(sources) == len(destinations) == 288
            and set(destinations) <= set(targets)
            and [Counter(destinations)[i] for i in targets] == graph['merge_counts'], 'Invalid native graph')
    digest = hashlib.sha256(f'{VERSION}|{image_sha256}|{archived_seed}|{kind}|{spec["seed"]}'.encode()).hexdigest()
    rng = random.Random(int(digest, 16)); protected = []; assigned = None
    if kind == 'shuffle':
        assigned = list(destinations); rng.shuffle(assigned)
    elif kind == 'protect-E':
        protected = sorted(targets, key=lambda i: (-stage['errors'][i], i))[:PROTECTED_COUNT]
    else:
        protected = sorted(rng.sample(targets, PROTECTED_COUNT))
    return {'condition': condition, 'routing': kind, 'seed_id': spec['seed'],
        'derived_seed_sha256': digest, 'target_indices': targets, 'dropped_indices': sources,
        'native_assigned_target_indices': destinations, 'assigned_target_indices': assigned,
        'protected_targets': protected, 'uses_gold_or_predictions': False}


def assert_routing_identity(reference, observed, plan, protected_exact=None):
    checks = {key: reference[key] == observed[key] for key in
        ('source_clip', 'source_patches', 'pre_merge', 'input', 'basis_slot_to_source',
         'merge_argument_indices', 'merge_basis_tokens', 'merge_scaling')}
    checks['mask'] = reference['encoded']['mask'] == observed['encoded']['mask']
    old, new = reference['input_merge'], observed['input_merge']
    for key in ('target_indices', 'dropped_indices', 'basis_slots', 'basis_assignment_slots', 'scaling'):
        checks[key] = old[key] == new[key]
    assigned = new['assigned_target_indices']; protected = set(plan['protected_targets'])
    checks['complete_valid_routing'] = (len(assigned) == len(new['dropped_indices'])
        and set(assigned) <= set(new['target_indices'])-protected
        and [Counter(assigned)[i] for i in new['target_indices']] == new['merge_counts'])
    if plan['routing'] == 'shuffle':
        checks['prespecified_permutation'] = assigned == plan['assigned_target_indices']
        checks['incoming_counts_preserved'] = new['merge_counts'] == old['merge_counts']
    else:
        checks['protected_pre_post_exact'] = protected_exact is True
        checks['protected_count'] = len(protected) == PROTECTED_COUNT
    require(all(checks.values()), 'Routing identity failed: '+', '.join(k for k,v in checks.items() if not v))
    changed = sum(a != b for a,b in zip(old['assigned_target_indices'], assigned))
    return {'status': 'passed', **checks, 'changed_assignments': changed,
        'unchanged_assignments': len(assigned)-changed,
        'protected_native_incoming_total': sum(n for t,n in zip(old['target_indices'], old['merge_counts']) if t in protected),
        'incoming_counts_before': old['merge_counts'], 'incoming_counts_after': new['merge_counts']}


def summarize(rows):
    groups = defaultdict(lambda: defaultdict(dict)); allowed = set(load_profile()['case_ids'])
    for row in rows:
        cid, condition = row['case_id'], row['condition']
        require(cid in allowed and condition in CONDITIONS, 'Unknown routing row')
        bucket = groups[cid.split(':')[0]][condition]
        require(cid not in bucket, 'Repeated primary routing row'); bucket[cid] = row
    result = {}
    for dataset, arms in groups.items():
        require(set(arms) == set(CONDITIONS), 'Incomplete routing conditions')
        ids = set(arms['baseline576'])
        require(all(set(a) == ids for a in arms.values()), 'Routing conditions have different questions')
        base = {cid for cid,r in arms['baseline576'].items() if r['scoring']['correct']}
        conditions = {}
        for name, values in arms.items():
            correct = {cid for cid,r in values.items() if r['scoring']['correct']}
            scores = [_credit(r['scoring'], dataset) for r in values.values()]
            require(all(math.isfinite(x) and 0 <= x <= 1 for x in scores), 'Invalid routing credit')
            conditions[name] = {'correct': len(correct), 'accuracy': len(correct)/len(ids),
                'APR': len(correct & base)/len(base) if base else None, 'mean_score': sum(scores)/len(scores),
                'losses': sorted(base-correct), 'gains': sorted(correct-base)}
        effects = {}
        for metric in ('accuracy', 'mean_score'):
            get = lambda name: conditions[name][metric]
            shuffle = [get(f'shuffled-{s}') for s in RANDOM_SEEDS]
            protect = [get(f'protect-random-{s}') for s in RANDOM_SEEDS]
            avg = lambda x: sum(x)/len(x)
            effects[metric] = {'shuffled_seed_values': shuffle, 'random_protected_seed_values': protect,
                'native_minus_shuffled_mean': get('native-nearest')-avg(shuffle),
                'high_E_protection_minus_native': get('protect-high-E')-get('native-nearest'),
                'random_protection_mean_minus_native': avg(protect)-get('native-nearest'),
                'high_E_minus_random_protection_mean': get('protect-high-E')-avg(protect),
                'native_minus_no_merge': get('native-nearest')-get('no-merge')}
        paired = {}
        for name, values in arms.items():
            if name in ('baseline576', 'native-nearest'): continue
            native = arms['native-nearest']
            paired[name] = {'rescues_vs_native': sorted(cid for cid in ids if values[cid]['scoring']['correct'] and not native[cid]['scoring']['correct']),
                'new_losses_vs_native': sorted(cid for cid in ids if native[cid]['scoring']['correct'] and not values[cid]['scoring']['correct']),
                'both_correct': sorted(cid for cid in ids if native[cid]['scoring']['correct'] and values[cid]['scoring']['correct']),
                'both_wrong': sorted(cid for cid in ids if not native[cid]['scoring']['correct'] and not values[cid]['scoring']['correct']),
                'raw_answer_changes': sorted(cid for cid in ids if native[cid].get('response') != values[cid].get('response'))}
        result[dataset] = {'unique_questions': len(ids), 'baseline_correct': len(base),
            'conditions': conditions, 'effects': effects, 'paired_vs_native': paired,
            'limits': 'Selected diagnostic panel; seeds repeat the same questions. Protection reroutes sources and changes recipient loads; its effect is a combined policy effect.'}
    return result
