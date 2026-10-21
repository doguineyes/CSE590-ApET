"""Fixed GQA development800 staged comparison, with image-clustered statistics."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
from rq1_protocol import REPO, sha256
from rq2_protocol import require
from rq2_staged_protocol import CONDITIONS, SCHEDULES, RANDOM_SEEDS, VERSION as RANDOM_VERSION
from rq2_staged_textvqa_protocol import load_calibration_reference, CALIBRATION_REFERENCE
from gqa_scoring import INSTRUCTION
VERSION = 'rq2-staged-gqa800-v1'
PROFILE = REPO / 'research/protocols/rq2_staged_gqa800_v1.json'
MANIFEST_SHA256 = '99860b9c8768a02a775d515e649234a6d2ea753448e25325ab759978d4c14a24'
STUDY_SHA256 = '81a49ac9a5f534aa43c1de51eb843284e53515d4d46773f933948a4afea3f962'
STUDY_ID = '3153bd275dbefd346e132c18ac16d7cbe46e5534c4f9e84fed2786abe038f512'
REPEAT_IDS = ('gqa:20542858', 'gqa:201636988')
TENSOR_IDS = REPEAT_IDS
PROBE_CASE = REPEAT_IDS[0]
FRESH_CASE = REPEAT_IDS[1]
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = (59022, 2000)

def load_profile(path=PROFILE):
    p = json.loads(Path(path).read_text())
    require(p.get('version') == VERSION and p.get('manifest_sha256') == MANIFEST_SHA256 and (p.get('study_sha256') == STUDY_SHA256) and (p.get('study_id') == STUDY_ID) and (p.get('role') == 'development') and (p.get('questions') == 800) and (p.get('images') == 100) and (p.get('conditions') == CONDITIONS) and (p.get('schedules') == [list(s) for s in SCHEDULES]) and (p.get('layer') == 16) and (p.get('native_basis_tokens') == 10) and (p.get('random_policy_version') == RANDOM_VERSION) and (p.get('random_seeds') == list(RANDOM_SEEDS)) and (p.get('calibration_reference_sha256') == sha256(CALIBRATION_REFERENCE)), 'Frozen GQA development/staged protocol changed')
    ids, images, assignment = (p['case_ids'], p['image_ids'], p['case_image_ids'])
    require(len(ids) == len(set(ids)) == 800 and len(images) == len(set(images)) == 100 and (set(assignment) == set(ids)) and (Counter(assignment.values()) == dict.fromkeys(images, 8)) and ([assignment[ids[i]] for i in range(0, 800, 8)] == images) and all(([assignment[c] for c in ids[i:i + 8]] == [assignment[ids[i]]] * 8 for i in range(0, 800, 8))), 'GQA image groups or ordered question identities changed')
    require(p['image_shards'] == [images[i:i + 25] for i in range(0, 100, 25)] and p['identity_case_ids'] == [ids[i * 8] for i in (0, 1, 25, 26, 50, 75)] and (p['fixed_repeat_case_ids'] == list(REPEAT_IDS)) and (p['tensor_case_ids'] == list(TENSOR_IDS)) and (p['cache_probe_case_id'] == PROBE_CASE) and (p['fresh_raw_native_case_id'] == FRESH_CASE) and (p['bootstrap'] == {'unit': 'image', 'replicates': BOOTSTRAP_REPLICATES, 'seed': BOOTSTRAP_SEED}) and (p['generation'] == {'conversation': 'vicuna_v1', 'max_new_tokens': 16, 'min_new_tokens': 0, 'do_sample': False, 'use_cache': True, 'batch_size': 1}), 'GQA shards, controls or generation changed')
    load_calibration_reference()
    return p

def selected_ids(profile, shards):
    require(isinstance(shards, (list, tuple)) and list(shards) == sorted(set(shards)) and (len(shards) > 0) and all((type(i) is int and 0 <= i < 4 for i in shards)), 'Choose unique ordered image shards 0..3')
    images = {image for i in shards for image in profile['image_shards'][i]}
    return [cid for cid in profile['case_ids'] if profile['case_image_ids'][cid] in images]

def adapt_manifest(manifest):
    """Only adapt field names; retain all original questions/types and image seeds."""
    cases = []
    for q in manifest['questions']:
        meta = manifest['images'][q['imageId']]
        cases.append(dict(case_id='gqa:' + q['id'], question_id=q['id'], image_id=q['imageId'], image_sha256=meta['sha256'], compression_seed=meta['compression_seed'], question=q['question'], generation_question=q['question'] + '\n' + INSTRUCTION, canonical_target=q['answer'], gold_answers=[q['answer']], types=q['types'], known_repeatability_case=q['id'] == '2058428'))
    return {'dataset': 'gqa', 'cases': cases, 'images': {key: {**m, 'size': m['image_size']} for key, m in manifest['images'].items()}}

def validate_inputs(path):
    path = Path(path).resolve()
    require(sha256(path) == MANIFEST_SHA256, 'Require the unchanged development800 manifest; evaluation/filtered sets refused')
    from eval_original_gqa_subset import load_manifest
    manifest, study = load_manifest(path)
    p = load_profile()
    require(manifest['study']['role'] == 'development' and sha256(study) == STUDY_SHA256 and (manifest['study']['id'] == STUDY_ID), 'Wrong GQA development study')
    data = adapt_manifest(manifest)
    require([c['case_id'] for c in data['cases']] == p['case_ids'] and {c['case_id']: c['image_id'] for c in data['cases']} == p['case_image_ids'], 'Development cohort identities changed')
    return (p, data, study)

def clustered_difference(left, right, ids, image_for, *, intervals):
    """Resample image means; admitted images each contain all eight questions."""
    groups = defaultdict(list)
    for cid in ids:
        groups[image_for[cid]].append(left[cid] - right[cid])
    means = [sum(v) / len(v) for _, v in sorted(groups.items())]
    result = dict(questions=len(ids), images=len(means), mean_accuracy_delta=sum(means) / len(means))
    if intervals:
        rng = random.Random(BOOTSTRAP_SEED)
        draws = sorted((sum((means[rng.randrange(len(means))] for _ in means)) / len(means) for _ in range(BOOTSTRAP_REPLICATES)))

        def quantile(q):
            position = (len(draws) - 1) * q
            lo = int(position)
            hi = min(lo + 1, len(draws) - 1)
            return draws[lo] + (draws[hi] - draws[lo]) * (position - lo)
        result['paired_image_bootstrap_95_ci'] = [quantile(0.025), quantile(0.975)]
    return result

def _condition_summary(groups, ids):
    base = {c for c in ids if groups['baseline576'][c]['scoring']['correct']}
    result = {}
    for condition, g in groups.items():
        good = {c for c in ids if g[c]['scoring']['correct']}
        result[condition] = dict(correct=len(good), accuracy=len(good) / len(ids), baseline_correct_preserved=len(good & base), APR=len(good & base) / len(base) if base else None, losses=sorted(base - good), gains=sorted(good - base), empty_response_ids=[c for c in ids if not g[c]['response'].strip()], output_cap_ids=[c for c in ids if g[c].get('generated_tokens') == 16])
    return result

def summarize(rows, *, expected_ids=None, intervals=False):
    p = load_profile()
    groups = defaultdict(dict)
    for row in rows:
        cid, condition = (row['case_id'], row['condition'])
        require(cid in p['case_image_ids'] and condition in CONDITIONS and (not row.get('control_only', False)), 'Unknown GQA primary or included control')
        require(cid not in groups[condition] and type(row['scoring'].get('correct')) is bool, 'Duplicate/invalid GQA primary')
        groups[condition][cid] = row
    if not rows:
        return {}
    ids = [c for c in p['case_ids'] if c in groups['baseline576']]
    require(set(groups) == set(CONDITIONS) and all((set(g) == set(ids) for g in groups.values())), 'Incomplete matched GQA arms')
    require(all((n == 8 for n in Counter((p['case_image_ids'][c] for c in ids)).values())), 'Summaries require complete eight-question image groups')
    if expected_ids is not None:
        require(ids == list(expected_ids), 'Missing/filtered requested GQA shard questions')
    conditions = _condition_summary(groups, ids)
    scores = {condition: {c: float(g[c]['scoring']['correct']) for c in ids} for condition, g in groups.items()}
    pairs = []
    random_summary = {}
    for a, b in SCHEDULES:
        schedule = f'{a}-to-{b}'
        native = 'native-' + schedule
        avg = 'pure-random-' + schedule + '-seed-mean'
        scores[avg] = {c: sum((scores[f'pure-random-{schedule}-s{s}'][c] for s in RANDOM_SEEDS)) / 3 for c in ids}
        pairs.append((native, 'baseline576'))
        pairs.extend(((f'pure-random-{schedule}-s{s}', native) for s in RANDOM_SEEDS))
        pairs.append((avg, native))
        random_summary[schedule] = {}
        for metric in ('accuracy', 'APR'):
            values = [conditions[f'pure-random-{schedule}-s{s}'][metric] for s in RANDOM_SEEDS]
            random_summary[schedule][metric] = dict(seed_values=dict(zip(map(str, RANDOM_SEEDS), values)), mean=sum(values) / 3 if None not in values else None, min=min(values) if None not in values else None, max=max(values) if None not in values else None)
    pairs.append(('native-96-to-32', 'native-288-to-96'))
    pairs.extend(((f'pure-random-96-to-32-s{s}', f'pure-random-288-to-96-s{s}') for s in RANDOM_SEEDS))
    pairs.append(('pure-random-96-to-32-seed-mean', 'pure-random-288-to-96-seed-mean'))
    comparisons = {}
    for left, right in pairs:
        result = clustered_difference(scores[left], scores[right], ids, p['case_image_ids'], intervals=intervals)
        if left in groups and right in groups:
            result.update(rescues=[c for c in ids if scores[left][c] > scores[right][c]], losses=[c for c in ids if scores[left][c] < scores[right][c]])
        comparisons[left + ' minus ' + right] = result
    types = {}
    for field in ('structural', 'semantic', 'detailed'):
        grouped = defaultdict(list)
        for c in ids:
            grouped[groups['baseline576'][c]['types'][field]].append(c)
        types[field] = {key: dict(questions=len(v), conditions=_condition_summary(groups, v)) for key, v in sorted(grouped.items())}
    return dict(questions=len(ids), images=len(ids) // 8, full_development800=len(ids) == 800, baseline_correct=conditions['baseline576']['correct'], conditions=conditions, random_seed_summary=random_summary, paired_comparisons=comparisons, type_summaries=types, limits='Fixed image-balanced development subset, not full GQA or reserved evaluation. Image-clustered intervals condition on three seeds. Same-image questions and seeds are not independent samples. Whole-procedure comparison; no speed claim.')
