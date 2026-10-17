"""Frozen RQ1 validity checks: CPU protocol, option mapping and summaries."""

from collections import Counter
import json
import math
from pathlib import Path
import re
import statistics

from mmstar_scoring import option_texts, score_answer
from rq1_protocol import SOURCES, TEXTVQA_SELECTION, load_manifest, measurement_target, sha256, spearman

REPO = Path(__file__).resolve().parents[1]
PROFILE = REPO / 'research/protocols/rq1_validation_v1.json'
VERSION = 'rq1-validity-v1'
METHODS = ('zero', 'half', 'neighbor_mean')
CONDITIONS = ('real', 'question_only', 'blank')
SHORTCUT_CASE_IDS = ['mmstar:1014','mmstar:1030','mmstar:1188','mmstar:776','mmstar:1372','mmstar:136']
PERTURBATION_CASE_IDS = ['gqa:202218829','gqa:202218593','mmstar:895','mmstar:136','textvqa:35962','textvqa:35670']
METHOD_DEFINITIONS = {
    'zero':'Set one complete post-projector patch vector to zero, preserving sequence/mask/positions.',
    'half':'Replace one patch vector by 0.5 times its original value; a dose check along the zeroing direction.',
    'neighbor_mean':'Replace by the FP32 mean of original, unmodified cardinal neighbors inside the 24x24 grid (no wrapping), then cast to the original FP16 dtype. Photo/padding regions may mix at boundaries.',
    'score':'Signed I = original full-context reference mean teacher-forced log probability minus perturbed score; no EOS, no decoder pruning.',
    'sham':'Strength zero for each method returns a clone exactly equal to the full reference.',
    'interpretation':'Perturbations differ in size and may preserve information. They are diagnostics, not ground-truth importance or native ApET compression.'}
INSTRUCTION = 'Answer with only the option letter (A, B, C, or D).'


def rotate_options(question, gold, shift):
    """Move answer content only; labels, punctuation, whitespace and stem stay fixed."""
    options = option_texts(question)
    if len(options) != 4 or any(not text or text.casefold() == 'nan' for text in options.values()):
        raise ValueError('Validation expects exactly four real options')
    if gold not in options or shift not in range(4):
        raise ValueError('Invalid gold or cyclic shift')
    matches = list(re.finditer(r'(?<!\w)(?:\(([A-D])\)|([A-D])[:.)])\s*', question))
    old_labels = list('ABCD')
    rotated = old_labels[shift:] + old_labels[:shift]
    mapping = dict(zip(old_labels, rotated))
    result = question
    for i in reversed(range(4)):
        start = matches[i].end()
        end = matches[i+1].start() if i < 3 else len(question)
        segment = question[start:end]
        offset = len(segment) - len(segment.lstrip())
        text = options[old_labels[i]]
        if segment[offset:offset+len(text)] != text:
            raise ValueError('Option parser did not locate exact original content')
        begin = start + offset
        result = result[:begin] + options[rotated[i]] + result[begin+len(text):]
    if shift == 0 and result != question:
        raise ValueError('Identity ordering changed the original question')
    return {'question': result, 'shift': shift, 'display_to_original': mapping,
            'gold_display': next(label for label, old in mapping.items() if old == gold),
            'gold_original': gold, 'gold_content': options[gold], 'original_options': options}


def score_rotated(response, variant):
    scored = score_answer(response, variant['question'], variant['gold_display'])
    original = variant['display_to_original'].get(scored['prediction'])
    return {**scored, 'selected_original_label': original,
            'selected_content': variant['original_options'].get(original),
            'gold_display': variant['gold_display'], 'gold_original': variant['gold_original']}


def neighbors(index):
    if not isinstance(index, int) or not 0 <= index < 576:
        raise ValueError('Patch index must be in 0..575')
    row, col = divmod(index, 24)
    return [r*24+c for r,c in [(row-1,col),(row+1,col),(row,col-1),(row,col+1)]
            if 0 <= r < 24 and 0 <= c < 24]


def validate_profile(inputs, path=PROFILE):
    profile = json.loads(Path(path).read_text())
    if profile['version'] != VERSION or tuple(profile['methods']) != METHODS or profile['method_definitions'] != METHOD_DEFINITIONS:
        raise ValueError('Unexpected validation protocol')
    if profile['seed'] != 590 or profile['control_tolerance'] != .001:
        raise ValueError('Unexpected fixed seed/control policy')
    data = {name: load_manifest(Path(inputs)/name/'manifest.json', name) for name in SOURCES}
    lookup = {case['case_id']: case for d in data.values() for case in d['cases']}
    if sha256(TEXTVQA_SELECTION) != profile['textvqa_selection_sha256']:
        raise ValueError('TextVQA frozen screening selection changed')
    text_selection={e['case_id']:e for e in json.loads(TEXTVQA_SELECTION.read_text())['cases']}
    for dataset, info in profile['inputs'].items():
        if info['manifest_sha256'] != SOURCES[dataset][1]:
            raise ValueError('Frozen plan uses a different accepted manifest')
    shortcuts = profile['shortcuts']['case_ids']
    if (shortcuts != SHORTCUT_CASE_IDS or profile['shortcuts']['conditions'] != list(CONDITIONS)
            or profile['shortcuts']['shifts'] != list(range(4))):
        raise ValueError('Expected six fixed shortcut cases')
    for cid in shortcuts:
        case = lookup[cid]
        if not cid.startswith('mmstar:'):
            raise ValueError('Shortcut panel must be MMStar')
        for shift in range(4):
            rotate_options(case['question'], case['canonical_target'], shift)
        if case['generation_question'] != case['question'] + '\n' + INSTRUCTION:
            raise ValueError('Unexpected original MMStar prompt instruction')
    entries = profile['perturbations']
    if [e['case_id'] for e in entries] != PERTURBATION_CASE_IDS:
        raise ValueError('Expected six fixed perturbation questions')
    for entry in entries:
        case = lookup[entry['case_id']]
        if case['case_id'].startswith('textvqa:'):
            screened=text_selection[case['case_id']]
            if screened['role'] != entry['archived_role']:
                raise ValueError('TextVQA archived role differs from screening evidence')
            # The immutable v1 image bundle predates TextVQA screening. Overlay
            # separately frozen screening records without editing that manifest.
            case={**case,'predictions':screened['archived_predictions'],
                  'outcomes':{**case['outcomes'],'input-only':screened['role']}}
            lookup[entry['case_id']]=case
        for key in ['image_id','image_sha256','canonical_target']:
            if case[key] != entry[key]:
                raise ValueError('Frozen case identity mismatch: '+key)
        if measurement_target(case)['text'] != entry['measurement_target']:
            raise ValueError('Frozen reference changed')
        ids = [p['token_index'] for p in entry['patches']]
        if len(ids) != 32 or len(set(ids)) != 32 or any(i not in range(576) for i in ids):
            raise ValueError('Expected 32 unique frozen patches')
        counts = Counter(p['stratum'] for p in entry['patches'])
        if counts != {'random':16,'high_I':8,'low_E_high_I':8}:
            raise ValueError('Frozen patch strata differ')
        if any(not math.isfinite(p[k]) for p in entry['patches'] for k in ['archived_E','archived_I']):
            raise ValueError('Non-finite archived measurement')
    return profile, data, lookup


def shortcut_summary(rows):
    trials = [r for r in rows if r.get('kind') == 'trial']
    conditions = {}
    for condition in CONDITIONS:
        items = [r for r in trials if r['condition'] == condition]
        conditions[condition] = {'trials':len(items),'correct':sum(r['scoring']['correct'] for r in items),
            'parsed':sum(r['scoring']['prediction'] is not None for r in items),
            'display_letter_counts':dict(Counter(r['scoring']['prediction'] for r in items if r['scoring']['prediction'] is not None))}
    cases = []
    for cid in dict.fromkeys(r['case_id'] for r in trials):
        case = {'case_id':cid}
        for condition in CONDITIONS:
            items = [r for r in trials if r['case_id']==cid and r['condition']==condition]
            case[condition] = {'trials':len(items),'correct':sum(r['scoring']['correct'] for r in items),
                'selected_original_labels':[r['scoring']['selected_original_label'] for r in items]}
        original = next((r for r in trials if r['case_id']==cid and r['condition']=='real' and r['shift']==0), None)
        repeated = next((r for r in rows if r['case_id']==cid and r.get('kind')=='reference_repeat'), None)
        case['original_repeat_same_response'] = original['response']==repeated['response'] if original and repeated else None
        case['original_repeat_same_semantic_choice'] = original['scoring']['selected_original_label']==repeated['scoring']['selected_original_label'] if original and repeated else None
        cases.append(case)
    return {'conditions':conditions,'cases':cases,
        'interpretation':'Execution/format and content-choice diagnostics on selected cases; question-only success is not proof of a shortcut or general accuracy.'}


def perturbation_summary(rows):
    result = {}
    for stratum in ['random','high_I','low_E_high_I','all_selected']:
        items = [r for r in rows if stratum=='all_selected' or r['stratum']==stratum]
        methods = {}
        for method in METHODS:
            effects = [r['effects'][method]['I'] for r in items]
            base = [r['effects']['zero']['I'] for r in items]
            methods[method] = {'patches':len(items),'median_signed_I':statistics.median(effects) if effects else None,
                'rho_E_I':spearman([r['current_E'] for r in items],effects),
                'rho_vs_fresh_zero':spearman(base,effects),
                'positive_count':sum(i>0 for i in effects),
                'sign_agreement_beyond_1e_minus4':sum(a*b>0 for a,b in zip(base,effects) if min(abs(a),abs(b))>1e-4),
                'sign_comparison_patches':sum(min(abs(a),abs(b))>1e-4 for a,b in zip(base,effects))}
        result[stratum]=methods
    return result
