"""Pure Random-K calibration policy. Importable without Torch or model packages."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from rq1_protocol import REPO, sha256
from rq2_protocol import require, _credit
from rq2_frozen_protocol import PROFILE as PARENT_PROFILE, validate_bundle as parent_bundle

VERSION = "rq2-pure-random-calibration-v1"
PROFILE = REPO / "research/protocols/rq2_pure_random_calibration_v1.json"
BUDGETS = (288, 64)
RANDOM_SEEDS = (101, 202, 303)
CASE_IDS = ("gqa:201636988", "gqa:201637316", "mmstar:1014",
            "textvqa:37533", "textvqa:35670", "textvqa:35962")
REPEAT_IDS = ("gqa:201636988", "textvqa:37533")
FRESH_CASE = "textvqa:37533"
CONDITIONS = {"baseline576": {"method": "baseline", "budget": 576,
                              "merge": False, "random_seed": None}}
for budget in BUDGETS:
    for merge in (True, False):
        CONDITIONS[f"native-{budget}" + ("-merge" if merge else "-no-merge")] = {
            "method": "native", "budget": budget, "merge": merge, "random_seed": None}
    for seed in RANDOM_SEEDS:
        CONDITIONS[f"pure-random-{budget}-s{seed}"] = {
            "method": "pure-random", "budget": budget, "merge": False, "random_seed": seed}
PARITY_CONDITION = "pure-random-576-s101"


def specification(condition):
    if condition == PARITY_CONDITION:
        return {"method": "pure-random", "budget": 576, "merge": False, "random_seed": 101}
    require(condition in CONDITIONS, "Unknown pure-random calibration condition")
    return dict(CONDITIONS[condition])


def load_profile(path=PROFILE):
    p = json.loads(Path(path).read_text())
    require(p.get("version") == VERSION and p.get("parent_profile_sha256") == sha256(PARENT_PROFILE),
            "Pure-random calibration version or parent identity changed")
    require(p.get("case_ids") == list(CASE_IDS) and p.get("budgets") == list(BUDGETS)
            and p.get("random_seeds") == list(RANDOM_SEEDS) and p.get("conditions") == CONDITIONS,
            "Pure-random calibration cohort, budgets or conditions changed")
    require(p.get("input_bundle") == "rq1-ih-inputs-v1.zip"
            and p.get("basis_tokens_native") == 10 and p.get("decoder_layers") == []
            and p.get("parity_condition") == PARITY_CONDITION
            and p.get("fixed_repeat_case_ids") == list(REPEAT_IDS)
            and p.get("fresh_native_case_id") == FRESH_CASE
            and p.get("fresh_native_captures_per_budget") == 2
            and p.get("generation") == {"conversation": "vicuna_v1", "max_new_tokens": 16,
                                       "do_sample": False, "use_cache": True, "batch_size": 1},
            "Pure-random calibration controls changed")
    require(p.get("random_policy") == {
        "pool": "All 576 original spatial positions; no basis slots or E eligibility",
        "sampling": "One isolated Python Random permutation per image and replicate; prefix K, then spatial sort",
        "seed": "SHA256(version|image_sha256|replicate); independent of K, answers and Torch RNG",
        "compression": "Native uncompressed CLIP/projector branch followed by a pure selection mask; no FPS/E/basis/merge",
        "scope": "Input only; native multimodal insertion and compact position policy; no decoder pruning"},
        "Pure-random selection policy changed")
    return p


def random_selection(image_sha256, replicate, budget):
    require(isinstance(image_sha256, str) and len(image_sha256) == 64
            and all(c in "0123456789abcdef" for c in image_sha256), "Invalid image SHA256")
    require(type(replicate) is int and replicate in RANDOM_SEEDS, "Unknown random replicate")
    require(type(budget) is int and budget in (*BUDGETS, 576), "Unsupported random budget")
    digest = hashlib.sha256(f"{VERSION}|{image_sha256}|{replicate}".encode()).hexdigest()
    order = list(range(576))
    random.Random(int(digest, 16)).shuffle(order)
    selected = sorted(order[:budget])
    return selected, {"replicate": replicate, "budget": budget, "derived_seed_sha256": digest,
        "permutation_sha256": hashlib.sha256(json.dumps(order, separators=(",", ":")).encode()).hexdigest(),
        "selected_indices_sorted": selected, "effective_basis_tokens": 0,
        "fps_executed": False, "reconstruction_executed": False, "merge_executed": False}


def validate_bundle(inputs):
    profile = load_profile()
    parent, index, manifests, records = parent_bundle(inputs)
    chosen = [{**r, "rq2": entry} for r, entry in zip(records, parent["cases"])
              if r["case"]["case_id"] in CASE_IDS]
    require([r["case"]["case_id"] for r in chosen] == list(CASE_IDS), "Calibration input cases differ")
    return profile, index, manifests, chosen


def summarize(rows):
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        cid, condition = row["case_id"], row["condition"]
        require(cid in CASE_IDS and condition in CONDITIONS, "Unknown calibration row")
        dest = groups[cid.split(":")[0]][condition]
        require(cid not in dest, "Repeat controls must not be counted as primary questions")
        dest[cid] = row
    result = {}
    for dataset, conditions in groups.items():
        ids = set(conditions["baseline576"])
        require(set(conditions) == set(CONDITIONS)
                and all(set(v) == ids for v in conditions.values()), "Incomplete matched conditions")
        base = {cid for cid in ids if conditions["baseline576"][cid]["scoring"]["correct"]}
        summary = {"unique_questions": len(ids), "baseline_correct": len(base), "conditions": {}}
        for condition, values in conditions.items():
            correct = {cid for cid in ids if values[cid]["scoring"]["correct"]}
            summary["conditions"][condition] = {
                "correct": len(correct), "accuracy": len(correct) / len(ids),
                "mean_score": sum(_credit(v["scoring"], dataset) for v in values.values()) / len(ids),
                "APR": len(correct & base) / len(base) if base else None,
                "losses": sorted(base - correct), "gains": sorted(correct - base)}
        summary["random_seed_summary"] = {}
        for budget in BUDGETS:
            values = [summary["conditions"][f"pure-random-{budget}-s{s}"] for s in RANDOM_SEEDS]
            summary["random_seed_summary"][str(budget)] = {
                metric: {"seed_values": [v[metric] for v in values],
                         "mean": sum(v[metric] for v in values) / len(values),
                         "min": min(v[metric] for v in values), "max": max(v[metric] for v in values)}
                for metric in ("accuracy", "mean_score")}
        summary["limits"] = "Selected execution calibration; seeds repeat questions; no population or speed claim."
        result[dataset] = summary
    return result
