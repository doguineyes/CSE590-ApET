"""Frozen TextVQA100 comparison and paired statistics; no Torch imports."""
from collections import defaultdict
import json
from pathlib import Path
import random

from rq1_protocol import REPO, SOURCES, load_manifest, sha256
from rq2_protocol import require, score_response, _credit
from rq2_random_protocol import (VERSION as RANDOM_VERSION, CONDITIONS, BUDGETS,
                                 RANDOM_SEEDS, random_selection)

VERSION = "rq2-textvqa100-random-v1"
PROFILE = REPO / "research/protocols/rq2_textvqa100_random_v1.json"
REFERENCE = REPO / "research/protocols/rq2_textvqa100_reference_v1.json"
DIAGNOSTIC_IDS = ("textvqa:37533", "textvqa:35670", "textvqa:35962",
                  "textvqa:39323", "textvqa:36001", "textvqa:38025")
REPEAT_IDS = ("textvqa:37109", "textvqa:37533")
TENSOR_IDS = ("textvqa:37533", "textvqa:35670")
FRESH_CASE = "textvqa:37533"
BOOTSTRAP_REPLICATES = 2000
BOOTSTRAP_SEED = 59021


def load_reference(path=REFERENCE):
    reference = json.loads(Path(path).read_text())
    require(reference.get("version") == "rq2-textvqa100-reference-v1"
            and reference.get("manifest_sha256") == SOURCES["textvqa"][1],
            "Wrong TextVQA100 historical reference")
    rows = reference["cases"]
    require(len(rows) == len({r["case_id"] for r in rows}) == 100
            and len({r["image_sha256"] for r in rows}) == 100,
            "Expected 100 unchanged screening questions/images")
    require(all(set(r["historical_predictions"]) == {"baseline", "input-only"} for r in rows),
            "Incomplete historical TextVQA predictions")
    return reference


def load_profile(path=PROFILE):
    profile = json.loads(Path(path).read_text())
    reference = load_reference()
    expected = {
        "version": VERSION, "manifest_sha256": SOURCES["textvqa"][1],
        "reference_sha256": sha256(REFERENCE), "questions": 100, "images": 100,
        "case_ids": [r["case_id"] for r in reference["cases"]],
        "random_policy_version": RANDOM_VERSION, "conditions": CONDITIONS,
        "budgets": list(BUDGETS), "random_seeds": list(RANDOM_SEEDS),
        "native_basis_tokens": 10, "decoder_layers": [], "seed": 590,
        "diagnostic_case_ids": list(DIAGNOSTIC_IDS), "fixed_repeat_case_ids": list(REPEAT_IDS),
        "tensor_case_ids": list(TENSOR_IDS), "fresh_native_case_id": FRESH_CASE,
        "fresh_native_captures_per_budget": 2,
        "bootstrap": {"unit": "question/image", "replicates": BOOTSTRAP_REPLICATES,
                      "seed": BOOTSTRAP_SEED, "interval": "paired percentile 95%"},
        "generation": {"conversation": "vicuna_v1", "max_new_tokens": 16,
                       "do_sample": False, "use_cache": True, "batch_size": 1},
    }
    require(all(profile.get(k) == v for k, v in expected.items()),
            "TextVQA100 frozen cohort, methods or controls changed")
    require(set(DIAGNOSTIC_IDS) <= set(profile["case_ids"])
            and set(REPEAT_IDS) <= set(profile["case_ids"]), "Control case missing")
    return profile


def validate_inputs(inputs):
    """Accept the identical TextVQA manifest from either existing portable bundle."""
    profile, reference = load_profile(), load_reference()
    manifest = Path(inputs) / "textvqa/manifest.json"
    data = load_manifest(manifest, "textvqa")
    require([c["case_id"] for c in data["cases"]] == profile["case_ids"],
            "Do not filter or reorder the 100 screening questions")
    from run_rq1_input import case_seed
    for case, old in zip(data["cases"], reference["cases"]):
        require(all(case[k] == old[k] for k in (
            "case_id", "question_id", "image_id", "image_sha256", "question",
            "generation_question", "canonical_target", "gold_answers")),
            "Historical question/image/reference identity changed: " + case["case_id"])
        require(case_seed(case, profile["seed"]) == old["seed"], "Historical image seed changed")
        for prediction in old["historical_predictions"].values():
            require(score_response(case, prediction["response"]) == prediction["scoring"],
                    "Historical scorer/reference changed")
    return profile, data, reference


def historical_comparison(case, arms, reference):
    old = next(r for r in reference["cases"] if r["case_id"] == case["case_id"])
    result = {}
    for previous, current in (("baseline", "baseline576"), ("input-only", "native-288-merge")):
        historic, row = old["historical_predictions"][previous], arms[current]
        result[current] = {"previous": historic,
            "response_equal": row["response"] == historic["response"],
            "normalized_answer_equal": row["scoring"]["prediction"] == historic["scoring"]["prediction"],
            "consensus_score_delta": row["scoring"]["consensus_score"] - historic["scoring"]["consensus_score"],
            "full_credit_equal": row["scoring"]["correct"] == historic["scoring"]["correct"],
            "policy": "Record drift; never filter, retarget or retry a question because of its answer"}
    return result


def _quantile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def paired_effect(left, right, ids, *, intervals):
    """One paired difference per question, including for seed-averaged random."""
    delta = [left[cid] - right[cid] for cid in ids]
    result = {"questions": len(ids), "mean_score_delta": sum(delta) / len(delta),
              "score_increased_ids": [cid for cid, d in zip(ids, delta) if d > 1e-12],
              "score_decreased_ids": [cid for cid, d in zip(ids, delta) if d < -1e-12]}
    if intervals:
        rng = random.Random(BOOTSTRAP_SEED)
        means = [sum(delta[rng.randrange(len(delta))] for _ in delta) / len(delta)
                 for _ in range(BOOTSTRAP_REPLICATES)]
        result["paired_bootstrap_95_ci"] = [_quantile(means, .025), _quantile(means, .975)]
    return result


def _summary(groups, ids, *, intervals):
    if not ids:
        return {"questions": 0, "conditions": {}, "paired_comparisons": {}}
    baseline = groups["baseline576"]
    base_correct = {cid for cid in ids if baseline[cid]["scoring"]["correct"]}
    scores = {name: {cid: _credit(rows[cid]["scoring"], "textvqa") for cid in ids}
              for name, rows in groups.items()}
    baseline_mean = sum(scores["baseline576"].values()) / len(ids)
    out = {"questions": len(ids), "baseline_full_credit": len(base_correct), "conditions": {}}
    for name, rows in groups.items():
        correct = {cid for cid in ids if rows[cid]["scoring"]["correct"]}
        mean = sum(scores[name].values()) / len(ids)
        out["conditions"][name] = {"mean_consensus_score": mean,
            "mean_score_delta_from_baseline": mean - baseline_mean,
            "paired_mean_score_ratio": mean / baseline_mean if baseline_mean else None,
            "full_credit_count": len(correct), "full_credit_rate": len(correct) / len(ids),
            "baseline_full_credit_preserved": len(correct & base_correct),
            "APR": len(correct & base_correct) / len(base_correct) if base_correct else None,
            "full_credit_losses": sorted(base_correct - correct),
            "full_credit_gains": sorted(correct - base_correct),
            "positive_credit_count": sum(scores[name][cid] > 0 for cid in ids),
            "empty_response_ids": [cid for cid in ids if not rows[cid]["response"].strip()],
            "output_cap_ids": [cid for cid in ids if rows[cid].get("generated_tokens") == 16],
            "normalized_answer_changed_ids": [cid for cid in ids
                if rows[cid]["scoring"]["prediction"] != baseline[cid]["scoring"]["prediction"]]}
    comparisons, random_summary = {}, {}
    for budget in BUDGETS:
        native, no_merge = f"native-{budget}-merge", f"native-{budget}-no-merge"
        pairs = [(no_merge, native)]
        for seed in RANDOM_SEEDS:
            arm = f"pure-random-{budget}-s{seed}"
            pairs.extend([(arm, native), (arm, no_merge)])
        average = f"pure-random-{budget}-seed-mean"
        scores[average] = {cid: sum(scores[f"pure-random-{budget}-s{s}"][cid]
                                    for s in RANDOM_SEEDS) / len(RANDOM_SEEDS) for cid in ids}
        pairs.extend([(average, native), (average, no_merge)])
        for left, right in pairs:
            value = paired_effect(scores[left], scores[right], ids, intervals=intervals)
            if left in groups:
                value["full_credit_rescues_vs_right"] = [cid for cid in ids
                    if groups[left][cid]["scoring"]["correct"] and not groups[right][cid]["scoring"]["correct"]]
                value["full_credit_losses_vs_right"] = [cid for cid in ids
                    if groups[right][cid]["scoring"]["correct"] and not groups[left][cid]["scoring"]["correct"]]
            comparisons[left + " minus " + right] = value
        random_summary[str(budget)] = {}
        for metric in ("mean_consensus_score", "full_credit_rate", "APR"):
            values = [out["conditions"][f"pure-random-{budget}-s{s}"][metric] for s in RANDOM_SEEDS]
            random_summary[str(budget)][metric] = {"seed_values": dict(zip(map(str, RANDOM_SEEDS), values)),
                "mean": sum(values) / len(values) if None not in values else None,
                "min": min(values) if None not in values else None,
                "max": max(values) if None not in values else None}
    for left, right in [("native-64-merge", "native-288-merge"),
                         ("native-64-no-merge", "native-288-no-merge")]:
        comparisons[left + " minus " + right] = paired_effect(scores[left], scores[right], ids, intervals=intervals)
    out.update(random_seed_summary=random_summary, paired_comparisons=comparisons,
        limits="Screening/development cohort, not held-out or full TextVQA. Seeds repeat the same questions; intervals condition on these three seeds. No speed claim.")
    return out


def summarize(rows, *, expected_ids=None, intervals=False):
    """Only complete matched question blocks enter a partial/final summary."""
    allowed = load_profile()["case_ids"]
    groups = defaultdict(dict)
    for row in rows:
        cid, condition = row["case_id"], row["condition"]
        require(cid in allowed and condition in CONDITIONS, "Unknown TextVQA100 question/condition")
        require(cid not in groups[condition], "Duplicate primary row; controls are not cohort questions")
        _credit(row["scoring"], "textvqa")
        groups[condition][cid] = row
    if not rows:
        return {}
    ids = [cid for cid in allowed if cid in groups["baseline576"]]
    require(set(groups) == set(CONDITIONS) and all(set(g) == set(ids) for g in groups.values()),
            "Incomplete matched TextVQA100 conditions")
    if expected_ids is not None:
        require(ids == list(expected_ids), "Full TextVQA100 cohort is incomplete or filtered")
    return {"all100": _summary(groups, ids, intervals=intervals),
        "outside_previous_six": _summary(groups, [cid for cid in ids if cid not in DIAGNOSTIC_IDS], intervals=intervals),
        "previous_six": _summary(groups, [cid for cid in ids if cid in DIAGNOSTIC_IDS], intervals=intervals)}
