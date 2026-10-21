"""Frozen staged TextVQA100 cohort and paired statistics; CPU-only policy."""
from collections import defaultdict
import json
from pathlib import Path

from rq1_protocol import REPO, SOURCES, sha256
from rq2_protocol import require, _credit
from rq2_staged_protocol import (VERSION as RANDOM_VERSION, PROFILE as CALIBRATION_PROFILE,
    CONDITIONS, SCHEDULES, RANDOM_SEEDS, LAYER, IDENTITY_CONDITION)
from rq2_textvqa_protocol import (REFERENCE, DIAGNOSTIC_IDS, REPEAT_IDS, load_reference,
    validate_inputs as validate_screening_inputs, paired_effect, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)

VERSION = "rq2-staged-textvqa100-v1"
PROFILE = REPO / "research/protocols/rq2_staged_textvqa100_v1.json"
CALIBRATION_REFERENCE = REPO / "research/protocols/rq2_staged_calibration_reference_v1.json"
TENSOR_IDS = ("textvqa:37533", "textvqa:35670")
PROBE_CASE = "textvqa:35670"
FRESH_CASE = "textvqa:37533"
PROBE_CONDITIONS = tuple(f"{method}-{a}-to-{b}" + ("-s101" if method == "pure-random" else "")
                         for a, b in SCHEDULES for method in ("native", "pure-random"))


def load_calibration_reference(path=CALIBRATION_REFERENCE):
    """Reviewed metadata travels with Git; raw ZIP/log remain separately archived."""
    ref = json.loads(Path(path).read_text())
    from run_rq2_staged_calibration import measurement_hashes
    require(ref.get("version") == "rq2-staged-calibration-reference-v1"
            and ref.get("source_archive_sha256") == "8ec5c16997e80cf6051436e9c511f5f77b592441cfede9238ddd428843c3943a"
            and ref.get("calibration_version") == RANDOM_VERSION and ref.get("status") == "passed"
            and ref.get("calibration_passed") is True
            and ref.get("profile_sha256") == sha256(CALIBRATION_PROFILE)
            and ref.get("conditions") == CONDITIONS
            and ref.get("measurement_sources_sha256") == measurement_hashes(),
            "Reviewed staged calibration or calibrated operator sources changed")
    require(ref.get("progress") == dict(completed_questions=6, expected_questions=6,
                completed_primary_predictions=54, expected_primary_predictions=54)
            and ref.get("controls") == dict(identity576=6, fixed_replay=18, forced_cache_probe=4, fresh_raw_native=4)
            and ref.get("prerequisites") == dict(tests_run=26, skipped=0, status="passed"),
            "Reviewed staged calibration is incomplete")
    return ref


def load_profile(path=PROFILE):
    p = json.loads(Path(path).read_text())
    old = load_reference()
    expected = {"version": VERSION, "manifest_sha256": SOURCES["textvqa"][1],
        "reference_sha256": sha256(REFERENCE), "calibration_reference_sha256": sha256(CALIBRATION_REFERENCE),
        "case_ids": [c["case_id"] for c in old["cases"]], "questions": 100, "images": 100,
        "conditions": CONDITIONS, "random_policy_version": RANDOM_VERSION,
        "schedules": [list(s) for s in SCHEDULES], "layer": LAYER, "native_basis_tokens": 10,
        "random_seeds": list(RANDOM_SEEDS), "seed": 590,
        "identity_condition": IDENTITY_CONDITION, "identity_case_ids": list(DIAGNOSTIC_IDS),
        "fixed_repeat_case_ids": list(REPEAT_IDS), "tensor_case_ids": list(TENSOR_IDS),
        "cache_probe_case_id": PROBE_CASE, "probe_conditions": list(PROBE_CONDITIONS),
        "fresh_raw_native_case_id": FRESH_CASE, "fresh_repeats_per_schedule": 2,
        "primary_generation": {"conversation": "vicuna_v1", "do_sample": False,
            "use_cache": True, "batch_size": 1, "max_new_tokens": 16, "min_new_tokens": 0},
        "probe_generation": {"max_new_tokens": 4, "min_new_tokens": 4},
        "bootstrap": {"unit": "question/image", "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED, "interval": "paired percentile 95%; conditional on three random seeds"}}
    require(all(p.get(k) == v for k, v in expected.items()), "Frozen staged TextVQA100 protocol changed")
    require(set((*DIAGNOSTIC_IDS, *REPEAT_IDS, *TENSOR_IDS, PROBE_CASE, FRESH_CASE)) <= set(p["case_ids"]),
            "Staged control case absent")
    load_calibration_reference()
    return p


def validate_inputs(inputs):
    p = load_profile()
    _, data, reference = validate_screening_inputs(inputs)
    require([c["case_id"] for c in data["cases"]] == p["case_ids"],
            "Do not filter/reorder TextVQA100, including baseline-wrong cases")
    return p, data, reference


def historical_comparison(case, arms, reference):
    old = next(c for c in reference["cases"] if c["case_id"] == case["case_id"])
    previous, current = old["historical_predictions"]["baseline"], arms["baseline576"]
    return {"baseline576": {"previous": previous,
        "response_equal": current["response"] == previous["response"],
        "normalized_answer_equal": current["scoring"]["prediction"] == previous["scoring"]["prediction"],
        "consensus_score_delta": current["scoring"]["consensus_score"] - previous["scoring"]["consensus_score"],
        "policy": "Record drift; never filter, retarget or retry based on the answer. Historical input-only288 is not a staged comparator."}}


def _summary(groups, ids, intervals):
    if not ids:
        return {"questions": 0, "conditions": {}, "paired_comparisons": {}}
    base = groups["baseline576"]
    correct_base = {cid for cid in ids if base[cid]["scoring"]["correct"]}
    scores = {name: {cid: _credit(g[cid]["scoring"], "textvqa") for cid in ids} for name, g in groups.items()}
    base_mean = sum(scores["baseline576"].values()) / len(ids)
    out = {"questions": len(ids), "baseline_full_credit": len(correct_base), "conditions": {}}
    for name, g in groups.items():
        correct = {cid for cid in ids if g[cid]["scoring"]["correct"]}
        mean = sum(scores[name].values()) / len(ids)
        out["conditions"][name] = {"mean_consensus_score": mean,
            "mean_score_delta_from_baseline": mean - base_mean,
            "paired_mean_score_ratio": mean / base_mean if base_mean else None,
            "full_credit_count": len(correct), "full_credit_rate": len(correct) / len(ids),
            "baseline_full_credit_preserved": len(correct & correct_base),
            "APR": len(correct & correct_base) / len(correct_base) if correct_base else None,
            "full_credit_losses": sorted(correct_base - correct), "full_credit_gains": sorted(correct - correct_base),
            "positive_credit_count": sum(scores[name][cid] > 0 for cid in ids),
            "empty_response_ids": [cid for cid in ids if not g[cid]["response"].strip()],
            "output_cap_ids": [cid for cid in ids if g[cid].get("generated_tokens") == 16],
            "normalized_answer_changed_ids": [cid for cid in ids
                if g[cid]["scoring"]["prediction"] != base[cid]["scoring"]["prediction"]]}
    comparisons, seed_summary = {}, {}
    pairs = []
    for a, b in SCHEDULES:
        label = f"{a}-to-{b}"
        native = "native-" + label
        average = "pure-random-" + label + "-seed-mean"
        scores[average] = {cid: sum(scores[f"pure-random-{label}-s{s}"][cid] for s in RANDOM_SEEDS) / 3 for cid in ids}
        pairs.append((native, "baseline576"))
        pairs.extend((f"pure-random-{label}-s{s}", native) for s in RANDOM_SEEDS)
        pairs.append((average, native))
        seed_summary[label] = {}
        for metric in ("mean_consensus_score", "full_credit_rate", "APR"):
            v = [out["conditions"][f"pure-random-{label}-s{s}"][metric] for s in RANDOM_SEEDS]
            seed_summary[label][metric] = {"seed_values": dict(zip(map(str, RANDOM_SEEDS), v)),
                "mean": sum(v) / 3 if None not in v else None,
                "min": min(v) if None not in v else None, "max": max(v) if None not in v else None}
    pairs.append(("native-96-to-32", "native-288-to-96"))
    pairs.extend((f"pure-random-96-to-32-s{s}", f"pure-random-288-to-96-s{s}") for s in RANDOM_SEEDS)
    pairs.append(("pure-random-96-to-32-seed-mean", "pure-random-288-to-96-seed-mean"))
    for left, right in pairs:
        value = paired_effect(scores[left], scores[right], ids, intervals=intervals)
        if left in groups and right in groups:
            value["full_credit_rescues_vs_right"] = [cid for cid in ids
                if groups[left][cid]["scoring"]["correct"] and not groups[right][cid]["scoring"]["correct"]]
            value["full_credit_losses_vs_right"] = [cid for cid in ids
                if groups[right][cid]["scoring"]["correct"] and not groups[left][cid]["scoring"]["correct"]]
        comparisons[left + " minus " + right] = value
    out.update(random_seed_summary=seed_summary, paired_comparisons=comparisons,
        limits="Screening/development cohort, not held-out/full TextVQA. Seeds repeat questions; intervals condition on these seeds. Whole-procedure comparison; cannot isolate E, basis or merging. No speed claim.")
    return out


def summarize(rows, *, expected_ids=None, intervals=False):
    allowed = load_profile()["case_ids"]
    groups = defaultdict(dict)
    for row in rows:
        cid, condition = row["case_id"], row["condition"]
        require(cid in allowed and condition in CONDITIONS and not row.get("control_only", False),
                "Unknown staged primary question/condition or included control")
        require(cid not in groups[condition], "Duplicate primary row")
        _credit(row["scoring"], "textvqa")
        groups[condition][cid] = row
    if not rows:
        return {}
    ids = [cid for cid in allowed if cid in groups["baseline576"]]
    require(set(groups) == set(CONDITIONS) and all(set(g) == set(ids) for g in groups.values()),
            "Incomplete matched staged conditions")
    if expected_ids is not None:
        require(ids == list(expected_ids), "Full staged TextVQA100 cohort is incomplete or filtered")
    return {"all100": _summary(groups, ids, intervals),
        "outside_previous_six": _summary(groups, [c for c in ids if c not in DIAGNOSTIC_IDS], intervals),
        "previous_six": _summary(groups, [c for c in ids if c in DIAGNOSTIC_IDS], intervals)}
