"""CPU-only policy for native staged ApET versus matched-schedule pure random."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random

from rq1_protocol import REPO, sha256
from rq2_protocol import require, _credit
from rq2_frozen_protocol import PROFILE as PARENT_PROFILE, validate_bundle as parent_bundle

VERSION = "rq2-staged-calibration-v1"
PROFILE = REPO / "research/protocols/rq2_staged_calibration_v1.json"
CASE_IDS = ("gqa:201636988", "gqa:201637316", "mmstar:136",
            "textvqa:37533", "textvqa:35670", "textvqa:35962")
SCHEDULES = ((96, 32), (288, 96))
RANDOM_SEEDS = (101, 202, 303)
LAYER = 16  # AFTER the sixteenth block; native forward handles later cached steps.
REPEAT_IDS = ("gqa:201636988", "textvqa:37533")
PROBE_CASE = "mmstar:136"
TENSOR_IDS = ("mmstar:136", "textvqa:37533")
IDENTITY_CONDITION = "pure-random-576-to-576-s101"
CONDITIONS = {"baseline576": {"method": "baseline", "input_tokens": 576,
    "decoder_tokens": None, "layer": None, "random_seed": None}}
for first, last in SCHEDULES:
    CONDITIONS[f"native-{first}-to-{last}"] = {"method": "native", "input_tokens": first,
        "decoder_tokens": last, "layer": LAYER, "random_seed": None}
    for seed in RANDOM_SEEDS:
        CONDITIONS[f"pure-random-{first}-to-{last}-s{seed}"] = {
            "method": "pure-random", "input_tokens": first, "decoder_tokens": last,
            "layer": LAYER, "random_seed": seed}


def specification(condition):
    if condition == IDENTITY_CONDITION:
        return {"method": "pure-random", "input_tokens": 576, "decoder_tokens": 576,
                "layer": LAYER, "random_seed": 101}
    require(condition in CONDITIONS, "Unknown staged condition")
    return dict(CONDITIONS[condition])


def random_plan(image_sha256, replicate, input_tokens, decoder_tokens):
    require(isinstance(image_sha256, str) and len(image_sha256) == 64
            and all(c in "0123456789abcdef" for c in image_sha256), "Invalid image SHA256")
    require(type(replicate) is int and replicate in RANDOM_SEEDS, "Unknown staged random seed")
    require((input_tokens, decoder_tokens) in (*SCHEDULES, (576, 576)), "Unsupported staged schedule")
    def permutation(size, boundary):
        digest = hashlib.sha256(f"{VERSION}|{image_sha256}|{replicate}|{boundary}".encode()).hexdigest()
        order = list(range(size))
        random.Random(int(digest, 16)).shuffle(order)
        return order, digest
    first, input_seed = permutation(576, "input")
    selected = sorted(first[:input_tokens])
    second, decoder_seed = permutation(input_tokens, f"after-block16-pool{input_tokens}")
    compact = sorted(second[:decoder_tokens])
    return {"version": VERSION, "replicate": replicate,
        "input_original_positions": selected, "decoder_compact_positions": compact,
        "decoder_original_positions": [selected[i] for i in compact],
        "input_seed_sha256": input_seed, "decoder_seed_sha256": decoder_seed,
        "input_permutation_sha256": hashlib.sha256(json.dumps(first).encode()).hexdigest(),
        "decoder_permutation_sha256": hashlib.sha256(json.dumps(second).encode()).hexdigest(),
        "policy": "Uniform without replacement at each stage; independent stage RNG streams; spatial order restored. Layer16 states are contextualized, not isolated original patches."}


def load_profile(path=PROFILE):
    profile = json.loads(Path(path).read_text())
    expected = {"version": VERSION, "parent_profile_sha256": sha256(PARENT_PROFILE),
        "case_ids": list(CASE_IDS), "schedules": [list(v) for v in SCHEDULES],
        "conditions": CONDITIONS, "random_seeds": list(RANDOM_SEEDS), "layer": LAYER,
        "native_basis_tokens": 10, "input_bundle": "rq1-ih-inputs-v1.zip",
        "fixed_repeat_case_ids": list(REPEAT_IDS), "identity_condition": IDENTITY_CONDITION,
        "cache_probe_case_id": PROBE_CASE, "tensor_case_ids": list(TENSOR_IDS),
        "probe_conditions": [f"{method}-{a}-to-{b}" + ("-s101" if method == "pure-random" else "")
                             for a, b in SCHEDULES for method in ("native", "pure-random")],
        "primary_generation": {"do_sample": False, "use_cache": True, "max_new_tokens": 16,
                               "min_new_tokens": 0, "conversation": "vicuna_v1", "batch_size": 1},
        "probe_generation": {"max_new_tokens": 4, "min_new_tokens": 4},
        "policy": "First native realization; input tuples shared per image/condition, layer captures per question. Exact clone replays are controls; fresh native variations are recorded separately. No accuracy pass threshold."}
    require(all(profile.get(k) == v for k, v in expected.items()), "Staged protocol/cohort changed")
    return profile


def validate_bundle(inputs):
    profile = load_profile()
    parent, index, manifests, records = parent_bundle(inputs)
    lookup = {r["case"]["case_id"]: {**r, "rq2": entry}
              for r, entry in zip(records, parent["cases"])}
    require(set(CASE_IDS) <= set(lookup), "Staged cases absent from accepted inputs")
    chosen = [lookup[cid] for cid in CASE_IDS]
    require(profile["cases"] == [r["rq2"] for r in chosen], "Staged question/image/seed identity changed")
    return profile, index, manifests, chosen


def summarize(rows):
    groups = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        cid, condition = row["case_id"], row["condition"]
        require(cid in CASE_IDS and condition in CONDITIONS, "Unknown primary staged row")
        require(cid not in groups[cid.split(":")[0]][condition], "Control repeat is not a primary question")
        _credit(row["scoring"], cid.split(":")[0])
        groups[cid.split(":")[0]][condition][cid] = row
    result = {}
    for dataset, conditions in groups.items():
        ids = set(conditions["baseline576"])
        require(set(conditions) == set(CONDITIONS) and all(set(g) == ids for g in conditions.values()),
                "Staged summaries require complete matched conditions")
        baseline = {cid for cid in ids if conditions["baseline576"][cid]["scoring"]["correct"]}
        summary = {"questions": len(ids), "baseline_correct": len(baseline), "conditions": {}}
        for name, group in conditions.items():
            correct = {cid for cid in ids if group[cid]["scoring"]["correct"]}
            summary["conditions"][name] = {"correct": len(correct), "accuracy": len(correct) / len(ids),
                "mean_score": sum(_credit(row["scoring"], dataset) for row in group.values()) / len(ids),
                "APR": len(correct & baseline) / len(baseline) if baseline else None,
                "losses": sorted(baseline - correct), "gains": sorted(correct - baseline)}
        summary["random_seed_summary"] = {}
        for a, b in SCHEDULES:
            values = [summary["conditions"][f"pure-random-{a}-to-{b}-s{s}"] for s in RANDOM_SEEDS]
            summary["random_seed_summary"][f"{a}-to-{b}"] = {
                metric: {"seed_values": [v[metric] for v in values],
                         "mean": sum(v[metric] for v in values) / 3,
                         "min": min(v[metric] for v in values), "max": max(v[metric] for v in values)}
                for metric in ("accuracy", "mean_score")}
        summary["limits"] = "Selected six-question calibration. Seeds repeat questions. No population, speed or full-paper reproduction claim."
        result[dataset] = summary
    return result
