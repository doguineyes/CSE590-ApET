"""Frozen full MMStar comparison: identities, preparation and paired statistics.

Project-owned evaluation. No native ApET changes, model imports or answer-based
selection. Identical image bytes are clustered because MMStar reuses images.
"""
from collections import Counter, defaultdict
import copy
import hashlib
import io
import json
from pathlib import Path

from rq1_protocol import REPO, sha256, write_json
from rq2_protocol import require
from rq2_staged_protocol import CONDITIONS, SCHEDULES, RANDOM_SEEDS
from rq2_staged_textvqa_protocol import load_calibration_reference, CALIBRATION_REFERENCE
from eval_mmstar import DATASET_ID, DATASET_REVISION, prepare_question
from eval_original_mmstar import PARQUET_SHA256
from mmstar_scoring import score_answer

VERSION = "rq2-staged-mmstar1500-v1"
PROFILE = REPO / "research/protocols/rq2_staged_mmstar1500_v1.json"
REFERENCE = REPO / "research/protocols/rq2_staged_mmstar1500_reference_v1.json"
REFERENCE_SHA256 = "917486e1f1a70c39a352d64c35b82a650dd81e916d87dd589ff406aaf3107e8d"
REPEAT_IDS = TENSOR_IDS = ("mmstar:136", "mmstar:617")
PROBE_CASE = FRESH_CASE = "mmstar:136"
BOOTSTRAP_SEED, BOOTSTRAP_REPLICATES = 59023, 2000


def load_reference():
    require(sha256(REFERENCE) == REFERENCE_SHA256, "Frozen full MMStar reference changed")
    return json.loads(REFERENCE.read_text())


def load_profile(path=PROFILE):
    p = json.loads(Path(path).read_text())
    ref = load_reference()
    from rq2_staged_protocol import VERSION as RANDOM_VERSION, IDENTITY_CONDITION
    expected = dict(version=VERSION, reference_sha256=REFERENCE_SHA256,
        calibration_reference_sha256=sha256(CALIBRATION_REFERENCE), questions=1500, images=1430,
        conditions=CONDITIONS, schedules=[list(s) for s in SCHEDULES], layer=16,
        native_basis_tokens=10, random_seeds=list(RANDOM_SEEDS), random_policy_version=RANDOM_VERSION,
        seed=590, identity_condition=IDENTITY_CONDITION, fixed_repeat_case_ids=list(REPEAT_IDS),
        tensor_case_ids=list(TENSOR_IDS), cache_probe_case_id=PROBE_CASE,
        fresh_raw_native_case_id=FRESH_CASE, fresh_repeats_per_schedule=2,
        primary_generation=dict(conversation="vicuna_v1", do_sample=False, use_cache=True,
            batch_size=1, max_new_tokens=16, min_new_tokens=0),
        bootstrap=dict(unit="image-byte-hash cluster; question-weighted ratio of cluster sums/counts",
            replicates=BOOTSTRAP_REPLICATES, seed=BOOTSTRAP_SEED,
            interval="paired percentile 95%; conditional on three fixed random seeds"))
    require(all(p.get(k) == v for k, v in expected.items()), "Frozen staged MMStar protocol changed")
    require(p["case_ids"] == [c["case_id"] for c in ref["cases"]]
        and p["case_image_ids"] == {c["case_id"]: c["image_sha256"] for c in ref["cases"]},
        "MMStar identities changed")
    # Recompute the image-group assignment without reading historical answers.
    groups = defaultdict(list)
    for c in ref["cases"]:
        groups[c["image_sha256"]].append(c)
    parts, counts, totals = [[] for _ in range(6)], [Counter() for _ in range(6)], [0] * 6
    for _, cases in sorted(groups.items(), key=lambda kv: min(c["index"] for c in kv[1])):
        cats = Counter(c["category"] for c in cases)
        i = min(range(6), key=lambda j: (sum(counts[j][k] * n for k, n in cats.items()), totals[j], j))
        parts[i].extend(c["case_id"] for c in cases)
        counts[i].update(cats)
        totals[i] += len(cases)
    rank = {c["case_id"]: c["index"] for c in ref["cases"]}
    for part in parts:
        part.sort(key=rank.get)
    require(p["shards"] == {str(i): part for i, part in enumerate(parts)}
        and totals == [250] * 6 and p["identity_case_ids"] == [part[0] for part in parts],
        "Fixed full-image MMStar parts or controls changed")
    load_calibration_reference()
    return p


def selected_ids(profile, shards):
    require(shards and all(type(i) is int and 0 <= i < 6 for i in shards)
        and shards == sorted(set(shards)), "Use unique increasing MMStar shards 0..5")
    accepted = {c for i in shards for c in profile["shards"][str(i)]}
    return [c for c in profile["case_ids"] if c in accepted]


def manifest_data(reference=None):
    ref = load_reference() if reference is None else reference
    cases = [{k: copy.deepcopy(v) for k, v in c.items() if k != "historical_baseline"} for c in ref["cases"]]
    return dict(version=VERSION + "-inputs", dataset="mmstar", reference_sha256=REFERENCE_SHA256,
        source=ref["dataset"], cases=cases, images={c["image_id"]: dict(file="images/" + c["image_sha256"] + ".img",
            sha256=c["image_sha256"], size=c["image_size"], rgb_pixels_sha256=c["rgb_pixels_sha256"]) for c in cases})


def validate_inputs(path):
    from PIL import Image
    path = Path(path).resolve()
    p, ref = load_profile(), load_reference()
    data = json.loads(path.read_text())
    require(data == manifest_data(ref), "Require unchanged full1500 MMStar inputs; filtered/edited sets refused")
    for meta in data["images"].values():
        image = path.parent / meta["file"]
        require(sha256(image) == meta["sha256"], "MMStar source image changed")
        with Image.open(image) as raw:
            rgb = raw.convert("RGB")
            require(list(rgb.size) == meta["size"]
                and hashlib.sha256(rgb.tobytes()).hexdigest() == meta["rgb_pixels_sha256"],
                "Decoded MMStar RGB/geometry changed")
    return p, data, ref


def prepare_rows(rows, output, reference=None):
    """Decode the pinned parquet's rows and verify every identity before inference."""
    from PIL import Image
    ref = load_reference() if reference is None else reference
    require(len(rows) == len(ref["cases"]) and len({int(r["index"]) for r in rows}) == len(rows),
        "Missing/duplicate MMStar parquet rows")
    output = Path(output)
    require(not output.exists(), "Preserve existing prepared inputs; choose a new directory")
    data = manifest_data(ref)
    (output / "images").mkdir(parents=True)
    for number, (raw, c) in enumerate(zip(rows, ref["cases"])):
        require(int(raw["index"]) == c["index"] and number == c["row"]
            and raw["question"] == c["question"] and prepare_question(raw["question"]) == c["generation_question"]
            and raw["answer"].strip().upper() == c["canonical_target"]
            and raw["category"] == c["category"] and raw["l2_category"] == c["l2_category"],
            "Pinned MMStar question/label/order/category changed")
        blob = raw["image"].get("bytes") if isinstance(raw["image"], dict) else raw["image"]
        require(isinstance(blob, bytes) and hashlib.sha256(blob).hexdigest() == c["image_sha256"],
            "Pinned MMStar embedded image changed")
        with Image.open(io.BytesIO(blob)) as image:
            rgb = image.convert("RGB")
            require(list(rgb.size) == c["image_size"] and hashlib.sha256(rgb.tobytes()).hexdigest() == c["rgb_pixels_sha256"],
                "MMStar decoded pixels changed")
        destination = output / data["images"][c["image_id"]]["file"]
        if not destination.exists():
            destination.write_bytes(blob)
    write_json(output / "manifest.json", data)
    return output / "manifest.json"


def historical_comparison(case, arms, reference):
    previous = reference["cases"][case["index"]]["historical_baseline"]
    now = arms["baseline576"]
    return dict(previous=previous, response_equal=now["response"] == previous["response"],
        prediction_equal=now["scoring"]["prediction"] == previous["scoring"]["prediction"],
        correct_equal=now["scoring"]["correct"] == previous["scoring"]["correct"],
        policy="Record baseline drift; no answer-based filtering, replacement or retry")


def paired_difference(left, right, ids, image_for, *, intervals):
    groups = defaultdict(list)
    for c in ids:
        groups[image_for[c]].append(left[c] - right[c])
    sums = [sum(v) for _, v in sorted(groups.items())]
    counts = [len(v) for _, v in sorted(groups.items())]
    result = dict(questions=len(ids), images=len(groups), mean_accuracy_delta=sum(sums) / len(ids))
    if intervals:
        import numpy as np
        rng = np.random.default_rng(BOOTSTRAP_SEED)
        draws = rng.integers(0, len(groups), size=(BOOTSTRAP_REPLICATES, len(groups)))
        effects = np.asarray(sums)[draws].sum(axis=1) / np.asarray(counts)[draws].sum(axis=1)
        result["paired_image_bootstrap_95_ci"] = np.quantile(effects, [.025, .975]).tolist()
    return result


def condition_summary(groups, ids):
    base = {c for c in ids if groups["baseline576"][c]["scoring"]["correct"]}
    result = {}
    for condition, g in groups.items():
        good = {c for c in ids if g[c]["scoring"]["correct"]}
        result[condition] = dict(correct=len(good), accuracy=len(good) / len(ids),
            strict_correct=sum(g[c]["scoring"]["strict_correct"] for c in ids),
            strict_accuracy=sum(g[c]["scoring"]["strict_correct"] for c in ids) / len(ids),
            baseline_correct_preserved=len(good & base), APR=len(good & base) / len(base) if base else None,
            losses=sorted(base - good), gains=sorted(good - base),
            unparsed_ids=[c for c in ids if g[c]["scoring"]["prediction"] is None],
            format_noncompliant_ids=[c for c in ids if not g[c]["scoring"]["format_compliant"]],
            empty_response_ids=[c for c in ids if not g[c]["response"].strip()],
            output_cap_ids=[c for c in ids if g[c].get("generated_tokens") == 16])
    return result


def summarize(rows, *, expected_ids=None, intervals=False):
    p, ref = load_profile(), load_reference()
    cases = {c["case_id"]: c for c in ref["cases"]}
    groups = defaultdict(dict)
    for r in rows:
        cid, name = r["case_id"], r["condition"]
        require(cid in cases and name in CONDITIONS and not r.get("control_only", False)
            and cid not in groups[name], "Unknown/duplicate/control MMStar primary")
        require(r["scoring"] == score_answer(r["response"], cases[cid]["question"], cases[cid]["canonical_target"]),
            "MMStar saved score disagrees with frozen parser")
        groups[name][cid] = r
    if not rows:
        return {}
    ids = [c for c in p["case_ids"] if c in groups["baseline576"]]
    require(set(groups) == set(CONDITIONS) and all(set(g) == set(ids) for g in groups.values()),
        "Incomplete matched MMStar arms")
    if expected_ids is not None:
        require(ids == list(expected_ids), "Missing/filtered requested MMStar questions")
    summary = condition_summary(groups, ids)
    scores = {name: {c: float(g[c]["scoring"]["correct"]) for c in ids} for name, g in groups.items()}
    pairs, seed_summary = [], {}
    for a, b in SCHEDULES:
        label = f"{a}-to-{b}"
        native, average = "native-" + label, "pure-random-" + label + "-seed-mean"
        scores[average] = {c: sum(scores[f"pure-random-{label}-s{s}"][c] for s in RANDOM_SEEDS) / 3 for c in ids}
        pairs += [(native, "baseline576"), (average, native)]
        pairs += [(f"pure-random-{label}-s{s}", native) for s in RANDOM_SEEDS]
        seed_summary[label] = {}
        for metric in ("accuracy", "APR"):
            v = [summary[f"pure-random-{label}-s{s}"][metric] for s in RANDOM_SEEDS]
            seed_summary[label][metric] = dict(seed_values=dict(zip(map(str, RANDOM_SEEDS), v)),
                mean=sum(v) / 3 if None not in v else None, min=min(v) if None not in v else None,
                max=max(v) if None not in v else None)
    pairs += [("native-96-to-32", "native-288-to-96")]
    pairs += [(f"pure-random-96-to-32-s{s}", f"pure-random-288-to-96-s{s}") for s in RANDOM_SEEDS]
    pairs += [("pure-random-96-to-32-seed-mean", "pure-random-288-to-96-seed-mean")]
    comparisons = {}
    for left, right in pairs:
        v = paired_difference(scores[left], scores[right], ids, p["case_image_ids"], intervals=intervals)
        if left in groups and right in groups:
            v.update(rescues=[c for c in ids if scores[left][c] > scores[right][c]],
                losses=[c for c in ids if scores[left][c] < scores[right][c]])
        comparisons[left + " minus " + right] = v
    categories = {}
    for field in ("category", "l2_category"):
        labels = sorted({cases[c][field] for c in ids})
        categories[field] = {k: dict(questions=sum(cases[c][field] == k for c in ids),
            conditions=condition_summary(groups, [c for c in ids if cases[c][field] == k])) for k in labels}
    return dict(questions=len(ids), images=len({p["case_image_ids"][c] for c in ids}),
        full_mmstar1500=len(ids) == 1500, baseline_correct=summary["baseline576"]["correct"],
        conditions=summary, random_seed_summary=seed_summary, paired_comparisons=comparisons,
        category_summaries=categories, limits="Full pinned MMStar with project explicit-choice-v2 scoring, not official leaderboard reproduction. Reused image bytes clustered; random seeds are not extra questions. Whole-procedure comparison; no E-only or latency claim.")
