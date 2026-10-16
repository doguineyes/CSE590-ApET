"""Project-owned RQ1 input-pilot protocol and CPU-only validation/statistics."""

from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

REPO = Path(__file__).resolve().parents[1]
VERSION = "rq1-input-zero-v2"
# Existing portable inputs remain byte-for-byte valid; only measurement changes.
INPUT_VERSION = "rq1-input-zero-v1"
SOURCES = {
    "gqa": ("rq1-gqa-reviewed-v1-s591/recommended-manifest.json",
            "07a6d8f449d1b3081f34b9568828712178085c16923dc7a9a72832b90feb5111"),
    "mmstar": ("rq1-mmstar-reviewed-v1-s592/recommended-manifest.json",
               "1c01e44702cac2cda7f3f0fd5f10b54de899caeb498af5389fc95d44d302d5b5"),
    "textvqa": ("rq1-candidates-v1-s590/textvqa/screening-manifest.json",
                "d01ad107653ba9d0b215d927d35037464c39f1309756fd27329a82649812ab0f"),
}
# Fixed before new inference: four paired loss images + two preserved pairs.
# Calibration is the first pair, and is nested within the 12-question pilot.
GQA_IDS = ["201636988", "201637316", "201713640", "201713599",
           "201428959", "201429017", "201882905", "201883221",
           "201998268", "201998118", "202218593", "202218829"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def load_manifest(path, expected_dataset=None):
    """Validate immutable accepted manifests and every encoded image before inference."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    dataset = data["dataset"]
    if dataset not in SOURCES or (expected_dataset and dataset != expected_dataset):
        raise ValueError("Unexpected dataset")
    if sha256(path) != SOURCES[dataset][1]:
        raise ValueError(f"Manifest differs from accepted working selection: {path}")
    ids = [case["case_id"] for case in data["cases"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate case IDs")
    checked = set()
    for case in data["cases"]:
        meta = data["images"][case["image_id"]]
        image = (path.parent / meta["file"]).resolve()
        if path.parent not in image.parents:
            raise ValueError("Image escapes manifest directory")
        if case["image_sha256"] != meta["sha256"]:
            raise ValueError("Case and image hashes differ")
        if image not in checked and sha256(image) != meta["sha256"]:
            raise ValueError(f"Image hash mismatch: {image}")
        checked.add(image)
        if not case["canonical_target"] or not case["gold_answers"]:
            raise ValueError("Missing fixed reference target")
        if dataset == "textvqa" and len(case["gold_answers"]) != 10:
            raise ValueError("TextVQA requires all ten annotator answers")
    return data


def pilot_cases(data, limit=12):
    if data["dataset"] == "gqa":
        lookup = {case["question_id"]: case for case in data["cases"]}
        return [lookup[qid] for qid in GQA_IDS[:limit]]
    if data["dataset"] == "mmstar":
        # Later confirmation: stable seed order independent of reconstruction scores.
        return sorted(data["cases"], key=lambda c: hashlib.sha256(
            ("rq1-mmstar-pilot-590:" + c["case_id"]).encode()).hexdigest())[:limit]
    raise ValueError("TextVQA interventions are deferred; establish baseline results first")


def token_order(image_sha256, seed=590):
    # Shared across same-image questions and nested across 32/576-token runs.
    mixed = int(hashlib.sha256(f"{seed}:{image_sha256}".encode()).hexdigest()[:16], 16)
    result = list(range(576))
    random.Random(mixed).shuffle(result)
    return result


def target_audit_candidates(case, generated_response):
    """Diagnostic spelling comparison, not a new target-selection policy.

    The first two forms depend only on the saved gold label. The generated form
    is explicitly post-inference and may not become an experimental gold target.
    """
    raw = case["canonical_target"]
    candidates = [("manifest_literal", raw),
                  ("initial_capital", raw[:1].upper() + raw[1:]),
                  ("generated_diagnostic_only", generated_response.strip())]
    return [{"name": name, "text": text, "post_inference": name == "generated_diagnostic_only"}
            for name, text in candidates if text]


def measurement_target(case):
    """Fixed reference rendering from annotations only, never from model output."""
    raw = case["canonical_target"]
    dataset = case["case_id"].split(":", 1)[0]
    if dataset == "gqa":
        return {"text": raw[:1].upper() + raw[1:], "policy": "gqa-initial-capital-v1"}
    if dataset == "mmstar":
        return {"text": raw, "policy": "mmstar-literal-option-v1"}
    raise ValueError("TextVQA interventions need their own reference policy; generation baseline is unchanged")


def target_alignment(target_ids, generated_ids, baseline_correct):
    """Correct generation must reproduce the fixed target prefix before token runs.

    Wrong-baseline cases still measure the fixed gold answer; they must never be
    retargeted to the generated wrong answer to pass this check.
    """
    if not target_ids or not generated_ids:
        raise ValueError("Missing target or generated tokens for alignment check")
    matches = target_ids == generated_ids[:len(target_ids)]
    return {"baseline_scored_correct": bool(baseline_correct),
            "target_matches_generated_prefix": matches,
            "passed": matches or not baseline_correct,
            "note": "Mismatch is expected for some baseline-wrong cases; the gold target is never replaced"}


def average_ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + end - 1) / 2 + 1
        for index in order[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def spearman(x, y):
    if len(x) != len(y) or not all(math.isfinite(v) for v in [*x, *y]):
        raise ValueError("Correlation needs equal-length finite observations")
    if len(x) < 3:
        return None
    x, y = average_ranks(x), average_ranks(y)
    x = [v - sum(x) / len(x) for v in x]
    y = [v - sum(y) / len(y) for v in y]
    scale = math.sqrt(sum(v*v for v in x) * sum(v*v for v in y))
    return sum(a*b for a, b in zip(x, y)) / scale if scale else None


def textvqa_evaluator():
    # Load the actual upstream scorer without importing llava's model package on Mac.
    source = REPO / "llava/eval/m4c_evaluator.py"
    spec = importlib.util.spec_from_file_location("rq1_native_vqa_scorer", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TextVQAAccuracyEvaluator()


def consensus_score(evaluator, response, answers):
    if len(answers) != 10:
        raise ValueError("Expected ten TextVQA reference answers")
    normalized = evaluator.answer_processor(response)
    score = evaluator._compute_answer_scores(list(answers)).get(normalized, 0.0)
    return {"prediction": normalized, "consensus_score": score,
            "full_credit": score >= 1.0 - 1e-12, "positive_credit": score > 0,
            "scoring_version": "native-textvqa-consensus-v1"}


def textvqa_summary(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[row["configuration"]][row["case_id"]] = row
    result = {}
    baseline = groups.get("baseline", {})
    for name, group in groups.items():
        scores = [r["consensus_score"] for r in group.values()]
        paired = sorted(set(baseline) & set(group))
        full = [key for key in paired if baseline[key]["full_credit"]]
        denominator = sum(baseline[key]["consensus_score"] for key in paired)
        result[name] = {
            "n": len(scores), "mean_consensus_score": sum(scores) / len(scores),
            "full_credit_count": sum(r["full_credit"] for r in group.values()),
            "positive_credit_count": sum(r["consensus_score"] > 0 for r in group.values()),
            "empty_response_count": sum(not r.get("response", "").strip() for r in group.values()),
            "paired_n": len(paired), "baseline_full_credit_denominator": len(full),
            "full_credit_preservation": sum(group[k]["full_credit"] for k in full) / len(full) if full else None,
            "paired_mean_score_ratio": sum(group[k]["consensus_score"] for k in paired) / denominator if denominator else None,
        }
    return result
