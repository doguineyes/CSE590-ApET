"""Frozen RQ2 input merge/no-merge pilot; CPU-only identity and scoring helpers.

This module owns evaluation and protocol checks, not native ApET compression.
Importing it never imports Torch or changes an experiment's random state.
"""

from collections import Counter, defaultdict
import copy
import json
import math
from pathlib import Path

from rq1_input_ih import SOURCE_FILES as RQ1_REFERENCE_SOURCES
from rq1_input_ih import validate_bundle as validate_rq1_bundle
from rq1_protocol import REPO, consensus_score, sha256, textvqa_evaluator

VERSION = "rq2-input-merge-v1"
PROFILE = REPO / "research/protocols/rq2_input_merge_v1.json"
ARMS = {
    "baseline576": {"input_tokens": 576, "basis_tokens": 10,
                    "decoder_layers": [], "merge_enabled": False},
    "native-input288": {"input_tokens": 288, "basis_tokens": 10,
                        "decoder_layers": [], "merge_enabled": True},
    "native-input288-no-merge": {"input_tokens": 288, "basis_tokens": 10,
                                 "decoder_layers": [], "merge_enabled": False},
}
IDENTITY_FIELDS = ("case_id", "question_id", "image_id", "image_sha256",
                   "question", "generation_question", "canonical_target", "gold_answers")
SOURCE_FILES = (
    "scripts/rq2_protocol.py", "research/protocols/rq2_input_merge_v1.json",
    "scripts/rq1_input_ih.py", "scripts/rq1_protocol.py", "scripts/rq1_native.py",
    "scripts/run_rq1_input.py", "scripts/smoke_original_apet.py",
    "scripts/check_original_env.py", "scripts/original_diagnostics.py",
    "scripts/gqa_scoring.py", "scripts/mmstar_scoring.py", "scripts/eval_mmstar.py",
    "llava/eval/m4c_evaluator.py", "environments/original/uv.lock",
    "research/provenance/original_llava_reference.json",
    "research/protocols/rq1_input_ih_v1.json",
    "research/protocols/rq1_score_reanalysis_v1.json",
)


def require(value, message):
    if not value:
        raise ValueError(message)


def historical_predictions(record):
    """Keep old generated strings/scores as references, never new variant targets."""
    fields = {"response", "prediction", "correct", "consensus_score", "full_credit",
              "positive_credit", "scoring_version", "seed"}
    return {arm: {k: v for k, v in record["reference"]["archived_predictions"][arm].items()
                  if k in fields} for arm in ("baseline", "input-only")}


def load_profile(path=PROFILE):
    profile = json.loads(Path(path).read_text())
    require(profile.get("version") == VERSION, "Wrong RQ2 profile version")
    upstream_path = REPO / profile["source_input_ih_profile"]
    upstream = json.loads(upstream_path.read_text())
    flags_path = REPO / profile["flags_profile"]
    flags = json.loads(flags_path.read_text())
    for path_key, hash_key in (("source_input_ih_profile", "source_input_ih_profile_sha256"),
                               ("flags_profile", "flags_profile_sha256")):
        require(sha256(REPO / profile[path_key]) == profile[hash_key],
                "Pinned RQ2 source profile differs: " + path_key)
    require(profile["reference_index_sha256"] == upstream["reference_index_sha256"],
            "RQ2 reference index differs from the completed RQ1 panel")
    require(profile["native_reference_sha256"] == sha256(REPO / "research/provenance/original_llava_reference.json")
            and profile["lock_sha256"] == sha256(REPO / "environments/original/uv.lock"),
            "RQ2 native provenance or original runtime lock differs")
    require(profile.get("input_bundle") == "rq1-ih-inputs-v1.zip", "Wrong RQ2 input bundle")
    require(list(profile["arms"]) == list(ARMS), "RQ2 arms were added, removed or reordered")
    for arm, expected in ARMS.items():
        require(all(profile["arms"][arm].get(k) == v for k, v in expected.items()),
                "RQ2 compression settings differ: " + arm)
        require(bool(profile["arms"][arm].get("owner")), "Missing native/variant ownership")
    settings = profile["generation"]
    require(all(settings.get(k) == v for k, v in {
        "conversation": "vicuna_v1", "do_sample": False, "use_cache": True,
        "max_new_tokens": 16, "min_new_tokens": 0, "batch_size": 1}.items()),
        "RQ2 generation policy changed")
    entries = profile["cases"]
    require(profile["case_ids"] == upstream["case_ids"] == [c["case_id"] for c in entries]
            and len(entries) == len({c["case_id"] for c in entries}) == 30,
            "RQ2 must preserve the ordered 30-question RQ1 panel")
    require(Counter(c["case_id"].split(":")[0] for c in entries) ==
            {"gqa": 12, "mmstar": 12, "textvqa": 6}
            and len({c["image_sha256"] for c in entries}) == 24,
            "RQ2 dataset/image counts changed")
    image_seeds = {}
    for entry in entries:
        cid = entry["case_id"]
        require(all(entry.get(k) for k in IDENTITY_FIELDS), "Incomplete frozen RQ2 identity")
        require(type(entry["seed"]) is int and 0 <= entry["seed"] < 2**31,
                "Invalid archived image seed")
        old_seed = image_seeds.setdefault(entry["image_sha256"], entry["seed"])
        require(entry["seed"] == old_seed, "Same-image RQ2 questions have different seeds")
        require(entry["flags"] == {
            "mmstar_grounding": cid in flags["mmstar_grounding_flags"],
            "mmstar_prior_quality": cid in flags["mmstar_prior_quality_flags"]},
            "RQ2 must retain previous MMStar flags without filtering")
        for prediction in entry["historical_predictions"].values():
            actual = score_response(entry, prediction["response"])
            require(actual["correct"] == prediction["correct"]
                    and actual["prediction"] == prediction["prediction"]
                    and actual["scoring_version"] == prediction["scoring_version"],
                    "Frozen historical response score differs")
            if cid.startswith("textvqa:"):
                require(actual["consensus_score"] == prediction["consensus_score"],
                        "Frozen historical TextVQA consensus differs")
    require(set(profile["controls"]["reference_repeats"]) <= set(profile["case_ids"]),
            "Unknown reference repeat case")
    return profile


def validate_bundle(inputs):
    """Validate all old evidence, then require exact identities for the new pilot."""
    profile = load_profile()
    _, index, manifests, records = validate_rq1_bundle(inputs)
    require([r["case"]["case_id"] for r in records] == profile["case_ids"],
            "RQ2 reference cohort differs")
    for entry, record in zip(profile["cases"], records):
        require(all(record["case"][k] == entry[k] for k in IDENTITY_FIELDS),
                "RQ2 frozen image/question/target identity differs: " + entry["case_id"])
        require(record["reference"]["seed"] == entry["seed"]
                and historical_predictions(record) == entry["historical_predictions"],
                "RQ2 archived seed or generated references differ: " + entry["case_id"])
    return profile, index, manifests, records


def case_panel(records, limit=None):
    """Attach frozen flags/references; optional prefixes are preflight runs only."""
    profile = load_profile()
    require([r["case"]["case_id"] for r in records] == profile["case_ids"],
            "Missing, reordered or filtered RQ2 panel")
    if limit is None:
        limit = 30
    require(type(limit) is int and 1 <= limit <= 30, "RQ2 prefix size must be 1..30")
    return [{**record, "rq2": copy.deepcopy(entry)}
            for record, entry in zip(records[:limit], profile["cases"][:limit])]


def score_response(case, response):
    """Use existing dataset scoring unchanged; TextVQA correct means full credit."""
    dataset = case["case_id"].split(":", 1)[0]
    if dataset == "gqa":
        from gqa_scoring import score_answer
        return score_answer(response, case["canonical_target"])
    if dataset == "mmstar":
        from mmstar_scoring import score_answer
        return score_answer(response, case["question"], case["canonical_target"])
    if dataset == "textvqa":
        scored = consensus_score(textvqa_evaluator(), response, case["gold_answers"])
        return {**scored, "correct": scored["full_credit"],
                "correct_definition": "Full ten-annotator consensus credit; partial credit retained separately"}
    raise ValueError("Unsupported RQ2 dataset")


def source_hashes(extra=()):
    native = json.loads((REPO / "research/provenance/original_llava_reference.json").read_text())
    paths = (set(SOURCE_FILES) | set(RQ1_REFERENCE_SOURCES)
             | set(native["native_source_sha256"]) | set(extra))
    for name in paths:
        path = (REPO / name).resolve()
        require(REPO.resolve() in path.parents and path.is_file(),
                "Missing/unsafe RQ2 evidence source: " + name)
    return {name: sha256(REPO / name) for name in sorted(paths)}


def _credit(scoring, dataset):
    require(type(scoring.get("correct")) is bool, "Missing explicit correctness")
    if dataset != "textvqa":
        return float(scoring["correct"])
    score = scoring.get("consensus_score")
    require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1,
            "Invalid TextVQA consensus score")
    require(scoring.get("full_credit") == scoring["correct"] == (score >= 1 - 1e-12),
            "TextVQA full-credit correctness differs from consensus score")
    return score


def paired_summary(rows):
    """Per-dataset descriptive paired metrics, with strict matched arm coverage.

    Rows have ``case_id``, ``arm`` and ``scoring``. Repeats belong in a separate
    control table and cannot inflate this diagnostic panel's denominator.
    """
    groups = defaultdict(lambda: defaultdict(dict))
    allowed = set(load_profile()["case_ids"])
    for row in rows:
        cid, arm = row["case_id"], row["arm"]
        require(cid in allowed and arm in ARMS, "Unknown RQ2 case or arm")
        dataset = cid.split(":")[0]
        require(cid not in groups[dataset][arm], "Duplicate RQ2 case/arm; repeats are controls")
        _credit(row["scoring"], dataset)
        groups[dataset][arm][cid] = row
    require(bool(groups), "No RQ2 generated answers")
    result = {}
    for dataset, arms in groups.items():
        require(set(arms) == set(ARMS), "All three RQ2 arms are required")
        baseline = arms["baseline576"]
        require(all(set(group) == set(baseline) for group in arms.values()),
                "RQ2 arms do not share exactly the same questions")
        count = len(baseline)
        correct_ids = {cid for cid, row in baseline.items() if row["scoring"]["correct"]}
        baseline_mean = sum(_credit(row["scoring"], dataset) for row in baseline.values()) / count
        out = {"questions": count, "baseline_correct": len(correct_ids), "arms": {}}
        for arm, group in arms.items():
            correct = {cid for cid, row in group.items() if row["scoring"]["correct"]}
            mean = sum(_credit(row["scoring"], dataset) for row in group.values()) / count
            out["arms"][arm] = {
                "correct": len(correct), "accuracy": len(correct) / count,
                "baseline_correct_preserved": len(correct & correct_ids),
                "APR": len(correct & correct_ids) / len(correct_ids) if correct_ids else None,
                "gains": sorted(correct - correct_ids), "losses": sorted(correct_ids - correct),
                "mean_score": mean, "mean_score_delta_from_baseline": mean - baseline_mean,
            }
            if dataset == "mmstar":
                out["arms"][arm]["unparsed_ids"] = sorted(
                    cid for cid, row in group.items() if row["scoring"].get("prediction") is None)
                out["arms"][arm]["format_noncompliant_ids"] = sorted(
                    cid for cid, row in group.items() if not row["scoring"].get("format_compliant", False))
        native, variant = arms["native-input288"], arms["native-input288-no-merge"]
        out["no_merge_vs_native"] = {
            "rescues": sorted(cid for cid in baseline if not native[cid]["scoring"]["correct"]
                              and variant[cid]["scoring"]["correct"]),
            "new_losses": sorted(cid for cid in baseline if native[cid]["scoring"]["correct"]
                                 and not variant[cid]["scoring"]["correct"]),
            "mean_score_delta": out["arms"]["native-input288-no-merge"]["mean_score"]
                                - out["arms"]["native-input288"]["mean_score"],
        }
        result[dataset] = out
    return result
