#!/usr/bin/env python3
"""Small, baseline-competent MMStar extension with separate qualitative cases."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import re
import shutil
import zipfile

from mmstar_scoring import normalize_text, option_texts
from prepare_gqa_rq1_round2 import render_decisions
from prepare_rq1_candidates import MMSTAR_ARCHIVE, dump
from rq1_candidates import Archive, CONFIGURATIONS, REPO, digest, outcome, rank, save_image

POLICY = {
    "id": "mmstar-rq1-round2-quality-v1", "seed": 592,
    "new_candidate_limit": 12, "reserve_review_limit": 6,
    "main_eligibility": "Human/assistant Keep and baseline correct; input-only loss or preserved-correct",
    "context_eligibility": "Original human Keep gains/both-wrong retained separately; no additions deliberately sampled here",
    "new_image_rule": "Exclude every first-round image byte hash; one added question per new image",
    "new_option_rule": "Four distinct, nonempty, non-placeholder choices; original prompts/choices never rewritten",
    "quality_rules": {
        "speculative_or_subjective": r"what will happen|likely purpose|possible reason|season|weather|humid|profession",
        "identity_or_demographic": r"who is the person|ethnicity|caucasian|elon musk|xiang liu|jay chou|ming yao",
        "expert_knowledge": r"calcification|cfa\b|phasor|enantiomer|chirality|pith|protoxylem|metaxylem|pharmacokinetic|operating point|hydrophobic|price.to.earnings|cholesterol",
    },
    "tie_break": "Prefer underrepresented gold option in the growing main set, then seeded SHA256; within declared task families",
    "slots": [
        {"category": "math", "role": "loss", "family": "geometry loss", "match": r"find.*angle"},
        {"category": "science & technology", "role": "loss", "family": "labeled structure loss", "match": r"what is [a-z] in the diagram"},
        {"category": "math", "role": "preserved_correct", "family": "chart/table reading", "l2": "statistical reasoning", "match": r"how many|value|difference|range|minimum age"},
        {"category": "math", "role": "preserved_correct", "family": "measurement/arithmetic", "l2": "numeric commonsense and calculation", "match": r"fraction|ruler|length|bars.*values"},
        {"category": "science & technology", "role": "preserved_correct", "family": "map reading", "match": r"ocean.*highlighted|area on the map.*china"},
        {"category": "science & technology", "role": "preserved_correct", "family": "plant/life-cycle labels", "match": r"label.*seedling|letter.*plant flowers|cycle.*letter|area.*bud"},
        {"category": "logical reasoning", "role": "preserved_correct", "family": "diagram reading", "l2": "diagram reasoning", "match": r"bar|higher value|direction"},
        {"category": "logical reasoning", "role": "preserved_correct", "family": "visible object/activity", "l2": "common reasoning", "match": r"position|piece of clothing|activity|vegetation|predominant color|closest to the bottom"},
        {"category": "coarse perception", "role": "preserved_correct", "family": "scene content", "l2": "image scene and topic"},
        {"category": "coarse perception", "role": "preserved_correct", "family": "image appearance", "l2": "image style & quality"},
        {"category": "fine-grained perception", "role": "preserved_correct", "family": "object recognition", "l2": "recognition"},
        {"category": "instance reasoning", "role": "preserved_correct", "family": "spatial relation", "match": r"where.*lamp|object relationship|direction|position"},
    ],
    "review_rubric": ["Visible evidence supports the original option rather than simply trusting baseline correctness",
                      "Disjoint choices, clear referents and legible labels; independently solve simple arithmetic/geometry",
                      "Reject unclear references, unsupported labels and tasks needing specialist interpretation",
                      "Retain all rejected candidates and record replacement origins"],
    "limitations": ["Assistant review is not independent human validation and is not outcome-blinded",
                    "Baseline-correct selection is conditional and can include guesses or option-position/language priors",
                    "This is a small cross-task stress subset, not an MMStar-wide accuracy estimate"],
}


def load_human(first_path, review_path):
    first, review = json.loads(first_path.read_text()), json.loads(review_path.read_text())
    if (review.get("schema") != "rq1-review-overlay-v1" or review.get("dataset") != "mmstar"
            or review.get("manifest_sha256") != digest(first_path)
            or set(review["decisions"]) != {c["case_id"] for c in first["cases"]}):
        raise ValueError("MMStar review identity differs from the exact first-round manifest")
    for d in review["decisions"].values():
        if d["status"] not in {"keep", "exclude", "ambiguous", "uncertain", "annotation_issue"} or not isinstance(d["notes"], str):
            raise ValueError("Incomplete human review")
    return first, review


def flags(records):
    b = records["baseline"]
    question = b["original_question"]
    result = [name for name, pattern in POLICY["quality_rules"].items() if re.search(pattern, question, re.I)]
    choices = option_texts(question)
    real = [normalize_text(s) for s in choices.values() if normalize_text(s) not in {"", "nan", "none", "n/a"}]
    if len(real) != 4 or len(set(real)) != 4:
        result.append("four_distinct_real_options_required")
    if any(r["generated_tokens"] >= 16 for r in records.values()):
        result.append("output_token_cap")
    return result


def select_additions(rows, first, human, seed=592):
    old_images = set(first["images"])
    core = [c for c in first["cases"] if human["decisions"][c["case_id"]]["status"] == "keep"
            and c["predictions"]["baseline"]["correct"]]
    gold_counts = Counter(c["canonical_target"] for c in core)
    selected, selected_images, shortfalls = [], set(), []
    for slot in POLICY["slots"]:
        pool = []
        for qid, records in rows.items():
            b = records["baseline"]
            if (b["image_sha256"] in old_images | selected_images or not b["correct"] or flags(records)
                    or b["category"] != slot["category"]
                    or outcome(b["correct"], records["input-only"]["correct"]) != slot["role"]):
                continue
            if slot.get("l2") and b["l2_category"] != slot["l2"]:
                continue
            if slot.get("match") and not re.search(slot["match"], b["original_question"], re.I):
                continue
            pool.append(qid)
        if not pool:
            shortfalls.append(slot)
            continue
        qid = min(pool, key=lambda q: (gold_counts[rows[q]["baseline"]["gold"]], rank(seed, slot["family"], q)))
        selected.append((qid, slot))
        b = rows[qid]["baseline"]
        selected_images.add(b["image_sha256"])
        gold_counts[b["gold"]] += 1
    return selected, shortfalls


def summarize(cases):
    return {"questions": len(cases), "images": len({c["image_id"] for c in cases}),
        "categories": dict(Counter(c["category"] for c in cases)),
        "input_outcomes": dict(Counter(c["outcomes"]["input-only"] for c in cases)),
        "gold_options": dict(Counter(c["canonical_target"] for c in cases))}


def validate_review(path, archive, first_path, first, picks):
    review = json.loads(path.read_text())
    initial = {qid for qid, _ in picks}
    if (review.get("schema") != "rq1-assistant-review-v1" or review.get("dataset") != "mmstar"
            or review.get("human_validated") is not False or review.get("outcome_blinded") is not False
            or review.get("first_manifest_sha256") != digest(first_path)
            or review.get("predictions_sha256") != archive.predictions_sha256
            or set(review["initial_candidate_question_ids"]) != initial):
        raise ValueError("Assistant review is not bound to these MMStar sources/candidates")
    ids = set(review["decisions"])
    if not {"mmstar:" + q for q in initial} <= ids or len(ids) > len(initial) + POLICY["reserve_review_limit"]:
        raise ValueError("Assistant review coverage or bounded replacement budget differs")
    image_ids = set()
    for case_id, decision in review["decisions"].items():
        qid = case_id.removeprefix("mmstar:")
        if not case_id.startswith("mmstar:") or qid not in archive.rows:
            raise ValueError("Unknown reviewed MMStar question")
        row = archive.rows[qid]["baseline"]
        if (not row["correct"] or row["image_sha256"] in first["images"] or row["image_sha256"] in image_ids
                or decision["image_id"] != row["image_sha256"] or decision["image_sha256"] != row["image_sha256"]):
            raise ValueError("Reviewed addition is not a distinct, baseline-correct new image")
        image_ids.add(row["image_sha256"])
        if (decision["status"] not in {"keep", "exclude", "ambiguous", "uncertain", "annotation_issue"}
                or decision["confidence"] not in {"high", "medium", "low"} or not decision["notes"].strip()):
            raise ValueError("Incomplete assistant visual decision")
        if decision["status"] == "keep" and flags(archive.rows[qid]):
            raise ValueError("Kept new case violates the declared option/quality rules")
    return review


def build(args):
    if args.output.exists() or args.output.with_suffix(".zip").exists():
        raise FileExistsError("Use a new output path; previous review artifacts are preserved")
    first, human = load_human(args.first_manifest, args.human_review)
    archive = Archive(args.archive, "mmstar")
    try:
        if (first["source"]["predictions_sha256"] != archive.predictions_sha256
                or first["source"]["archive_sha256"] != digest(args.archive)):
            raise ValueError("First-round candidates and full MMStar archive differ")
        reference = json.loads((REPO / "research/provenance/original_llava_reference.json").read_text())
        if any(digest(REPO / p) != sha for p, sha in reference["native_source_sha256"].items()):
            raise ValueError("Monitored native ApET source hashes changed")
        picks, shortfalls = select_additions(archive.rows, first, human, args.seed)
        review = validate_review(args.assistant_review, archive, args.first_manifest, first, picks)
        original_cases, additions, core, qualitative = [], [], [], []
        for original in first["cases"]:
            case = copy.deepcopy(original)
            decision = human["decisions"][case["case_id"]]
            case["curation_review"] = {**decision, "reviewer": "user", "human_validated": True}
            original_cases.append(case)
            if decision["status"] == "keep":
                if case["predictions"]["baseline"]["correct"]:
                    core.append(case)
                else:
                    case["analysis_role"] = "qualitative_gain" if case["outcomes"]["input-only"] == "gain" else "baseline_failure_context"
                    qualitative.append(case)
        pick_lookup = {qid: slot for qid, slot in picks}
        for case_id, decision in review["decisions"].items():
            qid = case_id.split(":", 1)[1]
            family = pick_lookup[qid]["family"] if qid in pick_lookup else "Targeted replacement after image/choice review"
            case = archive.case(qid, "targeted_baseline_correct" if qid in pick_lookup else "reviewed_replacement", family)
            case["curation_review"] = {**decision, "reviewer": "assistant", "human_validated": False}
            additions.append(case)
            if decision["status"] == "keep":
                core.append(case)
        for case in core:
            if not case["predictions"]["baseline"]["correct"] or case["outcomes"]["input-only"] not in {"loss", "preserved_correct"}:
                raise ValueError("A baseline-wrong case leaked into the primary set")
            case["analysis_role"] = "main_" + case["outcomes"]["input-only"]
            case["sensitivity_flags"] = []
            if "typo" in case["curation_review"]["notes"].lower():
                case["sensitivity_flags"].append("Original option-name typo retained; report sensitivity without this case")
            if len([s for s in option_texts(case["question"]).values() if normalize_text(s) not in {"", "nan", "none", "n/a"}]) != 4:
                case["sensitivity_flags"].append("Fewer than four real choices; do not assume 25% guessing probability")
            case["same_image_preserved_controls"] = [c["question_id"] for c in core
                if c["image_id"] == case["image_id"] and c["case_id"] != case["case_id"]
                and c["outcomes"]["input-only"] == "preserved_correct"]
            case["curation_cautions"] = list(case["sensitivity_flags"])
        args.output.mkdir(parents=True)
        images = {}
        for case in original_cases + additions:
            image = case["image_id"]
            if image not in images:
                row = archive.rows[case["question_id"]]["baseline"]
                images[image] = save_image(archive.raw_image(case["question_id"]), args.output,
                    row["image_sha256"], row["rgb_pixels_sha256"])
                images[image]["display_id"] = "MMStar image " + image[:12]
        source = {"archive_file": args.archive.name, "archive_sha256": digest(args.archive),
            "predictions_sha256": archive.predictions_sha256,
            "earlier_report_recorded_prediction_hash": bool(archive.report.get("predictions_sha256")),
            "first_manifest_sha256": digest(args.first_manifest), "human_review_sha256": digest(args.human_review),
            "assistant_review_sha256": digest(args.assistant_review), "policy_id": POLICY["id"]}
        subsets = []
        for name, cases in [("recommended-manifest.json", core), ("qualitative-manifest.json", qualitative)]:
            subset = {"schema": "rq1-reviewed-selection-v1", "status": "reviewed_proposal", "dataset": "mmstar",
                "analysis_scope": "baseline_correct_primary" if name.startswith("recommended") else "qualitative_context_only",
                "source": source, "selection_seed": args.seed, "frozen_for_intervention": False,
                "human_review_required_next_round": False, "limitations": POLICY["limitations"],
                "cases": cases, "images": {i: images[i] for i in images if i in {c["image_id"] for c in cases}},
                "counts": summarize(cases)}
            dump(args.output / name, subset)
            subsets.append(subset)
        kept = [c for c in original_cases if c["curation_review"]["status"] == "keep"]
        initial_core = [c for c in kept if c["predictions"]["baseline"]["correct"]]
        full_baseline = [r["baseline"] for r in archive.rows.values()]
        valid_options = Counter()
        for row in full_baseline:
            options = option_texts(row["original_question"])
            count = sum(normalize_text(v) not in {"", "nan", "none", "n/a"} for v in options.values())
            valid_options[str(count) if options else "unparsed"] += 1
        report = {"schema": "mmstar-rq1-round2-report-v1", "source": source,
            "human_review_counts": dict(Counter(d["status"] for d in human["decisions"].values())),
            "human_keeps": summarize(kept), "human_primary": summarize(initial_core),
            "initial_candidates": len(picks), "replacement_reviews": len(additions) - len(picks),
            "assistant_counts": dict(Counter(c["curation_review"]["status"] for c in additions)),
            "new_primary": summarize([c for c in additions if c["curation_review"]["status"] == "keep"]),
            "primary": summarize(core), "qualitative": summarize(qualitative), "slot_shortfalls": shortfalls,
            "full_native_baseline": {"correct": sum(r["correct"] for r in full_baseline), "questions": len(full_baseline),
                "accuracy": sum(r["correct"] for r in full_baseline) / len(full_baseline),
                "predicted_options": dict(Counter(r["prediction"] for r in full_baseline)),
                "gold_options": dict(Counter(r["gold"] for r in full_baseline)),
                "real_choice_counts": dict(valid_options), "scoring_policy": "mmstar-explicit-choice-v2; project policy"},
            "limitations": POLICY["limitations"] + ["No same-image paired-question design is claimed for MMStar",
                "Dataset labels/ambiguities also matter; low baseline accuracy does not isolate reasoning versus perception",
                "Future pilot needs reference/no-op repeats; limited question-only/option-order controls can check apparent competence"],
            "native_sources_unchanged": True, "gpu_inference_run": False, "new_environment": False}
        dump(args.output / "review-report.json", report)
        dump(args.output / "policy.json", POLICY)
        dump(args.output / "initial-candidates.json", {"cases": [c for c in additions if c["question_id"] in pick_lookup], "shortfalls": shortfalls})
        dump(args.output / "reviewed-additions.json", {"cases": additions})
        selected = {qid for qid, _ in picks}
        decisions = review["decisions"]
        dump(args.output / "selection-audit.json", {"schema": "mmstar-rq1-selection-audit-v1", "rows": [
            {"question_id": qid, "image_sha256": rs["baseline"]["image_sha256"], "category": rs["baseline"]["category"],
             "baseline_correct": rs["baseline"]["correct"],
             "input_outcome": outcome(rs["baseline"]["correct"], rs["input-only"]["correct"]),
             "quality_flags": flags(rs), "first_round_image": rs["baseline"]["image_sha256"] in first["images"],
             "selected_initial": qid in selected, "assistant_reviewed": "mmstar:" + qid in decisions}
             for qid, rs in sorted(archive.rows.items())]})
        for filename, source_path in [("round1-human-review.json", args.human_review),
                                      ("round1-manifest.json", args.first_manifest), ("assistant-review.json", args.assistant_review)]:
            shutil.copyfile(source_path, args.output / filename)
        links = [("qualitative-gallery.html", "Qualitative gains and baseline failures")]
        render_decisions(args.output / "index.html", "MMStar RQ1: baseline-correct stress subset", core, images, args.output,
            "Reviewed primary subset for cross-task RQ1 checks. All baseline answers are correct; correctness alone is not proof of visual grounding. No new human review is requested.", links)
        render_decisions(args.output / "qualitative-gallery.html", "MMStar qualitative context", qualitative, images, args.output,
            "All original human Keep gains and both-wrong cases are retained here, outside the baseline-correct primary subset.", links)
        render_decisions(args.output / "review-audit.html", "MMStar full curation audit", original_cases + additions, images, args.output,
            "Original human decisions plus every assistant image decision, including rejected additions. Empty human reasons remain unspecified.", links)
        snapshots = ["scripts/prepare_mmstar_rq1_round2.py", "scripts/prepare_gqa_rq1_round2.py", "scripts/rq1_candidates.py",
            "scripts/prepare_rq1_candidates.py", "scripts/mmstar_scoring.py", "scripts/eval_mmstar.py",
            "research/provenance/original_llava_reference.json", "research/MMSTAR_RQ1_ROUND2.md"]
        for name in snapshots:
            target = args.output / "preparation-source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / name, target)
        dump(args.output / "preparation-sha256.json", {name: digest(REPO / name) for name in snapshots})
        with zipfile.ZipFile(args.output.with_suffix(".zip"), "x", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(args.output.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(args.output))
        print(json.dumps({k: report[k] for k in ["human_review_counts", "human_primary", "new_primary", "primary", "qualitative"]}, indent=2))
        return report
    finally:
        archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=REPO / MMSTAR_ARCHIVE)
    parser.add_argument("--first-manifest", type=Path, default=REPO / "research/results/rq1-candidates-v1-s590/mmstar/manifest.json")
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=592)
    parser.add_argument("--assistant-review", type=Path)
    parser.add_argument("--output", type=Path, default=REPO / "research/results/rq1-mmstar-reviewed-v1-s592")
    parser.add_argument("--preview", type=Path, help="Write the bounded candidate list/images for assistant visual inspection")
    args = parser.parse_args()
    if args.assistant_review:
        if args.preview:
            parser.error("Choose --assistant-review for the final bundle or --preview for candidate inspection")
        build(args)
        return
    first, human = load_human(args.first_manifest, args.human_review)
    archive = Archive(args.archive, "mmstar")
    try:
        picks, shortfalls = select_additions(archive.rows, first, human, args.seed)
        if args.preview:
            args.preview.mkdir(parents=True, exist_ok=False)
        cases = []
        for qid, slot in picks:
            case = archive.case(qid, "targeted_baseline_correct", slot["family"])
            if args.preview:
                row = archive.rows[qid]["baseline"]
                case["image_file"] = save_image(archive.raw_image(qid), args.preview, row["image_sha256"], row["rgb_pixels_sha256"])["file"]
            cases.append(case)
            print(qid, slot["family"], case["category"], case["outcomes"]["input-only"], "REF", case["canonical_target"])
            print(case["question"])
        if args.preview:
            dump(args.preview / "candidates.json", {"cases": cases, "shortfalls": shortfalls,
                "predictions_sha256": archive.predictions_sha256, "first_manifest_sha256": digest(args.first_manifest)})
        print("CANDIDATE_COUNTS", json.dumps(summarize(cases)))
        print("SHORTFALLS", json.dumps(shortfalls))
    finally:
        archive.close()


if __name__ == "__main__":
    main()
