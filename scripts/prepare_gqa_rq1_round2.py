#!/usr/bin/env python3
"""Extend GQA candidates using a verified first review and explicit quality rules."""

import argparse
from collections import Counter, defaultdict
import copy
import html
import json
from pathlib import Path
import re
import shutil
import zipfile

from prepare_rq1_candidates import GQA_ARCHIVE, dump, skeleton
from rq1_candidates import (Archive, CONFIGURATIONS, EXCLUDED_GQA_IMAGES,
                            KNOWN_CAUTIONS, REPO, choose_diverse, digest, outcome,
                            rank, save_image, write_manifest)

POLICY = {
    "id": "gqa-rq1-round2-quality-v1", "seed": 591,
    "image_slots": {"metadata_reference": 3, "loss": 7, "gain": 3},
    "questions_per_image": 3,
    "rules": {
        "broad_scene_label": r"which (?:place|room) is it|where is this|what place is pictured",
        "demographic_attribute": r"ethnicity|caucasian|\byoung\b|\bold man\b|\bolder\b|not female",
        "subjective_attribute": r"\b(?:small|large|huge|low|tall|narrow|wide|soft|crispy|happy|thin|fat|clean|dirty|antique)\b|how (?:big|long|hard)|look long|and short",
        "malformed_or_vacuous": r"rocky grass|covered by the sky|full table|sky.*higher than",
        "language_prior": r"healthier|mirror.*made of glass|(?:orange.*cake|cake.*orange)",
    },
    "answer_alias_groups": [["car", "cars"], ["tree", "trees"], ["bike", "bicycle"],
                            ["bat", "baseball bat"], ["racket", "tennis racket"],
                            ["teddy bear", "stuffed bear"]],
    "review_rubric": ["Specific visible referent and image-dependent evidence",
                      "Reference answer supported by visible evidence at the original image resolution",
                      "No morphology, synonym or category-specificity explanation for an apparent change",
                      "Avoid subjective/demographic targets and unverifiable material/state claims",
                      "Apply the same rubric to gains, losses and controls; retain rejection reasons"],
    "review_limitations": "Assistant visual review, not independent human validation; review is not blinded to saved outcomes",
    "reserved_evaluation_used": False, "native_changes": False,
}


def review_input(manifest_path, review_path):
    manifest = json.loads(manifest_path.read_text())
    review = json.loads(review_path.read_text())
    if (review.get("schema") != "rq1-review-overlay-v1" or review.get("dataset") != "gqa"
            or review.get("manifest_sha256") != digest(manifest_path)):
        raise ValueError("Review does not match the first GQA manifest")
    if set(review["decisions"]) != {c["case_id"] for c in manifest["cases"]}:
        raise ValueError("Review must cover the exact first-round case IDs")
    allowed = {"keep", "exclude", "ambiguous", "annotation_issue", "uncertain"}
    if any(d["status"] not in allowed or not isinstance(d["notes"], str) for d in review["decisions"].values()):
        raise ValueError("Incomplete or invalid first review")
    return manifest, review


def quality_flags(records):
    baseline = records["baseline"]
    question = baseline["original_question"].lower()
    flags = [name for name, pattern in POLICY["rules"].items() if re.search(pattern, question)]
    normalize = lambda answer: re.sub(r"[^a-z0-9 ]", "", answer.lower()).strip()
    before, after = (normalize(records[c]["response"]) for c in ["baseline", "input-only"])
    if before != after and any(before in aliases and after in aliases for aliases in POLICY["answer_alias_groups"]):
        flags.append("likely_answer_alias")
    if str(baseline["question_id"]) in KNOWN_CAUTIONS:
        flags.append("previously_noted_ambiguity")
    if any(r["generated_tokens"] >= 16 for r in records.values()):
        flags.append("output_token_cap")
    return flags


def select_round2(rows, first_images, seed=591):
    groups, audit = defaultdict(list), []
    for qid in sorted(rows):
        records = rows[qid]
        baseline = records["baseline"]
        flags = quality_flags(records)
        exclusion = ("deferred_image" if baseline["image_id"] in EXCLUDED_GQA_IMAGES else
                     "round1_image" if baseline["image_id"] in first_images else None)
        audit.append({"question_id": qid, "image_id": baseline["image_id"], "quality_flags": flags,
                      "image_exclusion": exclusion, "input_outcome": outcome(baseline["correct"], records["input-only"]["correct"])})
        if not exclusion and not flags:
            groups[baseline["image_id"]].append({**baseline, "question_id": qid})
    groups = {i: qs for i, qs in groups.items() if len({q["original_question"] for q in qs}) >= 3}
    reference = sorted(groups, key=lambda image: rank(seed, "gqa-r2-reference", image))[:POLICY["image_slots"]["metadata_reference"]]
    images = [(i, "quality_reference", None) for i in reference]
    shortfalls = []
    def role(qid):
        return outcome(rows[qid]["baseline"]["correct"], rows[qid]["input-only"]["correct"])
    for target, count in [("loss", POLICY["image_slots"]["loss"]), ("gain", POLICY["image_slots"]["gain"])]:
        available = [i for i, group in groups.items() if i not in {x[0] for x in images}
                     and any(role(q["question_id"]) == target for q in group)
                     and any(role(q["question_id"]) == "preserved_correct" for q in group)]
        picked = sorted(available, key=lambda image: rank(seed, "gqa-r2-diagnostic:" + target, image))[:count]
        images.extend((i, "outcome_diagnostic", target) for i in picked)
        if len(picked) < count:
            shortfalls.append({"role": target, "requested": count, "available": len(picked)})
    remaining = sorted((i for i in groups if i not in {x[0] for x in images}),
                       key=lambda image: rank(seed, "gqa-r2-fallback", image))
    total_images = sum(POLICY["image_slots"].values())
    images.extend((i, "quality_reference_fallback", None) for i in remaining[:total_images - len(images)])
    if len(images) != total_images:
        raise ValueError("Insufficient eligible new images")
    selections = []
    for image, stratum, target in images:
        group, initial = groups[image], []
        if target:
            for needed in [target, "preserved_correct"]:
                pool = [q for q in group if role(q["question_id"]) == needed]
                initial.append(min(pool, key=lambda q: rank(seed, "gqa-r2-question", q["question_id"])))
        else:
            for kind in ["query", "verify"]:
                pool = [q for q in group if q["types"]["structural"] == kind]
                if pool:
                    initial.append(min(pool, key=lambda q: rank(seed, "gqa-r2-question", q["question_id"])))
        chosen = choose_diverse(group, initial, 3, seed, "gqa-r2-diverse")
        for q in chosen:
            reason = (f"New image with quality-filtered input-only {target} and preserved control" if target else
                      "Reference image selected by hash without correctness quotas, conditional on question/answer quality filters" if stratum == "quality_reference" else
                      "Diagnostic quota shortfall; hash-selected quality-reference replacement image")
            selections.append((q["question_id"], stratum, reason))
    return selections, audit, shortfalls


def controls(cases):
    for case in cases:
        case["preserved_controls_by_configuration"] = {cfg: [c["question_id"] for c in cases
            if c["image_id"] == case["image_id"] and c["case_id"] != case["case_id"]
            and c["outcomes"][cfg] == "preserved_correct"] for cfg in CONFIGURATIONS[1:]}
        case["same_image_preserved_controls"] = case["preserved_controls_by_configuration"]["input-only"]


def validated_assistant_review(path, archive, manifest, first_manifest_sha):
    review = json.loads(path.read_text())
    if (review.get("schema") != "rq1-assistant-review-v1" or review.get("dataset") != "gqa"
            or review.get("human_validated") is not False or review.get("outcome_blinded") is not False
            or review.get("predictions_sha256") != archive.predictions_sha256
            or review.get("first_manifest_sha256") != first_manifest_sha):
        raise ValueError("Assistant review provenance differs from the candidate sources")
    initial = {c["question_id"] for c in manifest["cases"]}
    if set(review["initial_candidate_question_ids"]) != initial:
        raise ValueError("Assistant review covers a different initial candidate selection")
    if review["image_sha256"] != {i: m["sha256"] for i, m in manifest["images"].items()}:
        raise ValueError("Assistant review image set or content differs")
    decisions = review["decisions"]
    if not {"gqa:" + qid for qid in initial} <= set(decisions):
        raise ValueError("Assistant review must decide every initial candidate")
    for case_id, decision in decisions.items():
        qid = case_id.removeprefix("gqa:")
        if not case_id.startswith("gqa:") or qid not in archive.rows:
            raise ValueError("Unknown assistant-reviewed question")
        row = archive.rows[qid]["baseline"]
        if (decision["image_id"] not in manifest["images"] or decision["image_id"] != row["image_id"]
                or decision["image_sha256"] != row["image_sha256"]):
            raise ValueError("Assistant review is not bound to a selected new image")
        if (decision["status"] not in {"keep", "exclude", "ambiguous", "uncertain", "annotation_issue"}
                or decision["confidence"] not in {"high", "medium", "low"} or not decision["notes"].strip()):
            raise ValueError("Incomplete assistant review")
        if decision["status"] == "keep" and quality_flags(archive.rows[qid]):
            raise ValueError("Kept assistant case violates the recorded automatic quality rules")
    return review


def render_decisions(path, title, cases, images, directory, introduction):
    """Read-only, offline review result; no new review is requested."""
    import base64

    esc = html.escape
    groups = defaultdict(list)
    for case in cases:
        groups[case["image_id"]].append(case)
    parts = []
    for image, questions in groups.items():
        meta = images[image]
        data = base64.b64encode((directory / meta["file"]).read_bytes()).decode()
        cards = []
        for case in questions:
            decision = case["curation_review"]
            provenance = "Human review" if decision["reviewer"] == "user" else "Assistant visual review"
            predictions = "".join(f"<tr><th>{esc(cfg)}</th><td>{esc(case['predictions'][cfg]['response'])}</td>"
                f"<td>{case['predictions'][cfg]['correct']}</td></tr>" for cfg in CONFIGURATIONS)
            eligibility = ("<p class='caution'>Set aside by the stricter rubric: " + esc(", ".join(case["policy_flags"])) + "</p>"
                           if case.get("policy_flags") and decision["status"] == "keep" else "")
            control_ids = ", ".join(case.get("same_image_preserved_controls", [])) or "None in recommended selection"
            cards.append(f"<section><p class='small'>{esc(case['question_id'])} · {esc(case['category'])} · "
                f"input-only: {esc(case['outcomes']['input-only'])}</p><h3>{esc(case['question'])}</h3>"
                f"<p>Original reference: <b>{esc(case['canonical_target'])}</b></p><p><b>{provenance}: "
                f"{esc(decision['status'])}</b> · confidence: {esc(decision.get('confidence', 'not recorded'))}</p>"
                f"<p>{esc(decision['notes'] or 'Kept in the supplied human review; no additional visual validation asserted.')}</p>"
                f"{eligibility}<p class='small'>Preserved same-image controls (input-only): {esc(control_ids)}</p>"
                f"<table><tr><th>Configuration</th><th>Saved raw answer</th><th>Saved score</th></tr>{predictions}</table></section>")
        parts.append(f"<article><div class='photo'><img src='data:{meta['mime']};base64,{data}' alt='{esc(image)}'>"
                     f"<p>{esc(image)} · {meta['size'][0]} × {meta['size'][1]}</p></div><div>{''.join(cards)}</div></article>")
    page = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{margin:0;background:#f5f6f8;color:#17202a;font:16px/1.5 system-ui}header,main{max-width:1280px;margin:auto;padding:24px}
article{display:grid;grid-template-columns:minmax(260px,40%) 1fr;gap:24px;background:white;border:1px solid #ddd;border-radius:8px;padding:20px;margin:20px 0}
.photo{position:sticky;top:12px;align-self:start}.photo img{width:100%;max-height:620px;object-fit:contain}h3{font-size:18px}section{border-bottom:1px solid #ddd;padding-bottom:20px;margin-bottom:20px}
.small{font-size:13px;color:#536171}.caution{background:#fff1c2;padding:10px}table{width:100%;border-collapse:collapse;font-size:14px}th,td{border:1px solid #ddd;padding:6px;text-align:left;overflow-wrap:anywhere}
a{color:#174d86}@media(max-width:800px){article{grid-template-columns:1fr}.photo{position:static}header,main{padding:12px}}</style></head><body>"""
    page = page.replace("<head>", "<head><title>" + esc(title) + "</title>")
    page += f"<header><h1>{esc(title)}</h1><p>{esc(introduction)}</p><p>{len(cases)} questions / {len(groups)} images. "
    page += "Assistant judgments are not independent human validation. Labels, prompts and saved scores are unchanged.</p>"
    page += "<p><a href='index.html'>Recommended selection</a> · <a href='review-audit.html'>All review decisions</a> · <a href='review-report.json'>Summary</a> · <a href='recommended-manifest.json'>Manifest</a></p></header>"
    path.write_text(page + "<main>" + "".join(parts) + "</main></body></html>", encoding="utf-8")


def apply_review(args, archive, first, human, manifest):
    review = validated_assistant_review(args.assistant_review, archive, manifest, digest(args.first_manifest))
    shutil.copyfile(args.assistant_review, args.output / "assistant-review.json")
    first_cases, reviewed_cases, recommended, set_aside = [], [], [], []
    for original in first["cases"]:
        case = copy.deepcopy(original)
        decision = human["decisions"][case["case_id"]]
        case["curation_review"] = {**decision, "reviewer": "user", "human_validated": True}
        flags = quality_flags(archive.rows[case["question_id"]])
        case["policy_flags"] = flags
        first_cases.append(case)
        if decision["status"] == "keep":
            if flags:
                set_aside.append({"case_id": case["case_id"], "question": case["question"], "flags": flags,
                                  "original_human_status": "keep", "action": "Set aside from stricter mechanism proposal; original review unchanged"})
            else:
                recommended.append(case)
    initial = {c["question_id"]: c for c in manifest["cases"]}
    for case_id, decision in review["decisions"].items():
        qid = case_id.split(":", 1)[1]
        case = copy.deepcopy(initial[qid]) if qid in initial else archive.case(qid, "same_image_replacement",
            "Reviewed as a same-image replacement after initial candidate quality screening")
        case["curation_review"] = {**decision, "reviewer": "assistant", "human_validated": False}
        case["policy_flags"] = quality_flags(archive.rows[qid])
        reviewed_cases.append(case)
        if decision["status"] == "keep":
            recommended.append(case)
    controls(recommended)
    image_metadata = dict(manifest["images"])
    for case in first_cases:
        image = case["image_id"]
        if image not in image_metadata:
            image_metadata[image] = save_image(archive.raw_image(case["question_id"]), args.output,
                case["image_sha256"], archive.images[image]["rgb_pixels_sha256"])
    ids = [c["case_id"] for c in recommended]
    if len(ids) != len(set(ids)) or any(c["image_id"] in EXCLUDED_GQA_IMAGES for c in recommended):
        raise ValueError("Invalid recommended case identities")
    for case in recommended:
        if case["policy_flags"] or case["curation_review"]["status"] != "keep":
            raise ValueError("Unqualified case entered the mechanism proposal")
        if digest(args.output / image_metadata[case["image_id"]]["file"]) != case["image_sha256"]:
            raise ValueError("Recommended image differs from source")
    counts = lambda cases: {"questions": len(cases), "images": len({c["image_id"] for c in cases}),
        "input_outcomes": dict(Counter(c["outcomes"]["input-only"] for c in cases)),
        "categories": dict(Counter(c["category"] for c in cases))}
    proposed = {"schema": "rq1-reviewed-selection-v1", "status": "reviewed_proposal", "dataset": "gqa",
        "selection_seed": args.seed, "source": manifest["source"], "policy_id": POLICY["id"],
        "assistant_review_sha256": digest(args.assistant_review), "first_review_sha256": digest(args.human_review),
        "human_review_required_next_round": False, "frozen_for_intervention": False,
        "review_limitations": POLICY["review_limitations"], "cases": recommended,
        "images": {i: image_metadata[i] for i in image_metadata if i in {c["image_id"] for c in recommended}},
        "counts": counts(recommended)}
    unpaired = [c["case_id"] for c in recommended if c["outcomes"]["input-only"] in {"loss", "gain"}
                and not c["same_image_preserved_controls"]]
    report = {"schema": "gqa-rq1-round2-report-v1", "source": manifest["source"],
        "human_round1_counts": dict(Counter(d["status"] for d in human["decisions"].values())),
        "human_keeps_preserved": sum(d["status"] == "keep" for d in human["decisions"].values()),
        "human_keeps_adopted": sum(c["curation_review"]["reviewer"] == "user" for c in recommended),
        "human_keeps_set_aside_by_stricter_policy": set_aside, "initial_round2_candidates": len(initial),
        "same_image_replacement_reviews": len(reviewed_cases) - len(initial),
        "assistant_round2_counts": dict(Counter(c["curation_review"]["status"] for c in reviewed_cases)),
        "new_cases_adopted": counts([c for c in recommended if c["curation_review"]["reviewer"] == "assistant"]),
        "recommended": counts(recommended), "quota_shortfalls": manifest["quota_shortfalls"],
        "loss_gain_cases_without_preserved_same_image_control": unpaired,
        "tentative_gain_case_ids": [c["case_id"] for c in recommended if c["curation_review"].get("evidence_type") == "tentative_color_gain"],
        "broad_absence_control_case_ids": [c["case_id"] for c in recommended if c["curation_review"].get("evidence_type") == "broad_absence_control"],
        "limitations": [POLICY["review_limitations"], "Selection is outcome-enriched, not a population accuracy sample",
                        "A saved gain is not proof of improved visual reasoning", "Same-image controls are identified by unchanged saved scores, not causal attribution",
                        "Intervention instrumentation and reference repeats remain to be run; numerical investigation is deferred"],
        "native_sources_unchanged": True, "gpu_inference_run": False, "reserved_gqa_evaluation_used": False}
    dump(args.output / "recommended-manifest.json", proposed)
    dump(args.output / "review-report.json", report)
    dump(args.output / "round2-reviewed-pool.json", {"schema": "rq1-reviewed-pool-v1", "cases": reviewed_cases})
    render_decisions(args.output / "index.html", "GQA RQ1 recommended selection", recommended, image_metadata,
                     args.output, "Your review has been incorporated and the new round has been reviewed by the assistant. No additional human review is required for this proposal.")
    render_decisions(args.output / "review-audit.html", "GQA RQ1 complete review audit", first_cases + reviewed_cases,
                     image_metadata, args.output, "All first-round human decisions and second-round assistant decisions, including rejected cases and stricter-policy exclusions.")
    return report


def build(args):
    first, human = review_input(args.first_manifest, args.human_review)
    reference = json.loads((REPO / "research/provenance/original_llava_reference.json").read_text())
    if any(digest(REPO / p) != expected for p, expected in reference["native_source_sha256"].items()):
        raise ValueError("Current monitored native ApET sources differ from the saved reference")
    archive = Archive(args.archive, "gqa")
    try:
        picks, audit, shortfalls = select_round2(archive.rows, set(first["images"]), args.seed)
        args.output.mkdir(parents=True)
        dump(args.output / "policy.json", POLICY)
        shutil.copyfile(args.human_review, args.output / "round1-human-review.json")
        shutil.copyfile(args.first_manifest, args.output / "round1-manifest.json")
        source = {"archive_file": args.archive.name, "archive_sha256": digest(args.archive),
                  "predictions_sha256": archive.predictions_sha256, "dataset": archive.report["dataset"],
                  "first_manifest_sha256": digest(args.first_manifest), "human_review_sha256": digest(args.human_review),
                  "policy_sha256": digest(args.output / "policy.json"), "reviewed_development_only": True}
        manifest = skeleton("gqa", args.seed,
            f"Round 2: {len(picks)} additional questions on {len(picks) // 3} new images; explicit quality rules and documented quota shortfalls. "
            "Assistant review is a separate overlay. This is an enriched development selection.", source)
        manifest["selection_round"] = 2
        manifest["quota_shortfalls"] = shortfalls
        manifest["cases"] = [archive.case(*p) for p in picks]
        images = {c["image_id"] for c in manifest["cases"]}
        selected_ids = {c["question_id"] for c in manifest["cases"]}
        manifest["context_questions"] = [archive.case(qid, "context_only", "Other question on a new candidate image; not selected")
            for qid in sorted(archive.rows) if archive.rows[qid]["baseline"]["image_id"] in images and qid not in selected_ids]
        for case in manifest["cases"]:
            image = case["image_id"]
            if image not in manifest["images"]:
                manifest["images"][image] = save_image(archive.raw_image(case["question_id"]), args.output,
                    case["image_sha256"], archive.images[image]["rgb_pixels_sha256"])
        controls(manifest["cases"])
        write_manifest(args.output, manifest)
        selected = {p[0] for p in picks}
        for row in audit:
            row["selected_round2"] = row["question_id"] in selected
        dump(args.output / "selection-audit.json", {"schema": "gqa-rq1-quality-audit-v1", "rows": audit,
            "quality_flag_counts": dict(Counter(f for row in audit for f in row["quality_flags"])), "quota_shortfalls": shortfalls})
        if args.assistant_review:
            report = apply_review(args, archive, first, human, manifest)
            print(json.dumps({k: report[k] for k in ["human_keeps_adopted", "assistant_round2_counts", "recommended",
                "loss_gain_cases_without_preserved_same_image_control", "tentative_gain_case_ids"]}, indent=2))
        snapshots = ["scripts/prepare_gqa_rq1_round2.py", "scripts/rq1_candidates.py", "scripts/prepare_rq1_candidates.py",
                     "research/templates/rq1_gallery.html", "research/provenance/original_llava_reference.json",
                     "research/GQA_RQ1_ROUND2.md"]
        for relative in snapshots:
            target = args.output / "preparation-source" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(REPO / relative, target)
        dump(args.output / "preparation-sha256.json", {p: digest(REPO / p) for p in snapshots})
        print(json.dumps({"counts": manifest["counts"], "quota_shortfalls": shortfalls,
            "images": sorted(images), "human_counts": dict(Counter(d["status"] for d in human["decisions"].values()))}, indent=2))
    finally:
        archive.close()
    bundle = args.output.with_suffix(".zip")
    with zipfile.ZipFile(bundle, "x", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.output.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(args.output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=REPO / GQA_ARCHIVE)
    parser.add_argument("--first-manifest", type=Path, default=REPO / "research/results/rq1-candidates-v1-s590/gqa/manifest.json")
    parser.add_argument("--human-review", type=Path, required=True)
    parser.add_argument("--assistant-review", type=Path, default=None,
                        help="Optional recorded visual review; adds read-only recommendation and complete decision audit")
    parser.add_argument("--output", type=Path, default=REPO / "research/results/rq1-gqa-reviewed-v1-s591")
    parser.add_argument("--seed", type=int, default=591)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".zip").exists():
        parser.error("Output exists; choose another --output to preserve prior artifacts")
    build(args)


if __name__ == "__main__":
    main()
