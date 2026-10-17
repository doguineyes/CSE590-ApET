"""CPU-only, role-aware review of saved RQ1 input measurements; no inference.

Writes a separate review folder; original results remain unchanged. Correlations
are descriptive, per question. Absolute-I correlation is an auxiliary influence
measure, not a replacement for the protocol's signed contribution score.
"""

import argparse
from collections import defaultdict
import csv
import html
import json
import math
import os
from pathlib import Path
import statistics

from rq1_protocol import average_ranks, sha256, spearman, write_json


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists():
        raise ValueError("Choose a new review directory; existing results are preserved")
    report = json.loads((source / "report.json").read_text())
    if report["status"] != "passed":
        raise ValueError("This summary expects a completed, passed run; inspect failed runs separately")
    cases = [json.loads(p.read_text()) for p in sorted((source / "questions").glob("*.json"))]
    for relative, digest in report["evidence_sha256"].items():
        if sha256(source / "evidence" / relative) != digest:
            raise ValueError(f"Saved evidence hash mismatch: {relative}")
    if len(cases) != report["progress"]["expected_questions"]:
        raise ValueError("Question files do not cover the complete run")
    rows, tokens, examples = [], [], []
    grouped = defaultdict(list)
    sections = []
    for case in cases:
        metadata = json.loads((source / "features" / f"{case['image_sha256']}.json").read_text())
        if sha256(source / "features" / metadata["tensor_file"]) != metadata["tensor_file_sha256"]:
            raise ValueError("Cached feature tensor hash mismatch")
        native = metadata["diagnostics"]
        basis = set(native["input"]["basis_indices"])
        selected = set(native["input"]["selected_indices"])
        retained = set(native["input_merge"]["target_indices"])
        slots = set(native["input_merge"]["basis_slots"])
        if len(basis) != 10 or len(retained) != 278 or len(slots) != 10 or retained & slots or selected != retained | slots:
            raise ValueError("Unexpected native 288-token selection/basis mapping")
        measured = case["tokens"]
        if case["status"] != "passed" or not case["controls"]["passed"] or not case["target_alignment"]["passed"]:
            raise ValueError("Question controls/alignment failed")
        if len({t["token_index"] for t in measured}) != len(measured):
            raise ValueError("Duplicate measured patch index")
        for token in measured:
            index = token["token_index"]
            expected = case["controls"]["reference"]["mean_logprob"] - token["perturbed_score"]["mean_logprob"]
            if not math.isfinite(expected) or abs(expected-token["intervention_score"]) > 1e-12:
                raise ValueError("Sensitivity does not equal recorded reference-minus-perturbed score")
            if abs(token["reconstruction_error"]-native["input"]["errors"][index]) > 1e-12:
                raise ValueError("Token error does not match native diagnostic")
        nonbasis = [t for t in measured if t["token_index"] not in basis]
        if len(nonbasis) < 3:
            raise ValueError("At least three measured non-basis patches are needed for role-aware correlations")
        eranks = average_ranks([t["reconstruction_error"] for t in nonbasis])
        iranks = average_ranks([t["intervention_score"] for t in nonbasis])
        for token, erank, irank in zip(nonbasis, eranks, iranks):
            n = len(nonbasis)
            if erank <= n*.25 and irank >= n*.75 and token["intervention_score"] > 0:
                examples.append({"case_id": case["case_id"], "image_id": case["image_id"],
                    "token_index": token["token_index"], "E": token["reconstruction_error"],
                    "I": token["intervention_score"], "direct_original_retained": token["token_index"] in retained,
                    "E_rank_percentile_nonbasis": (erank-1)/(n-1) if n > 1 else 0})
        for token in measured:
            index = token["token_index"]
            tokens.append({"case_id": case["case_id"], "image_id": case["image_id"], "token_index": index,
                "E": token["reconstruction_error"], "I": token["intervention_score"],
                "fps_basis_source": index in basis, "direct_original_retained": index in retained,
                "basis_replacement_slot": index in slots, "selected_mask_slot": index in selected})
        def correlation(items, absolute=False):
            return spearman([t["reconstruction_error"] for t in items],
                            [abs(t["intervention_score"]) if absolute else t["intervention_score"] for t in items])
        row = {"case_id": case["case_id"], "image_id": case["image_id"], "question": case["question"],
            "target": case["measurement_target"], "archived_input_outcome": case["outcomes"]["input-only"],
            "baseline_correct": case["free_generation"]["scoring"]["correct"], "measured_patches": len(measured),
            "rho_signed_all": correlation(measured), "rho_signed_nonbasis": correlation(nonbasis),
            "rho_abs_nonbasis_auxiliary": correlation(nonbasis, True),
            "positive_I": sum(t["intervention_score"] > 0 for t in measured),
            "negative_I": sum(t["intervention_score"] < 0 for t in measured),
            "zero_I": sum(t["intervention_score"] == 0 for t in measured),
            "I_min": min(t["intervention_score"] for t in measured),
            "I_max": max(t["intervention_score"] for t in measured)}
        rows.append(row)
        grouped[case["image_id"]].append(case)
        plot = os.path.relpath(source / "analysis" / f"{case['question_id']}.png", output)
        sections.append(f"<section><h2>{html.escape(case['case_id'])}: {html.escape(case['question'])}</h2>"
            f"<p>Target {html.escape(case['measurement_target'])}; archived input role {html.escape(row['archived_input_outcome'])}. "
            f"ρ all={row['rho_signed_all']:.3f}; non-basis={row['rho_signed_nonbasis']:.3f}.</p>"
            f"<img src='{html.escape(plot, quote=True)}' style='width:100%'></section>")
    pairs = []
    for image_id, items in grouped.items():
        for i, a in enumerate(items):
            for b in items[i+1:]:
                if a["clip_sha256"] != b["clip_sha256"] or a["projected_sha256"] != b["projected_sha256"]:
                    raise ValueError("Same-image questions used different features")
                first = {t["token_index"]: t["intervention_score"] for t in a["tokens"]}
                second = {t["token_index"]: t["intervention_score"] for t in b["tokens"]}
                common = sorted(set(first) & set(second))
                pairs.append({"image_id": image_id, "question_a": a["case_id"], "question_b": b["case_id"],
                    "common_patches": len(common), "rho_between_question_sensitivities": spearman(
                        [first[t] for t in common], [second[t] for t in common])})
    output.mkdir(parents=True)
    values = [r["rho_signed_all"] for r in rows if r["rho_signed_all"] is not None]
    nb_values = [r["rho_signed_nonbasis"] for r in rows if r["rho_signed_nonbasis"] is not None]
    summary = {"schema": "rq1-saved-pilot-review-v1", "source_commit": report["commit"],
        "measurement_version": report["version"], "source_report_sha256": sha256(source / "report.json"),
        "source_archive_sha256": sha256(source.with_suffix(".zip")) if source.with_suffix(".zip").exists() else None,
        "question_count": len(rows), "image_count": len(grouped), "intervention_count": len(tokens),
        "median_signed_rho": statistics.median(values), "signed_rho_range": [min(values), max(values)],
        "median_nonbasis_signed_rho": statistics.median(nb_values),
        "negative_question_rho_count": sum(v < 0 for v in values),
        "positive_question_rho_count": sum(v > 0 for v in values),
        "questions": rows, "pairs": pairs,
        "interpretation": "Descriptive, curated input-stage pilot. No independent-token population inference. "
            "Singleton zeroing is not native compression; basis sources are represented separately. "
            "A source absent from direct retained features may still contribute via merging. "
            "Low-E/high-I examples are post-hoc qualitative candidates, not established compression failures. "
            "Absolute-I rho is auxiliary influence magnitude, not the signed protocol score."}
    write_json(output / "summary.json", summary)
    write_csv(output / "questions.csv", rows)
    write_csv(output / "pairs.csv", pairs)
    write_csv(output / "tokens-with-native-roles.csv", tokens)
    write_csv(output / "low-error-positive-nonbasis-candidates.csv", sorted(examples, key=lambda x: (x["case_id"], -x["I"])))
    (output / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>RQ1 pilot review</title>"
        "<style>body{font:16px system-ui;max-width:1500px;margin:30px auto}section{border-top:1px solid #aaa;padding:20px}</style>"
        "<h1>Saved RQ1 pilot review</h1><p>Role-aware CPU analysis; original results are unchanged. "
        "FPS basis sources are not evidence of discarded low-error information. Padding patches contain contextualized CLIP features, "
        "so empty pixels do not imply an irrelevant embedding. See summary.json and CSV files.</p>" + "".join(sections), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.source, args.output)
    print(f"Reviewed {summary['question_count']} questions / {summary['image_count']} images / {summary['intervention_count']} interventions")
    print(f"Median signed rho: {summary['median_signed_rho']:.6f}; excluding basis: {summary['median_nonbasis_signed_rho']:.6f}")
