"""Rebuild midterm tables from saved, reviewed data; no model dependencies.

Run from the repository root with any Python 3.10+ interpreter:
    python3 research/reports/midterm/build_materials.py

The source data under research/results are local/ignored. The compact output
tables and their provenance belong with the report and can be committed.
This checks derived arithmetic, not the original GPU measurements again.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "materials"
RQ1 = ROOT / "research/results/rq1-input-ih-synthesis-v1-2026-10-18"
RQ2 = ROOT / "research/results/rq2-staged-mmstar1500-review-v1-2026-10-21"


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    paths = [RQ1 / "question-synthesis.csv", RQ1 / "cohort-summary.json",
             RQ1 / "same-image-pairs.csv", RQ1 / "depth-comparison.csv",
             RQ2 / "cross-dataset-staged-summary.json"]
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }
    questions = read_csv(paths[0])
    cohort = json.loads(paths[1].read_text())
    pairs = read_csv(paths[2])
    depth = read_csv(paths[3])
    staged = json.loads(paths[4].read_text())

    assert len(questions) == len({q["case_id"] for q in questions}) == 30
    assert all(int(q["positions"]) == 576 for q in questions)
    assert len({q["image_id"] for q in questions}) == 24
    assert Counter(q["dataset"] for q in questions) == {
        "gqa": 12, "mmstar": 12, "textvqa": 6}
    assert len(pairs) == 6 and all(p["input_features_E_exact"] == "True" for p in pairs)
    assert len(depth) == 6

    correlations = []
    score_columns = ("rho_E_zero_first", "rho_E_ih_first", "rho_zero_ih_first")
    for name, count in (("gqa", 12), ("mmstar", 12), ("textvqa", 6)):
        group = [q for q in questions if q["dataset"] == name]
        recorded = next(d for d in cohort["datasets"] if d["group"] == name)
        row = {"dataset": name, "questions": count}
        for column in score_columns:
            value = statistics.median(float(q[column]) for q in group)
            assert math.isclose(value, recorded[column + "_median"], abs_tol=1e-12)
            row[column + "_median"] = value
        correlations.append(row)

    assert staged["shared_runtime_checkpoint_and_operator_contract"] is True
    assert len(staged["rows"]) == 6
    expected = {"TextVQA100": 100, "GQA800": 800, "MMStar1500": 1500}
    for row in staged["rows"]:
        assert row["questions"] == expected[row["dataset"]]
        assert math.isclose(row["native"] - row["random_seed_mean"],
                            row["native_minus_random"], abs_tol=1e-12)
        assert row["native_advantage_95_low"] <= row["native_minus_random"] <= row["native_advantage_95_high"]
    assert Counter((r["dataset"], r["schedule"]) for r in staged["rows"]) == {
        (d, s): 1 for d in expected for s in ("96-to-32", "288-to-96")}
    assert sum(expected.values()) == staged["question_evaluations"] == 2400
    assert 9 * sum(expected.values()) == staged["primary_predictions"] == 21600

    OUT.mkdir(exist_ok=True)
    write_csv("rq1_correlations.csv", correlations)
    write_csv("rq2_staged_results.csv", staged["rows"])
    summary = {
        "evidence_cutoff": "2026-10-21",
        "repository_commit_at_preparation": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "verification_scope": "CPU-only consistency checks on reviewed summaries; no new inference or raw-archive re-audit.",
        "source_sha256": source_hashes,
        "rq1": {
            "questions": 30, "images": 24, "positions_per_question": 576,
            "positions_per_complete_input_operator": 17280,
            "correlations": correlations,
            "individual_E_IH_min": min(float(q["rho_E_ih_first"]) for q in questions),
            "individual_E_IH_max": max(float(q["rho_E_ih_first"]) for q in questions),
            "same_image_pairs": len(pairs),
            "between_question_IH_rho_min": min(float(p["rho_IH_between_questions"]) for p in pairs),
            "between_question_IH_rho_max": max(float(p["rho_IH_between_questions"]) for p in pairs),
            "depth_questions": len(depth),
            "cross_stage_IH_rho_min": min(float(d["cross_stage_IH_rank_rho"]) for d in depth),
            "cross_stage_IH_rho_max": max(float(d["cross_stage_IH_rank_rho"]) for d in depth),
            "gqa_photo_nonbasis_E_IH_median": statistics.median(
                float(q["photo_nonbasis_rho_E_ih_first"]) for q in questions if q["dataset"] == "gqa"),
            "dominion": next(d for d in depth if d["case_id"] == "textvqa:35670"),
        },
        "rq2": staged,
        "limits": [
            "RQ1 is a curated diagnostic cohort; positions are not independent examples.",
            "First-piece probability is not generated whole-answer correctness.",
            "RQ2 staged comparison changes FPS, selection, basis placement and merging together.",
            "Paired intervals condition on the fixed cohorts, three random seeds and primary native realizations.",
            "The selected input-only component panel is separate from the staged comparison.",
        ],
    }
    (OUT / "verified_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("Verified saved-summary consistency: RQ1 30 questions/24 images; RQ2 2400 question evaluations/21600 primary predictions.")
    print("Wrote 2 CSV tables and verified_summary.json to " + str(OUT.relative_to(ROOT)))


if __name__ == "__main__":
    main()
