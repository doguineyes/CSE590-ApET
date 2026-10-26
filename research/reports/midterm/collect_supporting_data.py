"""Collect midterm-v2 evidence and run a bounded, CPU-only E/loss probe.

Run from the repository root with a Python environment containing NumPy,
Pillow and Matplotlib. No Torch, model execution, source-archive mutation,
or changes to the report/protocols are involved.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "research/results"
SYNTHESIS = RESULTS / "rq1-input-ih-synthesis-v1-2026-10-18"
DEFAULT_OUTPUT = Path(__file__).with_name("supporting_data_2026-10-25")
SCHEDULES = ["96-to-32", "288-to-96"]
ARCHIVES = {
    "textvqa": "rq2-staged-textvqa100-v1-20261021T134751714241Z.zip",
    "gqa": "rq2-staged-gqa800-v1-20261021T142303483991Z.zip",
    "mmstar": "rq2-staged-mmstar1500-v1-20261021T153730115642Z.zip",
}
DESCRIPTIVE = ["nonbasis_mean", "nonbasis_std", "nonbasis_cv", "nonbasis_p90",
               "nonbasis_p90_over_median", "nonbasis_top10_mass"]
PRIMARY = ["log1p_nonbasis_mean", "nonbasis_cv", "nonbasis_top10_mass"]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    rows = list(rows)
    assert rows, path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def verify_correlations(out):
    """Independently recompute all plotted correlations from token records."""
    def ranks(values):
        a = np.asarray(values)
        order = np.argsort(a, kind="stable")
        sorted_a = a[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_a)) + 1]
        ends = np.r_[starts[1:], len(a)]
        result = np.empty(len(a))
        for start, end in zip(starts, ends):
            result[order[start:end]] = (start + end - 1) / 2
        return result
    tokens = read_csv(SYNTHESIS / "verified-comparison.csv")
    maximum_difference, count = 0., 0
    for q in read_csv(out / "rq1_correlations_30.csv"):
        records = [r for r in tokens if r["case_id"] == q["case_id"]]
        for field, x, y in [("rho_E_zero_first", "E", "zero_first_probability"),
                            ("rho_E_ih_first", "E", "ih_first_probability"),
                            ("rho_zero_ih_first", "zero_first_probability", "ih_first_probability")]:
            rho = np.corrcoef(ranks([float(r[x]) for r in records]), ranks([float(r[y]) for r in records]))[0, 1]
            maximum_difference = max(maximum_difference, abs(float(rho) - float(q[field])))
            count += 1
    assert count == 90 and maximum_difference < 1e-12
    write_json(out / "verification.json", {
        "recomputed_spearman_values": count,
        "maximum_absolute_difference_from_saved_synthesis": maximum_difference,
        "transition_identity": "All 27 conditions satisfy correct = baseline_correct - losses + gains",
        "fold_image_leakage": 0,
        "input_output_rows": {f: len(read_csv(out / f)) for f in ["panel_cases.csv", "table2_preservation.csv", "large_cohort_input_E.csv", "large_cohort_question_E_outcomes.csv", "decoder_E_statistics.csv", "e_loss_oof_predictions.csv"]},
    })


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def select_panel(out):
    protocol = read_json(ROOT / "research/protocols/rq2_input_merge_v2.json")
    ih_protocol = read_json(ROOT / "research/protocols/rq1_input_ih_v1.json")
    text_protocol = read_json(ROOT / "research/protocols/rq1_textvqa_input_v1.json")
    text_cases = {c["case_id"]: c for c in text_protocol["cases"]}
    assert {c["case_id"] for c in protocol["cases"]} == set(ih_protocol["case_ids"])
    manifest = {}
    with zipfile.ZipFile(RESULTS / "rq1-inputs-v1.zip") as z:
        for ds in ["gqa", "mmstar", "textvqa"]:
            name = next(n for n in z.namelist() if n.endswith(f"{ds}/manifest.json"))
            manifest[ds] = json.loads(z.read(name))
    rows = []
    for i, c in enumerate(protocol["cases"]):
        ds = c["case_id"].split(":")[0]
        predictions = c["historical_predictions"]
        if ds == "textvqa":
            predictions = text_cases[c["case_id"]]["archived_predictions"]
        base, compressed = predictions["baseline"], predictions["input-only"]
        assert base["correct"] is True
        role = "preserved" if compressed["correct"] else "loss"
        rows.append({
            "panel_order": i + 1, "case_id": c["case_id"], "dataset": ds,
            "image_id": c["image_id"], "image_sha256": c["image_sha256"],
            "question": c["question"], "target": c["canonical_target"],
            "selection_role": role, "selection_baseline_correct": True,
            "selection_input288_correct": compressed["correct"],
            "selection_baseline_response": base["response"],
            "selection_input288_response": compressed["response"],
            "selection_comparison": "Own archived full576 vs native input-only288; no decoder pruning",
            "not_selected_from": "Later RQ2 component/staged/random results or E/I values",
        })
    assert len(rows) == 30 and len({r["image_sha256"] for r in rows}) == 24
    assert Counter(r["selection_role"] for r in rows) == {"loss": 11, "preserved": 19}
    write_csv(out / "panel_cases.csv", rows)
    funnels = {}
    for ds, seed in [("gqa", 591), ("mmstar", 592)]:
        folder = RESULTS / f"rq1-{ds}-reviewed-v1-s{seed}"
        round1 = read_json(folder / "round1-manifest.json")["cases"]
        round2_name = "round2-reviewed-pool.json" if ds == "gqa" else "reviewed-additions.json"
        round2 = read_json(folder / round2_name)["cases"]
        recommended = read_json(folder / "recommended-manifest.json")["cases"]
        r1_ids, r2_ids = ({c["case_id"] for c in group} for group in (round1, round2))
        funnels[ds] = {
            "source_questions": 800 if ds == "gqa" else 1500,
            "source_scope": "GQA development; reserved evaluation untouched" if ds == "gqa" else "Pinned full MMStar set",
            "round1_candidates": len(round1),
            "round1_images": len({c["image_id"] for c in round1}),
            "round2_review_records": len(round2),
            "round2_unique_candidates": len(r2_ids),
            "round1_round2_overlap": len(r1_ids & r2_ids),
            "unique_candidates_across_rounds": len(r1_ids | r2_ids),
            "recommended_primary_questions": len(recommended),
            "recommended_primary_images": len({c["image_id"] for c in recommended}),
            "final_panel": dict(Counter(r["selection_role"] for r in rows if r["dataset"] == ds)),
            "curation_report": read_json(folder / "review-report.json"),
        }
    funnels["textvqa"] = {
        "source_questions": 5000, "source_scope": "TextVQA v0.5.1 validation annotations",
        "screening_questions": len(manifest["textvqa"]["cases"]),
        "screening_images": len({c["image_id"] for c in manifest["textvqa"]["cases"]}),
        "early_gallery_questions_nested_in_screening": 30,
        "selection": text_protocol["selection"],
        "final_panel": dict(Counter(r["selection_role"] for r in rows if r["dataset"] == "textvqa")),
    }
    funnels["final_selection_rule"] = {
        "gqa": "Fixed IDs, four loss/control image pairs plus two preserved/preserved image pairs, chosen from the reviewed 49-question pool.",
        "mmstar": "First 12 of the reviewed 18 primary cases under the stable SHA256 ordering in scripts/rq1_protocol.py.",
        "textvqa": "All three full-credit input288 losses from screening100 plus three visually reviewed literal-text preserved controls.",
        "limits": "Outcome-enriched, unblinded diagnostic selection; not a representative accuracy sample. Round 2 assistant review is not independent human adjudication. Selection labels and later fresh outputs are distinct.",
    }
    write_json(out / "panel_selection_funnel.json", funnels)
    return rows, funnels


def export_correlations_and_pair(out):
    rows = read_csv(SYNTHESIS / "question-synthesis.csv")
    assert len(rows) == 30 and all(int(r["positions"]) == 576 for r in rows)
    write_csv(out / "rq1_correlations_30.csv", rows)
    long = []
    mapping = {"rho_E_zero_first": "E vs zeroing", "rho_E_ih_first": "E vs retention",
               "rho_zero_ih_first": "Zeroing vs retention"}
    for row in rows:
        for key, label in mapping.items():
            long.append({"case_id": row["case_id"], "dataset": row["dataset"],
                         "image_id": row["image_id"], "comparison": label,
                         "rho": float(row[key]), "positions": 576,
                         "metric": "Signed first-reference-token probability difference",
                         "mmstar_grounding_flag": row["mmstar_grounding_flag"],
                         "mmstar_prior_quality_flag": row["mmstar_prior_quality_flag"]})
    write_csv(out / "rq1_correlations_long.csv", long)
    pair_ids = ["gqa:202218829", "gqa:202218593"]
    token_rows = read_csv(SYNTHESIS / "verified-comparison.csv")
    by_case = {cid: sorted((r for r in token_rows if r["case_id"] == cid), key=lambda r: int(r["token_index"])) for cid in pair_ids}
    qa = [read_json(SYNTHESIS / "questions" / (cid.replace(":", "-") + ".json")) for cid in pair_ids]
    assert qa[0]["image_sha256"] == qa[1]["image_sha256"]
    assert all(len(by_case[cid]) == 576 for cid in pair_ids)
    assert all([int(r["token_index"]) for r in by_case[cid]] == list(range(576)) for cid in pair_ids)
    e = np.array([float(r["E"]) for r in by_case[pair_ids[0]]])
    assert np.array_equal(e, [float(r["E"]) for r in by_case[pair_ids[1]]])
    grid = []
    for i in range(576):
        r = {"token_index": i, "patch_row": i // 24, "patch_col": i % 24,
             "E": float(e[i]), "fps_basis_source": by_case[pair_ids[0]][i]["fps_basis_source"]}
        for short, cid in zip(["window", "bottom"], pair_ids):
            for metric in ["ih_first_probability", "retained_probability", "full_probability", "remove_all_probability", "zero_first_probability"]:
                r[f"{short}_{metric}"] = float(by_case[cid][i][metric])
        grid.append(r)
    write_csv(out / "same_image_pair/grid.csv", grid)
    metadata = {"image_id": qa[0]["image_id"], "image_sha256": qa[0]["image_sha256"],
                "grid": [24, 24], "canvas_pixels": [336, 336], "patch_pixels": [14, 14],
                "E_arrays_exact_between_questions": True,
                "E_boundary": "Input CLIP states before projection; question-agnostic given image, FPS basis and seed",
                "retention_boundary": "Projected input states; all other visual positions removed, original position IDs retained",
                "interpretation": "Signed first-reference-token probability relative to the all-visual-removed counterfactual; not full-answer correctness or ground-truth importance. Different questions also have different target tokens and null priors.",
                "cases": []}
    for q, cid in zip(qa, pair_ids):
        r = by_case[cid][0]
        corr = next(x for x in rows if x["case_id"] == cid)
        metadata["cases"].append({"case_id": cid, "question": q["question"],
                                  "target": q["measurement_target"],
                                  "first_token": q["ih"]["first_reference_token_string"],
                                  "full_probability": float(r["full_probability"]),
                                  "remove_all_probability": float(r["remove_all_probability"]),
                                  "rho_E_retention": float(corr["rho_E_ih_first"]),
                                  "source": str((SYNTHESIS / "questions" / (cid.replace(":", "-") + ".json")).relative_to(ROOT))})
    with zipfile.ZipFile(RESULTS / ARCHIVES["gqa"]) as z:
        name = f"images/{metadata['image_sha256']}-canvas.png"
        (out / "same_image_pair/canvas.png").write_bytes(z.read(name))
        metadata["canvas_source"] = {"archive": ARCHIVES["gqa"], "member": name,
                                    "sha256": hashlib.sha256(z.read(name)).hexdigest()}
    write_json(out / "same_image_pair/metadata.json", metadata)
    return rows, grid, metadata


def error_stats(errors, basis):
    e = np.asarray(errors, dtype=np.float64)
    assert e.ndim == 1 and np.isfinite(e).all() and (e >= 0).all()
    basis = [int(i) for i in basis]
    assert len(set(basis)) == len(basis) and all(0 <= i < len(e) for i in basis)
    mask = np.ones(len(e), dtype=bool)
    mask[basis] = False
    answer = {"positions": len(e), "basis_positions": len(basis)}
    for prefix, a in [("all", e), ("nonbasis", e[mask])]:
        mean, std = float(a.mean()), float(a.std())
        median, p90 = float(np.median(a)), float(np.quantile(a, .9))
        total = float(a.sum())
        answer.update({f"{prefix}_mean": mean, f"{prefix}_std": std,
                       f"{prefix}_median": median, f"{prefix}_p90": p90,
                       f"{prefix}_min": float(a.min()), f"{prefix}_max": float(a.max()),
                       f"{prefix}_cv": std / mean if mean else 0.,
                       f"{prefix}_p90_over_median": p90 / median if median else 0.,
                       f"{prefix}_top10_mass": float(np.sort(a)[-math.ceil(.1 * len(a)):].sum() / total) if total else 0.})
    answer["log1p_nonbasis_mean"] = math.log1p(answer["nonbasis_mean"])
    return answer


def score_value(arm, ds):
    s = arm["scoring"]
    return float(s["consensus_score"] if ds == "textvqa" else s["correct"])


def collect_cohorts(out):
    input_rows, question_rows, decoder_rows, summaries, inventories = [], [], [], [], []
    prompt_example, source_hashes = None, {}
    for ds, file in ARCHIVES.items():
        path = RESULTS / file
        source_hashes[file] = sha_file(path)
        print(f"Reading {ds}: {file}", flush=True)
        with zipfile.ZipFile(path) as z:
            enc_names = sorted(n for n in z.namelist() if n.startswith("encodings/") and n.endswith(".json"))
            q_names = sorted(n for n in z.namelist() if n.startswith("questions/") and n.endswith(".json"))
            by_case = {}
            exact_errors = exact_basis = 0
            max_error_delta = 0.
            for name in enc_names:
                bank = json.loads(z.read(name))
                features = {}
                inputs = [bank["encodings"]["native-" + sched]["input"] for sched in SCHEDULES]
                exact_errors += inputs[0]["errors"] == inputs[1]["errors"]
                exact_basis += inputs[0]["basis_indices"] == inputs[1]["basis_indices"]
                max_error_delta = max(max_error_delta, float(np.max(np.abs(np.asarray(inputs[0]["errors"]) - inputs[1]["errors"]))))
                for sched, inp in zip(SCHEDULES, inputs):
                    assert len(inp["errors"]) == 576 and len(inp["basis_indices"]) == 10
                    features[sched] = {"dataset": ds, "encoding_file": name,
                        "image_id": bank.get("image_id", ""), "image_sha256": bank["image_sha256"],
                        "encoding_seed": bank["seed"], "schedule": sched,
                        **error_stats(inp["errors"], inp["basis_indices"])}
                    input_rows.append(features[sched])
                ids = bank.get("case_ids", [bank.get("case_id")])
                for cid in ids:
                    assert cid not in by_case
                    by_case[cid] = features
            conditions = defaultdict(list)
            prompt_checks = Counter()
            image_ids = set()
            for name in q_names:
                q = json.loads(z.read(name))
                assert q["status"] == "passed"
                cid, image = q["case_id"], q["case"]["image_sha256"]
                image_ids.add(image)
                b = bool(q["arms"]["baseline576"]["scoring"]["correct"])
                for condition, arm in q["arms"].items():
                    if arm.get("control_only") or condition not in ["baseline576"] + [prefix + sched + suffix for sched in SCHEDULES for prefix, suffix in [("native-", ""), ("pure-random-", "-s101"), ("pure-random-", "-s202"), ("pure-random-", "-s303")]]:
                        continue
                    c = bool(arm["scoring"]["correct"])
                    conditions[condition].append({"baseline_correct": b, "correct": c,
                                                  "score": score_value(arm, ds)})
                    if ds == "textvqa":
                        expected_question = q["case"]["question"] + "\nAnswer the question using a single word or phrase."
                        expected_prompt = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n" + expected_question + " ASSISTANT:"
                        prompt_checks["arms"] += 1
                        prompt_checks["generation_question_matches"] += arm["generation"]["question"] == expected_question
                        prompt_checks["prompt_matches"] += arm["generation"]["prompt"] == expected_prompt
                        if cid == "textvqa:37533" and condition == "baseline576":
                            prompt_example = {"case_id": cid, "archive": file, "member": name,
                                              "question": q["case"]["question"],
                                              "generation": arm["generation"],
                                              "scoring": arm["scoring"]}
                for sched in SCHEDULES:
                    arm = q["arms"]["native-" + sched]
                    c = bool(arm["scoring"]["correct"])
                    f = by_case[cid][sched]
                    assert f["image_sha256"] == image
                    assert q["encoding_reference"]["file"] == f["encoding_file"]
                    record = {**f, "case_id": cid, "question_file": name,
                              "image_id": q["case"].get("image_id", ""),
                              "question": q["case"]["question"],
                              "baseline_correct": b, "native_correct": c,
                              "baseline_score": score_value(q["arms"]["baseline576"], ds),
                              "native_score": score_value(arm, ds),
                              "eligible_for_loss_probe": b,
                              "loss": int(not c) if b else None,
                              "preserved": bool(b and c), "gain": bool(not b and c)}
                    question_rows.append(record)
                    de = arm["decoder"]["E"]
                    assert len(de["errors"]) == int(sched.split("-to-")[0])
                    decoder_rows.append({"dataset": ds, "case_id": cid, "image_sha256": image,
                                         "schedule": sched, "layer": de["layer"],
                                         **error_stats(de["errors"], de["basis_indices"])})
            expected_n = {"textvqa": 100, "gqa": 800, "mmstar": 1500}[ds]
            assert len(q_names) == expected_n and len(by_case) == expected_n
            if ds == "textvqa":
                assert prompt_checks == {"arms": 900, "generation_question_matches": 900, "prompt_matches": 900}
            for condition, values in conditions.items():
                assert len(values) == expected_n
                B = sum(v["baseline_correct"] for v in values)
                correct = sum(v["correct"] for v in values)
                preserved = sum(v["baseline_correct"] and v["correct"] for v in values)
                losses = B - preserved
                gains = correct - preserved
                assert correct == B - losses + gains
                summaries.append({"dataset": ds, "condition": condition,
                                  "questions": expected_n, "baseline_correct": B,
                                  "correct": correct, "score_percent": 100 * np.mean([v["score"] for v in values]),
                                  "preserved": preserved, "losses": losses, "gains": gains,
                                  "apr_percent": 100 * preserved / B,
                                  "both_wrong": expected_n - B - gains,
                                  "correct_definition": "Full ten-annotator consensus credit" if ds == "textvqa" else "Frozen project scorer"})
            inventories.append({"dataset": ds, "archive": file, "archive_sha256": source_hashes[file],
                                "questions": len(q_names), "distinct_images": len(image_ids),
                                "encoding_records": len(enc_names),
                                "input_E_records": len(enc_names) * 2,
                                "input_positions_per_record": 576, "basis_positions_per_record": 10,
                                "decoder_E_records": len(q_names) * 2,
                                "decoder_positions": [96, 288],
                                "input_errors_exact_between_schedules": exact_errors,
                                "input_basis_exact_between_schedules": exact_basis,
                                "max_input_error_delta_between_schedules": max_error_delta,
                                "text_prompt_checks": dict(prompt_checks) if ds == "textvqa" else None})
    write_csv(out / "large_cohort_input_E.csv", input_rows)
    write_csv(out / "large_cohort_question_E_outcomes.csv", question_rows)
    write_csv(out / "decoder_E_statistics.csv", decoder_rows)
    write_csv(out / "preservation_by_condition.csv", summaries)
    table = []
    for ds in ARCHIVES:
        baseline = next(r for r in summaries if r["dataset"] == ds and r["condition"] == "baseline576")
        for sched in SCHEDULES:
            native = next(r for r in summaries if r["dataset"] == ds and r["condition"] == "native-" + sched)
            random = [r for r in summaries if r["dataset"] == ds and r["condition"].startswith("pure-random-" + sched)]
            assert len(random) == 3
            row = {"dataset": ds, "schedule": sched, "questions": native["questions"],
                   "baseline_correct": baseline["correct"], "full576_score_percent": baseline["score_percent"],
                   **{"native_" + k: native[k] for k in ["score_percent", "correct", "preserved", "losses", "gains", "apr_percent", "both_wrong"]},
                   "correct_definition": native["correct_definition"], "random_seeds": "101;202;303"}
            for key in ["score_percent", "correct", "preserved", "losses", "gains", "apr_percent"]:
                row["random_mean_" + key] = float(np.mean([r[key] for r in random]))
                row["random_sample_sd_" + key] = float(np.std([r[key] for r in random], ddof=1))
            table.append(row)
    write_csv(out / "table2_preservation.csv", table)
    assert prompt_example is not None
    (out / "textvqa_prompt.txt").write_text(prompt_example["generation"]["prompt"] + "\n")
    prompt_example.update({
        "template": "<Vicuna-v1 system> USER: <image>\\n{question}\\nAnswer the question using a single word or phrase. ASSISTANT:",
        "inputs": "Image and question; no added OCR transcript or reference answers",
        "project_generation": {"do_sample": False, "max_new_tokens": 16, "use_cache": True, "dtype": "FP16", "batch_size": 1},
        "reference_code": {"shell": "scripts/llava1_5/textvqa.sh", "loader": "llava/eval/model_vqa_loader.py", "scorer": "llava/eval/eval_textvqa.py",
                           "question_file": "data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl",
                           "conv_mode": "vicuna_v1", "temperature": 0, "loader_default_max_new_tokens": 128,
                           "actual_question_file_available_locally": False},
        "paper_comparison": {"paper_full576_TextVQA": 58.2, "project_screening100_full576": 49.7,
                             "difference_pp": -8.5,
                             "warning": "Different evaluation cohort. The local reference launcher names an OCR question file, and the reference scorer supports OCR-formatted prompts, but that input file is absent. Exact paper prompt parity is unverified. Neither OCR nor length differences establish the cause of the score gap; no paired prompt ablation was run."}})
    write_json(out / "textvqa_prompt_and_protocol.json", prompt_example)
    write_json(out / "E_availability.json", inventories)
    return question_rows, summaries, table, inventories, source_hashes


def weighted_auc(y, score, weight=None):
    y, score = np.asarray(y, dtype=float), np.asarray(score, dtype=float)
    weight = np.ones(len(y)) if weight is None else np.asarray(weight, dtype=float)
    order = np.argsort(score, kind="stable")
    y, score, weight = y[order], score[order], weight[order]
    starts = np.r_[0, np.flatnonzero(np.diff(score)) + 1]
    pos = np.add.reduceat(weight * y, starts)
    neg = np.add.reduceat(weight * (1 - y), starts)
    if pos.sum() == 0 or neg.sum() == 0:
        return None
    return float(np.sum(pos * (np.cumsum(neg) - .5 * neg)) / (pos.sum() * neg.sum()))


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -40, 40)))


def fit_logistic(X, y):
    center, scale = X.mean(axis=0), X.std(axis=0)
    scale[scale < 1e-12] = 1.
    Z = np.c_[np.ones(len(X)), (X - center) / scale]
    beta = np.zeros(Z.shape[1])
    beta[0] = math.log((y.sum() + .5) / (len(y) - y.sum() + .5))
    penalty = np.diag([0.] + [1.] * (Z.shape[1] - 1))

    def objective(b):
        eta = Z @ b
        return float(np.sum(np.logaddexp(0, eta) - y * eta) + .5 * b @ penalty @ b)

    converged = False
    for iteration in range(100):
        p = sigmoid(Z @ beta)
        gradient = Z.T @ (p - y) + penalty @ beta
        hessian = (Z.T * (p * (1 - p))) @ Z + penalty
        step = np.linalg.solve(hessian, gradient)
        alpha, current = 1., objective(beta)
        while objective(beta - alpha * step) > current and alpha > 1e-8:
            alpha *= .5
        beta -= alpha * step
        if np.max(np.abs(alpha * step)) < 1e-8:
            converged = True
            break
    assert converged, "Logistic solver failed to converge"
    return beta, center, scale, iteration + 1


def feasibility(out, question_rows, bootstrap_draws):
    # Analytic tie/ranking sanity checks; independent of the research outcomes.
    assert weighted_auc([0, 1], [0, 1]) == 1.
    assert weighted_auc([0, 1], [1, 0]) == 0.
    assert weighted_auc([0, 1], [1, 1]) == .5
    assert weighted_auc([0, 1], [1, 1], [3, 2]) == .5
    metrics, oof_rows, fold_records = [], [], []
    for ds in ARCHIVES:
        all_ds = [r for r in question_rows if r["dataset"] == ds]
        groups_all = sorted({r["image_sha256"] for r in all_ds}, key=lambda h: hashlib.sha256(f"59025:{h}".encode()).hexdigest())
        fold_by_group = {h: i % 5 for i, h in enumerate(groups_all)}
        for sched in SCHEDULES:
            records = [r for r in all_ds if r["schedule"] == sched and r["eligible_for_loss_probe"]]
            y = np.array([r["loss"] for r in records], dtype=float)
            X = np.array([[r[k] for k in PRIMARY] for r in records], dtype=float)
            groups = np.array([r["image_sha256"] for r in records])
            folds = np.array([fold_by_group[h] for h in groups])
            pred, prev = np.zeros(len(y)), np.zeros(len(y))
            for fold in range(5):
                test, train = folds == fold, folds != fold
                assert not (set(groups[test]) & set(groups[train]))
                assert len(set(y[train])) == 2
                beta, center, scale, iterations = fit_logistic(X[train], y[train])
                pred[test] = sigmoid(np.c_[np.ones(test.sum()), (X[test] - center) / scale] @ beta)
                prev[test] = (y[train].sum() + .5) / (train.sum() + 1)
                fold_records.append({"dataset": ds, "schedule": sched, "fold": fold,
                                     "train_questions": int(train.sum()), "test_questions": int(test.sum()),
                                     "test_losses": int(y[test].sum()), "train_groups": len(set(groups[train])),
                                     "test_groups": len(set(groups[test])), "image_overlap": 0,
                                     "features": PRIMARY, "train_center": center.tolist(), "train_scale": scale.tolist(),
                                     "intercept_and_standardized_coefficients": beta.tolist(), "solver_iterations": iterations})
            for r, fold, p, b in zip(records, folds, pred, prev):
                oof_rows.append({"dataset": ds, "schedule": sched, "case_id": r["case_id"],
                                 "image_sha256": r["image_sha256"], "fold": int(fold),
                                 "loss": r["loss"], "oof_predicted_loss_probability": float(p),
                                 "oof_train_prevalence_probability": float(b)})
            image_groups, inverse = np.unique(groups, return_inverse=True)
            rng = np.random.default_rng(59025)
            weights = np.array([np.bincount(rng.integers(0, len(image_groups), len(image_groups)), minlength=len(image_groups))[inverse] for _ in range(bootstrap_draws)])
            both = sum(len(set(y[inverse == i])) == 2 for i in range(len(image_groups)))

            def metric_row(name, scores, kind="AUROC", reference=None):
                if kind == "AUROC":
                    value = weighted_auc(y, scores)
                    boot = [weighted_auc(y, scores, w) for w in weights]
                else:
                    errors = (scores - y) ** 2
                    if reference is not None:
                        errors = (reference - y) ** 2 - errors
                    value = float(errors.mean())
                    boot = [float(np.average(errors, weights=w)) for w in weights]
                valid = [b for b in boot if b is not None]
                lower, upper = np.quantile(valid, [.025, .975])
                metrics.append({"dataset": ds, "schedule": sched, "metric": name,
                                "kind": kind, "estimate": value, "ci_low": float(lower), "ci_high": float(upper),
                                "eligible_baseline_correct_questions": len(y), "losses": int(y.sum()),
                                "loss_rate": float(y.mean()), "image_groups": len(image_groups),
                                "images_with_loss_and_preserved_questions": both,
                                "bootstrap_draws": bootstrap_draws, "valid_bootstrap_draws": len(valid),
                                "uncertainty_scope": "Image-cluster bootstrap of fixed scores; no model refit/split uncertainty"})
            for feature in DESCRIPTIVE:
                metric_row(feature, np.array([r[feature] for r in records]))
            metric_row("three_feature_logistic_OOF", pred)
            metric_row("three_feature_logistic_OOF_Brier", pred, "Brier")
            metric_row("train_prevalence_OOF_Brier", prev, "Brier")
            metric_row("Brier_improvement_over_prevalence", pred, "Brier improvement", prev)
            print(f"{ds} {sched}: eligible={len(y)}, losses={int(y.sum())}, OOF AUROC={weighted_auc(y, pred):.3f}", flush=True)
    write_csv(out / "e_loss_feasibility_metrics.csv", metrics)
    write_csv(out / "e_loss_oof_predictions.csv", oof_rows)
    write_json(out / "e_loss_model_folds.json", fold_records)
    write_json(out / "e_loss_feasibility_summary.json", {
        "status": "Exploratory feasibility probe; not an adaptive policy or independent confirmation",
        "plan": "analysis_plan.json", "metrics": metrics,
        "primary_features": PRIMARY,
        "outcome": "Loss of a baseline-correct answer in the first captured native staged run",
        "validation": "Five outcome-independent image-group folds, train-only standardization, fixed L2=1; no tuning",
        "limits": ["All cohorts already inspected; not confirmatory.",
                   "TextVQA has only 44 baseline-full-credit questions and 10/7 losses.",
                   "Correlated questions remain question-weighted; uncertainty resamples image groups.",
                   "Confidence intervals condition on fixed predictions; they omit refitting and split uncertainty.",
                   "Input E is question-agnostic given the image, FPS basis and seed. MMStar sometimes uses different encoding seeds for questions sharing image bytes. Decoder E is post-compression and excluded from the predictor.",
                   "A relation to failure alone does not establish a useful budget rule, calibrated probabilities, or improved accuracy at matched average tokens."]})
    return metrics


def make_figures(out, correlations, grid, metadata):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "savefig.facecolor": "white"})
    folder = out / "figures"
    folder.mkdir(exist_ok=True)
    datasets = ["gqa", "mmstar", "textvqa"]
    colors = ["#267d9c", "#c27420", "#8464a8"]
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), sharey=True)
    for ax, key, title in zip(axes, ["rho_E_zero_first", "rho_E_ih_first", "rho_zero_ih_first"],
                              ["E vs zeroing", "E vs retention", "Zeroing vs retention"]):
        for i, (ds, color) in enumerate(zip(datasets, colors)):
            selected = sorted((r for r in correlations if r["dataset"] == ds), key=lambda r: r["case_id"])
            jitter = np.random.default_rng(59025 + i).uniform(-.15, .15, len(selected))
            values = [float(r[key]) for r in selected]
            ax.scatter(i + jitter, values, color=color, s=31, alpha=.9, edgecolor="white", linewidth=.4, zorder=3)
            ax.plot([i - .25, i + .25], [np.median(values)] * 2, color="#222222", lw=2.2, zorder=4)
        ax.axhline(0, color="#888888", lw=.8)
        ax.set_xticks(range(3), ["GQA\n(n=12)", "MMStar\n(n=12)", "TextVQA\n(n=6)"])
        ax.set_ylim(-.65, .65)
        ax.set_title(title)
        ax.grid(axis="y", alpha=.16)
    axes[0].set_ylabel("Per-question Spearman correlation")
    fig.suptitle("Thirty selected questions · 576 input positions per question", fontsize=12)
    fig.text(.5, .015, "Each dot is one question; black bars are medians. Both responses use signed first-reference-token probability differences.", ha="center", fontsize=8)
    fig.tight_layout(rect=[0, .07, 1, .94])
    for ext in ["png", "pdf"]:
        fig.savefig(folder / f"rq1_correlation_strip.{ext}", dpi=190)
    plt.close(fig)
    pair = out / "same_image_pair"
    canvas = np.asarray(Image.open(pair / "canvas.png"))
    E = np.array([r["E"] for r in grid]).reshape(24, 24)
    signed = [np.array([r[f"{target}_ih_first_probability"] for r in grid]).reshape(24, 24) for target in ["window", "bottom"]]
    probabilities = [np.array([r[f"{target}_retained_probability"] for r in grid]).reshape(24, 24) for target in ["window", "bottom"]]
    limit = max(float(np.abs(a).max()) for a in signed)
    spec = [("E_map", E, "Reconstruction error E", "viridis", 0, float(E.max()))]
    for i, (name, score, probability) in enumerate(zip(["window", "bottom"], signed, probabilities)):
        spec.extend([(name + "_retention_signed", score, f"Retain-one sensitivity: {name.capitalize()}", "RdBu_r", -limit, limit),
                     (name + "_retained_probability", probability, f"Retain-one probability: {name.capitalize()}", "viridis", 0, 1)])
    for filename, data, title, cmap, vmin, vmax in spec:
        fig, ax = plt.subplots(figsize=(4.8, 4.3))
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set(xticks=[0, 5, 10, 15, 20, 23], yticks=[0, 5, 10, 15, 20, 23], xlabel="Patch column (0–23)", ylabel="Patch row (0–23)", title=title)
        fig.colorbar(im, ax=ax, shrink=.84)
        fig.tight_layout()
        fig.savefig(pair / f"{filename}.png", dpi=190)
        plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(14.3, 4.1))
    axes[0].imshow(canvas)
    axes[0].set_title("Actual CLIP canvas\n336 × 336 px (padding shown)")
    axes[0].axis("off")
    for ax, data, cmap, vmin, vmax, title in zip(axes[1:], [E] + signed, ["viridis", "RdBu_r", "RdBu_r"],
                                               [0, -limit, -limit], [E.max(), limit, limit],
                                               ["Shared reconstruction error E", "Retention: ‘Window’\nfirst tokenizer piece: ▁Window", "Retention: ‘Bottom’\nfirst tokenizer piece: ▁B"]):
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set(xticks=[0, 6, 12, 18, 23], yticks=[0, 6, 12, 18, 23], xlabel="Patch column", ylabel="Patch row", title=title)
        fig.colorbar(im, ax=ax, fraction=.045, pad=.025)
    fig.text(.5, .075, "Q1: What is the pan in front of?  → Window     |     Q2: Are the white paper towels in the bottom part or in the top of the photo?  → Bottom", ha="center", fontsize=9)
    fig.text(.5, .025, "Retention panels share a symmetric scale: p(first target piece | retain one state) − p(first target piece | remove all visual states).", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .14, 1, 1])
    for ext in ["png", "pdf"]:
        fig.savefig(pair / f"same_image_comparison.{ext}", dpi=190)
    plt.close(fig)


def write_readme(out, funnels, table, inventories, metrics):
    lines = ["# Midterm v2 supporting evidence", "", "Collected October 25, 2026. These files support revision of the report; the report itself and frozen experimental protocols were not changed. All computations use saved records, with no model/GPU runs.", "", "## 1. How the 30-case panel was chosen", "", "The RQ1 panel and the later RQ2 component panel use the same 30 questions across 24 images. They were selected from **our own full576 versus native input-only288 results**, without decoder compression. The selection was enriched for compression failures and preserved controls: **11 losses and 19 preserved answers**, all baseline-correct. These are historical selection labels, not the outcomes of later component/staged captures. E/I values and later random/merge results were not used to choose these cases.", "", "| Dataset | Candidate history | Reviewed primary pool | Final panel |", "|---|---|---:|---|",
             f"| GQA | 60 first-round cases; 39 new-image candidates plus 10 same-image replacement reviews in round 2; {funnels['gqa']['unique_candidates_across_rounds']} unique cases across rounds | 49 | 12: 4 losses + 8 preserved; 6 images |",
             f"| MMStar | 30 first-round cases; 12 candidates plus 6 replacement reviews in round 2; {funnels['mmstar']['unique_candidates_across_rounds']} unique cases across rounds | 18 | 12: 4 losses + 8 preserved; 12 images |",
             "| TextVQA | Seeded 100-question/100-image screening cohort; the earlier 30-image gallery is nested within it | 100 screened | 6: all 3 full-credit input288 losses + 3 reviewed literal-text preserved controls |", "",
             "GQA first-round review had 30 keeps, 13 exclusions, 11 ambiguous, 3 uncertain and 3 possible annotation issues. A stricter mechanism rubric adopted 22 of the keeps; 27 round-2 additions produced the 49-question pool. MMStar first-round review had 17 keeps, 12 exclusions and 1 ambiguous: seven baseline-correct keeps plus 11 additions formed the primary 18; ten other keeps remained qualitative. Round 2 was assistant-reviewed and unblinded, not independently human-validated. GQA final IDs were fixed as four loss/control pairs and two preserved pairs; MMStar used a stable SHA ordering of its 18 accepted cases. This is a diagnostic panel, not a population accuracy sample.", "", "See [selection funnel](panel_selection_funnel.json) and [all 30 cases with historical responses](panel_cases.csv).", "", "## 2. Preservation and transitions for all six Table 2 rows", "", "APR = preserved / baseline-correct. Loss means baseline-correct → native-wrong; gain means baseline-wrong → native-correct. For TextVQA these counts use **full consensus credit**, while the report's main score remains partial-credit consensus. Counts therefore should not be substituted for its 41.50/44.20 scores.", "", "| Cohort | Schedule | Baseline correct | Preserved | Losses | Gains | Native correct | APR |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in table:
        lines.append(f"| {r['dataset']} {r['questions']} | {r['schedule'].replace('-to-', '→')} | {r['baseline_correct']} | {r['native_preserved']} | {r['native_losses']} | {r['native_gains']} | {r['native_correct']} | {r['native_apr_percent']:.2f}% |")
    lines += ["", "The exact identity is native correct = baseline correct − losses + gains. Aggregate changes conceal paired transitions in every cohort, not only MMStar. [Six-row export](table2_preservation.csv) also contains random-seed means and sample SDs; [all 27 conditions](preservation_by_condition.csv) preserves each random seed separately. Fractional random mean counts are averages over three runs, not additional independent questions.", "", "## 3. Thirty per-question correlations", "", "[Wide CSV: 30 questions](rq1_correlations_30.csv); [long CSV: 90 comparison values](rq1_correlations_long.csv); [strip plot PNG](figures/rq1_correlation_strip.png) and [vector PDF](figures/rq1_correlation_strip.pdf). The primary columns use all 576 positions and signed first-reference-token probability differences. Nonbasis/photo filters and existing MMStar caveat flags remain in the wide export; no cases were dropped for this plot.", "", "## 4. Same-image example", "", "The pan/window image **n578564** has identical input E arrays for questions gqa:202218829 (‘Window’) and gqa:202218593 (‘Bottom’). [Combined image](same_image_pair/same_image_comparison.png), [vector PDF](same_image_pair/same_image_comparison.pdf), [individual maps and canvas](same_image_pair/metadata.json), and [576-position data](same_image_pair/grid.csv) are included. The two signed retention maps use the same symmetric scale. Raw retained probabilities are also exported as individual maps with a shared 0–1 scale.", "", "The image is the actual 336×336 model canvas including padding. Every heatmap cell is one of its 24×24 positions (14×14 pixels). These are contextual CLIP features, not independent raw pixel patches. The ‘Bottom’ question has a much larger all-removed probability than ‘Window’; comparing signed maps must account for their different targets and null priors. This illustration supports question dependence, not a causal explanation of an ApET failure.", "", "## 5. Exact TextVQA prompting and comparison limits", "", "[Verbatim Brolio prompt](textvqa_prompt.txt) and [protocol details](textvqa_prompt_and_protocol.json) are included. Project prompts use the Vicuna-v1 wrapper, image + question, then `Answer the question using a single word or phrase.` No extra OCR transcript or gold answers are supplied. All 900 primary generation arms in the screening100 archive were checked against that template. The runner uses greedy generation, at most 16 new tokens, batch size 1, FP16.", "", "The local upstream-reference launcher points to `llava_textvqa_val_v051_ocr.jsonl`, and its loader defaults to 128 new tokens. The actual question JSONL is absent locally; its exact prompt contents and parity with the paper are **not verified**. The paper's 58.2 full576 score and our screening100 score of 49.7 use different evaluation cohorts. Do not attribute that 8.5-point gap to prompting/OCR alone without a matched experiment.", "", "## 6. Saved E availability and exploratory loss prediction", "", "All three large-cohort ZIPs contain complete 576-position input E arrays, ten FPS basis source indices, and per-question decoder-stage E arrays. The exports include [input encoding statistics](large_cohort_input_E.csv), [question-linked outcomes](large_cohort_question_E_outcomes.csv), [decoder E statistics](decoder_E_statistics.csv), and [availability/integrity inventory](E_availability.json). GQA shares one encoding over eight questions; MMStar contains repeated image bytes and sometimes different per-case encoding seeds. Validation therefore groups by actual image hash, not question ID.", "", "The [analysis plan](analysis_plan.json) was written before calculating these associations, after the cohorts/outcomes had already been reviewed; it is exploratory, not a confirmatory preregistration. The outcome is loss **conditional on baseline correctness**. Primary E summaries omit the ten FPS source positions. A fixed three-feature ridge logistic model uses log(1 + mean E), E coefficient of variation and the fraction of E mass in the top 10% of positions. Five outcome-independent image-group folds, training-only standardization and no tuning are used. No decoder E is used for the predictor because it is measured after input compression.", "", "| Cohort | Schedule | Eligible / losses | E mean AUROC | Three-feature OOF AUROC [95% interval] | Brier improvement vs prevalence |", "|---|---|---:|---:|---:|---:|"]
    for ds in ARCHIVES:
        for sched in SCHEDULES:
            a = next(m for m in metrics if m['dataset'] == ds and m['schedule'] == sched and m['metric'] == 'three_feature_logistic_OOF')
            e = next(m for m in metrics if m['dataset'] == ds and m['schedule'] == sched and m['metric'] == 'nonbasis_mean')
            b = next(m for m in metrics if m['dataset'] == ds and m['schedule'] == sched and m['metric'] == 'Brier_improvement_over_prevalence')
            lines.append(f"| {ds} | {sched.replace('-to-', '→')} | {a['eligible_baseline_correct_questions']} / {a['losses']} | {e['estimate']:.3f} | {a['estimate']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}] | {b['estimate']:+.4f} |")
    lines += ["", "AUROC 0.5 is chance ranking; descriptive feature AUROCs retain their original direction (higher statistic predicts more losses). Brier improvement is prevalence Brier minus model Brier, so positive is better. Intervals resample image groups 1,000 times over fixed scores/predictions and do not include model refitting, split choice, fresh native numerical variation or external confirmation. The analysis is question-weighted. TextVQA has only 44 baseline-full-credit cases and 10/7 losses, so its precision is especially limited.", "", "**Interpretation:** mean E has a modest descriptive association with loss on MMStar (AUROC 0.608/0.638), little association on GQA (0.523/0.496), and uncertain estimates on TextVQA. The fixed three-feature model gives weak and inconsistent out-of-fold rankings, with no clear Brier improvement over the prevalence predictor (all six improvement intervals include zero). This provides no robust cross-dataset adaptive-budget rule. It also does not rule out other features or a useful future policy. Image complexity, question type and model competence may confound these associations; they have not been controlled here.", "", "Input E is question-agnostic **conditional on the image, FPS basis and seed**; it need not be identical across different seeded encodings of the same image. All 1,700 encoding records do have exactly matching input E and FPS basis indices across the two staged schedules. In GQA, 56 images at 96→32 and 27 at 288→96 contain both lost and preserved baseline-correct questions under the shared encoding. An image-only statistic cannot distinguish those outcomes within the image.", "", "[All feature/model metrics](e_loss_feasibility_metrics.csv), [OOF predictions](e_loss_oof_predictions.csv), [folds/coefficients](e_loss_model_folds.json), and [machine-readable summary](e_loss_feasibility_summary.json) are retained. These numbers test feasibility of an image-level failure-risk signal. They do **not** test an adaptive budget policy or demonstrate improved accuracy at matched mean tokens. A useful policy would still require a fixed rule and an independent, matched-token evaluation.", "", "## Sources and reproducibility", "", "The three raw staged archives, the frozen protocols, reviewed manifests and the existing RQ1 synthesis are authoritative sources. [Provenance](provenance.json) records source hashes and the code commit. Raw archives and the existing report are unchanged. Rebuild with `research/reports/midterm/collect_supporting_data.py` in a CPU environment containing NumPy, Pillow and Matplotlib. No large tensors or model weights are opened."]
    location = lines.index("## 5. Exact TextVQA prompting and comparison limits")
    lines[location:location] = ["For this pair, ‘Window’ is scored through tokenizer piece `▁Window`, while ‘Bottom’ is scored through **`▁B`**, not the full word. Their all-visual-removed probabilities are 0.01390 and 0.71827 respectively. Use the figure to discuss the defined first-piece response, rather than confidence in the complete words.", ""]
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=1000)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    assert args.bootstrap_draws >= 100
    assert (out / "analysis_plan.json").exists(), "Preserve the prior analysis plan in the output folder"
    panel, funnels = select_panel(out)
    correlations, grid, pair = export_correlations_and_pair(out)
    questions, conditions, table, inventory, archives = collect_cohorts(out)
    metrics = feasibility(out, questions, args.bootstrap_draws)
    make_figures(out, correlations, grid, pair)
    write_readme(out, funnels, table, inventory, metrics)
    verify_correlations(out)
    write_json(out / "midterm_supporting_data.json", {
        "date": "2026-10-25", "readable_guide": "README.md",
        "item1_selection": {"questions": 30, "images": 24, "historical_input288_losses": 11,
                            "historical_input288_preserved": 19, "funnel": "panel_selection_funnel.json", "cases": "panel_cases.csv"},
        "item2_table2_preservation": table,
        "item3_correlations": {"questions": 30, "comparisons_per_question": 3,
                               "wide": "rq1_correlations_30.csv", "long": "rq1_correlations_long.csv",
                               "plot": "figures/rq1_correlation_strip.png"},
        "item4_same_image_maps": {"metadata": "same_image_pair/metadata.json", "image": "same_image_pair/same_image_comparison.png", "data": "same_image_pair/grid.csv"},
        "item5_prompt": {"verbatim": "textvqa_prompt.txt", "protocol": "textvqa_prompt_and_protocol.json",
                         "missing": "Exact upstream OCR question JSONL; exact paper prompt parity and cause of score gap are unverified"},
        "item6_E_availability": inventory,
        "item6_primary_feasibility_metrics": [m for m in metrics if m["metric"] in ["nonbasis_mean", "three_feature_logistic_OOF", "Brier_improvement_over_prevalence"]],
        "new_gpu_runs": 0, "report_v2_changed": False,
    })
    sources = [ROOT / "research/protocols" / f for f in ["rq1_input_ih_v1.json", "rq1_textvqa_input_v1.json", "rq2_input_merge_v2.json"]]
    sources += [SYNTHESIS / f for f in ["question-synthesis.csv", "verified-comparison.csv"]]
    for ds, seed in [("gqa", 591), ("mmstar", 592)]:
        folder = RESULTS / f"rq1-{ds}-reviewed-v1-s{seed}"
        sources += [folder / f for f in ["round1-manifest.json", "recommended-manifest.json", "review-report.json",
                                        "round2-reviewed-pool.json" if ds == "gqa" else "reviewed-additions.json"]]
    sources += [RESULTS / "rq1-inputs-v1.zip", ROOT / "scripts/rq1_protocol.py", ROOT / "scripts/rq2_staged_textvqa_protocol.py",
                ROOT / "scripts/llava1_5/textvqa.sh", ROOT / "llava/eval/model_vqa_loader.py"]
    sources += [ROOT / "research/reports/papers/0_Ma_ApET_Approximation-Error_Guided_Token_Compression_for_Efficient_VLMs_CVPR_2026_paper.pdf"]
    sources += [Path(__file__), out / "analysis_plan.json"]
    write_json(out / "provenance.json", {
        "date": "2026-10-25", "source_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "raw_archives_sha256": archives,
        "source_files_sha256": {str(p.relative_to(ROOT)): sha_file(p) for p in sources},
        "invariants": {"panel_questions": len(panel), "panel_images": 24,
                       "correlation_questions": len(correlations), "input_E_question_schedule_records": len(questions),
                       "table_rows": len(table), "conditions": len(conditions), "report_modified": False,
                       "no_gpu_or_model_execution": True},
        "dependencies": {"numpy": np.__version__},
        "output_sha256": {str(p.relative_to(out)): sha_file(p) for p in sorted(out.rglob("*")) if p.is_file() and p.name != "provenance.json"},
    })
    print(f"Wrote {out}", flush=True)


if __name__ == "__main__":
    main()
