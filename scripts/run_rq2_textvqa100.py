"""All 100 frozen TextVQA screening questions: native / pure random at input288/64.

Project-owned evaluation, using the already calibrated capture/replay machinery.
No interventions or teacher-forced reference scores; generated-answer consensus
is the primary outcome. Execute in the original locked Linux environment.
"""
import argparse
import copy
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_original_env import REPO, provenance, validate_runtime, verify_native_reference
from rq1_protocol import sha256, write_json
from rq2_protocol import require, score_response, source_hashes
from rq2_random_protocol import CONDITIONS, BUDGETS, RANDOM_SEEDS, PARITY_CONDITION, specification
from rq2_textvqa_protocol import (VERSION, PROFILE, REFERENCE, DIAGNOSTIC_IDS, REPEAT_IDS,
    TENSOR_IDS, FRESH_CASE, validate_inputs, historical_comparison, summarize)
from rq2_native import assert_selection_equal
from run_rq1_input import load_model, save_image, case_seed, interrupt_run
from run_rq2_random_calibration import (MEASUREMENT as CALIBRATION_SOURCES, prepare_pixels,
                                       save_bank)
from smoke_original_apet import MODEL_ID, VISION_ID
from summarize_rq1_pilot import write_csv

MEASUREMENT = (*CALIBRATION_SOURCES, "scripts/rq2_textvqa_protocol.py",
    "scripts/run_rq2_textvqa100.py", "research/protocols/rq2_textvqa100_random_v1.json",
    "research/protocols/rq2_textvqa100_reference_v1.json", "environments/original/pyproject.toml",
    "environments/original/.python-version", "bootstrap/setup_original_linux.sh")
TESTS = ("research.tests.test_rq2_native", "research.tests.test_rq2_random_protocol",
         "research.tests.test_rq2_random_native", "research.tests.test_rq2_random_runner",
         "research.tests.test_rq2_textvqa_protocol", "research.tests.test_rq2_textvqa_runner",
         "research.tests.test_rq2_textvqa_native")
AUXILIARY = tuple(name.replace(".", "/") + ".py" for name in TESTS) + (
    "research/notebooks/kaggle_rq2_textvqa100.ipynb", "research/RQ2_TEXTVQA100.md")


def measurement_hashes():
    return source_hashes(extra=MEASUREMENT)


def prerequisite_signature():
    return {**measurement_hashes(), **{name: sha256(REPO / name) for name in AUXILIARY}}


def verify_prerequisite_report(args):
    proof = json.loads(args.prerequisite_report.read_text())
    require(proof.get("version") == VERSION and proof.get("status") == "passed"
            and proof.get("test_modules") == list(TESTS)
            and type(proof.get("tests_run")) is int and proof["tests_run"] > 26
            and proof.get("skipped") == 0, "TextVQA100 actual native prerequisites did not pass")
    require(proof.get("source_sha256") == prerequisite_signature(),
            "Code/protocol/tests changed since prerequisites; rerun them")
    require(proof.get("runtime_report_sha256") == sha256(args.runtime_report)
            and proof.get("setup_log_sha256") == sha256(args.setup_log),
            "Prerequisite runtime/setup evidence changed")
    return proof


def validate_bank(observations, image_sha256):
    """Within-run exact structure gates; no baseline correctness requirement."""
    full = observations["baseline576"]
    for name, observation in observations.items():
        require(observation["source_clip"] == full["source_clip"]
                and observation["source_patches"] == full["source_patches"],
                "CLIP source differs between conditions: " + name)
        require(observation["encoded"]["kept_tokens"] == specification(name)["budget"],
                "Captured input budget differs: " + name)
    pairs = {}
    for budget in BUDGETS:
        merge, no_merge = (observations[f"native-{budget}-{suffix}"] for suffix in ("merge", "no-merge"))
        pairs[str(budget)] = assert_selection_equal(merge, no_merge)
        require(merge["input"]["errors"] == observations["native-288-merge"]["input"]["errors"]
                and merge["input"]["basis_indices"] == observations["native-288-merge"]["input"]["basis_indices"],
                "E/FPS differs between input budgets")
    from rq2_random_protocol import random_selection
    for seed in RANDOM_SEEDS:
        small, large = (observations[f"pure-random-{k}-s{seed}"] for k in (64, 288))
        require(set(small["encoded"]["selected_indices_sorted"]) <=
                set(large["encoded"]["selected_indices_sorted"]), "Random masks are not nested")
        for obs in (small, large):
            k = obs["experiment"]["budget"]
            require(obs["uncompressed_projected"] == full["encoded"]["features"]
                    and obs["controls"]["no_fps_E_basis_or_merge"]
                    and obs["encoded"]["selected_indices_sorted"] == random_selection(image_sha256, seed, k)[0],
                    "Pure random changed original vectors or membership")
    if PARITY_CONDITION in observations:
        require(observations[PARITY_CONDITION]["encoded"] == full["encoded"],
                "Pure576 tuple does not equal full native tuple")
    return {"status": "passed", "source_identity_exact": True, "paired_selection": pairs,
            "E_FPS_identical_across_budgets": True, "pure_random_nested_and_original_vectors_exact": True}


def generate_answer(tokenizer, model, processor, image, case, pixels, condition, snapshot, seed):
    """Replay one frozen realization, with the original greedy generation path."""
    from rq1_native import observe_generation_tokens
    from rq2_random_native import runtime_scope, replay_snapshot, validate_snapshot
    from smoke_original_apet import generate_native_answer
    spec = specification(condition)
    require(snapshot["capture"]["seed"] == seed, "Snapshot seed differs")
    validate_snapshot(snapshot, pixels, condition)
    generation, tokens = {}, {}
    try:
        with runtime_scope(model, spec["budget"], seed) as name:
            with replay_snapshot(model, pixels, snapshot) as audit:
                with observe_generation_tokens(model, tokenizer, tokens):
                    generate_native_answer(tokenizer, model, processor, image, case["generation_question"],
                        name, seed=seed, max_new_tokens=16, require_nonempty=False, record=generation)
    finally:
        validate_snapshot(snapshot, pixels, condition)
    require(generation["trace"]["input"] == {"before": 576, "after": spec["budget"]}
            and not generation["trace"].get("decoder", [])
            and generation["expanded_prompt_tokens"] == generation["prompt_tokens_before_expansion"] - 1 + spec["budget"],
            "Generation does not use the declared input-only budget")
    return {"case_id": case["case_id"], "image_sha256": case["image_sha256"],
        "condition": condition, **spec, "seed": seed, "response": generation["response"],
        "scoring": score_response(case, generation["response"]), "generation": generation,
        "generated_tokens": generation["generated_tokens"], "generation_tokens": tokens,
        "frozen_replay": audit, "encoded_fingerprint": copy.deepcopy(snapshot["capture"]["encoding"]["encoded"])}


def compare_replay(first, again):
    keys = ("case_id", "image_sha256", "seed", "response", "scoring", "generation_tokens", "encoded_fingerprint")
    require(all(first[k] == again[k] for k in keys)
            and first["generation"]["prompt"] == again["generation"]["prompt"]
            and first["generation"]["expanded_prompt_tokens"] == again["generation"]["expanded_prompt_tokens"],
            "Frozen replay/full576 parity changed features, IDs, processed scores or answer")
    return {"status": "passed", "generated_ids_and_processed_scores_exact": True,
            "features_mask_prompt_and_answer_exact": True, "row": again}


def fresh_comparison(observed, snapshot, fresh_obs, fresh, first, row):
    from rq2_parity_diagnostic import compare_snapshots, selection_comparison
    keys = ("source_clip", "source_patches", "pre_merge", "basis_slot_to_source", "input", "input_merge")
    require(all(observed[k] == fresh_obs[k] for k in keys)
            and observed["encoded"]["mask"] == fresh_obs["encoded"]["mask"],
            "Fresh native capture changed source, E, FPS, selection or merge graph")
    differences = compare_snapshots(snapshot, fresh)
    # The older diagnostic defaults its display count to288. This comparison
    # covers both budgets; make its expected-count flags explicit and correct.
    differences["selection"] = selection_comparison(observed["encoded"]["selected_indices_sorted"],
        fresh_obs["encoded"]["selected_indices_sorted"], expected_budget=observed["encoded"]["kept_tokens"])
    return {"status": "recorded", "structure_exact": True,
        "numeric_differences": differences,
        "generated_ids_equal": first["generation_tokens"]["generated_token_ids"] == row["generation_tokens"]["generated_token_ids"],
        "consensus_score_delta": row["scoring"]["consensus_score"] - first["scoring"]["consensus_score"],
        "policy": "Record fresh numerical/answer variation; never replace the first primary realization"}


def run_panel(args, report, data, reference, loaded):
    from PIL import Image
    from rq2_random_native import capture_condition
    tokenizer, model, processor = loaded
    ids = [c["case_id"] for c in data["cases"]]
    report.update(expected_case_ids=ids, questions=[], feature_sha256={}, question_sha256={},
        progress={"completed_questions": 0, "expected_questions": len(ids),
                  "completed_primary_predictions": 0, "expected_primary_predictions": len(ids) * len(CONDITIONS)})
    for name in ("questions", "features", "encodings", "images"):
        (args.output / name).mkdir(exist_ok=True)
    complete_rows = []
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for case in data["cases"]:
            cid, key = case["case_id"], case["image_sha256"]
            seed = case_seed(case, 590)
            args.manifest = args.inputs / "textvqa/manifest.json"
            image = save_image(args, data, case)
            q = {"case_id": cid, "case": case, "status": "running", "seed": seed, "arms": {}, "controls": {}}
            path = args.output / "questions" / (cid.replace(":", "-") + ".json")
            encoding_path = args.output / "encodings" / (key + ".json")
            report["active_case"] = cid
            write_json(path, q)
            try:
                pixels, canvas = prepare_pixels(model, processor, image)
                Image.fromarray(canvas).save(args.output / "images" / (key + "-canvas.png"))
                observations, snapshots = {}, {}
                conditions = (*CONDITIONS, PARITY_CONDITION) if cid in DIAGNOSTIC_IDS else tuple(CONDITIONS)
                for condition in conditions:
                    print(cid + ": capture " + condition, flush=True)
                    observations[condition], snapshots[condition] = capture_condition(model, pixels, seed, condition, key)
                    write_json(encoding_path, {"case_id": cid, "seed": seed, "encodings": observations, "status": "capturing"})
                q["controls"]["bank"] = validate_bank(observations, key)
                write_json(encoding_path, {"case_id": cid, "seed": seed, "encodings": observations,
                    "controls": q["controls"]["bank"], "status": "passed"})
                encoding_ref = {"file": str(encoding_path.relative_to(args.output)), "sha256": sha256(encoding_path)}
                q["encoding_reference"] = encoding_ref
                if cid in TENSOR_IDS:
                    q["tensor_bank"] = save_bank(args, report, key, snapshots)
                for condition in CONDITIONS:
                    row = generate_answer(tokenizer, model, processor, image, case, pixels, condition, snapshots[condition], seed)
                    row["encoding_reference"] = {**encoding_ref, "condition": condition}
                    q["arms"][condition] = row
                    write_json(path, q)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    report["progress"]["completed_primary_predictions"] += 1
                    print(f"{cid}: {condition} = {row['response']!r}; consensus={row['scoring']['consensus_score']:.2f}", flush=True)
                full = q["arms"]["baseline576"]
                require(all(r["generation"]["prompt"] == full["generation"]["prompt"] for r in q["arms"].values()),
                        "Condition prompts differ")
                q["historical_comparison"] = historical_comparison(case, q["arms"], reference)
                if cid in DIAGNOSTIC_IDS:
                    parity = generate_answer(tokenizer, model, processor, image, case, pixels,
                        PARITY_CONDITION, snapshots[PARITY_CONDITION], seed)
                    q["controls"]["pure576_parity"] = compare_replay(full, parity)
                if cid in REPEAT_IDS:
                    q["controls"]["fixed_replays"] = {}
                    for condition in CONDITIONS:
                        again = generate_answer(tokenizer, model, processor, image, case, pixels,
                            condition, snapshots[condition], seed)
                        q["controls"]["fixed_replays"][condition] = compare_replay(q["arms"][condition], again)
                        write_json(path, q)
                if cid == FRESH_CASE:
                    q["controls"]["fresh_native"] = []
                    for budget in BUDGETS:
                        condition = f"native-{budget}-merge"
                        for repeat in range(2):
                            obs, fresh = capture_condition(model, pixels, seed, condition, key)
                            feature = save_bank(args, report, key + "-" + condition + f"-fresh{repeat + 1}", {condition: fresh})
                            entry = {"condition": condition, "repeat": repeat + 1, "encoding": obs, "feature": feature}
                            q["controls"]["fresh_native"].append(entry)
                            write_json(path, q)
                            row = generate_answer(tokenizer, model, processor, image, case, pixels, condition, fresh, seed)
                            entry.update(row=row, comparison=fresh_comparison(observations[condition], snapshots[condition],
                                obs, fresh, q["arms"][condition], row))
                            write_json(path, q)
                            del fresh
                q["status"] = "passed"
                write_json(path, q)
                report["question_sha256"][str(path.relative_to(args.output))] = sha256(path)
                complete_rows.extend(q["arms"].values())
                report["questions"].append({"case_id": cid, "status": "passed"})
                report["progress"]["completed_questions"] = len(report["questions"])
                write_json(args.output / "report.json", report)
                print(f"Completed {len(report['questions'])}/{len(ids)}: {cid}", flush=True)
                del snapshots, observations, pixels
            except BaseException as error:
                q.update(status="failed", error=f"{type(error).__name__}: {error}")
                write_json(path, q)
                raise
    report["summaries"] = summarize(complete_rows, expected_ids=ids, intervals=True)
    report.pop("active_case", None)


def copy_evidence(args, report):
    root = args.output / "evidence"
    files = {**{str(REPO / n): "sources/" + n for n in (*measurement_hashes(), *AUXILIARY)},
        str(args.inputs / "textvqa/manifest.json"): "inputs/textvqa/manifest.json",
        str(args.setup_log): "notebook/setup.log", str(args.runtime_report): "notebook/runtime.json",
        str(args.prerequisite_report): "notebook/prerequisites.json"}
    for source, relative in files.items():
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    report["evidence_sha256"] = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}


def write_gallery(output, report, questions):
    escape = lambda value: html.escape(str(value))
    parts = ['<!doctype html><meta charset="utf-8"><title>RQ2 TextVQA100</title>',
        '<style>body{font:16px system-ui;max-width:1150px;margin:2em auto;padding:1em}table{border-collapse:collapse;width:100%}td,th{padding:.35em;border:1px solid #ccc;text-align:left}img{width:336px;max-width:100%}article{margin:2em 0;border-top:2px solid #aaa}.wrong{background:#fee}.partial{background:#ffe}pre{white-space:pre-wrap}</style>',
        '<h1>RQ2: TextVQA100, 288 and 64 input tokens</h1>',
        '<p>Status: ' + escape(report["status"]) + '. Screening/development cohort; all questions retained. '
        'Native merge/no-merge and three pure-random seeds. Decoder pruning off. No speed claim.</p>',
        '<p><a href="report.json">Report</a> · <a href="predictions.jsonl">Predictions</a> · <a href="answers.csv">Answers CSV</a></p>']
    summary = report.get("summaries", {}).get("all100", {})
    if summary:
        parts.append('<table><tr><th>Condition</th><th>Consensus mean</th><th>Full credit</th><th>APR</th></tr>')
        for name, value in summary["conditions"].items():
            apr = value["APR"]
            parts.append(f'<tr><td>{escape(name)}</td><td>{value["mean_consensus_score"]:.1%}</td><td>{value["full_credit_count"]}/{summary["questions"]}</td><td>{format(apr,".1%") if apr is not None else "n/a"}</td></tr>')
        parts.append('</table>')
    for q in questions:
        case = q["case"]
        key = case["image_sha256"]
        parts.append(f'<article><h2>{escape(q["case_id"])}</h2><p>{escape(case["question"])}</p><p>Status: {escape(q["status"])}</p><img src="images/{key}-canvas.png" alt="Actual 336 by 336 model canvas"><p>All ten references: {escape(case["gold_answers"])}</p><table><tr><th>Condition</th><th>Answer</th><th>Consensus</th></tr>')
        for condition, row in q.get("arms", {}).items():
            score = row["scoring"]["consensus_score"]
            css = "wrong" if score == 0 else "partial" if score < 1 else ""
            parts.append(f'<tr class="{css}"><td>{escape(condition)}</td><td>{escape(row["response"])}</td><td>{score:.2f}</td></tr>')
        parts.append('</table><p><a href="questions/' + q["case_id"].replace(":", "-") + '.json">Case and controls</a></p></article>')
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")


def finish_run(output, report, started):
    questions = [json.loads(p.read_text()) for p in sorted((output / "questions").glob("*.json"))]
    completed = [row for q in questions if q["status"] == "passed" for row in q.get("arms", {}).values()]
    if completed and "summaries" not in report:
        report["summaries"] = summarize(completed, intervals=False)
    report.update(wall_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
    report["question_sha256"] = {str(p.relative_to(output)): sha256(p) for p in (output / "questions").glob("*.json")}
    write_json(output / "report.json", report)
    rows = [{"case_id": q["case_id"], "case_status": q["status"], "condition": name,
             "question": q["case"]["question"], "image_sha256": q["case"]["image_sha256"],
             "response": r["response"], "consensus_score": r["scoring"]["consensus_score"],
             "full_credit": r["scoring"]["correct"], "seed": r["seed"],
             "random_seed": r["random_seed"], "budget": r["budget"]}
            for q in questions for name, r in q.get("arms", {}).items()]
    if rows:
        write_csv(output / "answers.csv", rows)
    history = [{"case_id": q["case_id"], "condition": name, **value}
               for q in questions for name, value in q.get("historical_comparison", {}).items()]
    write_json(output / "historical-comparison.json", history)
    write_gallery(output, report, questions)
    write_json(output / "artifact-sha256.json", {str(p.relative_to(output)): sha256(p)
               for p in sorted(output.rglob("*")) if p.is_file()})
    with zipfile.ZipFile(output.with_suffix(".zip"), "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    print("Status:", report["status"], "Download:", output.with_suffix(".zip"), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-textvqa100", action="store_true", required=True)
    parser.add_argument("--prerequisite-report", type=Path, required=True)
    parser.add_argument("--setup-log", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--vision-model", default=VISION_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-original-hf-cache"))
    parser.add_argument("--download-budget-gib", type=float, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-gpus", type=int, default=2)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.output.exists() or args.output.with_suffix(".zip").exists():
        parser.error("Preserve existing output; choose a new path")
    if args.inputs == args.output or args.inputs in args.output.parents:
        parser.error("Output must be outside accepted inputs")
    if args.expected_gpus < 1 or not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        parser.error("Invalid GPU count/download budget")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    signal.signal(signal.SIGTERM, interrupt_run)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = {"version": VERSION, "status": "running", "execution_passed": False,
        "scientific_claim": "No required accuracy threshold; preserve every method's results",
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    try:
        report.update(provenance("RQ2 TextVQA100 screening: generated-answer consensus, input288/64"))
        report["status"] = "running"
        validate_runtime(report)
        verify_native_reference(report)
        profile, data, reference = validate_inputs(args.inputs)
        report.update(profile=profile, profile_sha256=sha256(PROFILE), reference_sha256=sha256(REFERENCE),
            conditions=CONDITIONS, measurement_sources_sha256=measurement_hashes(),
            prerequisite=verify_prerequisite_report(args))
        copy_evidence(args, report)
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == args.expected_gpus,
                "Enable the expected CUDA GPUs")
        write_json(args.output / "report.json", report)
        print("Loading original LLaVA once: all100 questions, eleven primary conditions (1100 predictions).", flush=True)
        loaded = load_model(args, report)
        run_panel(args, report, data, reference, loaded)
        require([q["case_id"] for q in report["questions"]] == profile["case_ids"]
                and report["progress"]["completed_questions"] == 100
                and report["progress"]["completed_primary_predictions"] == 1100,
                "Incomplete or filtered TextVQA100 run")
        report.update(status="passed", execution_passed=True)
    except KeyboardInterrupt:
        report.update(status="interrupted", error="Partial evidence preserved")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    finally:
        finish_run(args.output, report, started)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
