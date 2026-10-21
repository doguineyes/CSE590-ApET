"""All fixed TextVQA100 questions: native staged ApET versus matched pure random.

Uses the unchanged, calibrated input/decoder adapter. Generated-answer consensus
is primary; no teacher-forced intervention scores and no accuracy pass threshold.
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
from rq2_protocol import require, source_hashes
from rq2_staged_protocol import CONDITIONS, SCHEDULES, RANDOM_SEEDS, IDENTITY_CONDITION, specification, random_plan
from rq2_staged_textvqa_protocol import (VERSION, PROFILE, REFERENCE, CALIBRATION_REFERENCE,
    DIAGNOSTIC_IDS, REPEAT_IDS, TENSOR_IDS, PROBE_CASE, PROBE_CONDITIONS, FRESH_CASE,
    validate_inputs, historical_comparison, summarize, load_calibration_reference)
from run_rq1_input import load_model, save_image, case_seed, interrupt_run
from run_rq2_random_calibration import prepare_pixels
from run_rq2_staged_calibration import (MEASUREMENT as CALIBRATION_SOURCES,
    compare_replay, save_input_bank, save_decoder)
from smoke_original_apet import MODEL_ID, VISION_ID
from summarize_rq1_pilot import write_csv

MEASUREMENT = (*CALIBRATION_SOURCES, "scripts/rq2_staged_textvqa_protocol.py",
    "scripts/run_rq2_staged_textvqa100.py", "scripts/rq2_textvqa_protocol.py",
    "research/protocols/rq2_staged_textvqa100_v1.json",
    "research/protocols/rq2_staged_calibration_reference_v1.json",
    "research/protocols/rq2_textvqa100_reference_v1.json", "research/protocols/rq2_textvqa100_random_v1.json")
TESTS = ("research.tests.test_rq2_native", "research.tests.test_rq2_staged_protocol",
    "research.tests.test_rq2_staged_runner", "research.tests.test_rq2_staged_native",
    "research.tests.test_rq2_staged_textvqa_protocol", "research.tests.test_rq2_staged_textvqa_runner")
AUXILIARY = tuple(m.replace(".", "/") + ".py" for m in TESTS) + (
    "research/notebooks/kaggle_rq2_staged_textvqa100.ipynb", "research/RQ2_STAGED_TEXTVQA100.md")


def measurement_hashes():
    return source_hashes(extra=MEASUREMENT)


def prerequisite_signature():
    return {**measurement_hashes(), **{p: sha256(REPO / p) for p in AUXILIARY}}


def verify_prerequisites(args):
    p = json.loads(args.prerequisite_report.read_text())
    require(p.get("version") == VERSION and p.get("status") == "passed"
            and p.get("test_modules") == list(TESTS) and p.get("skipped") == 0
            and type(p.get("tests_run")) is int and p["tests_run"] > 26,
            "Require all current staged/TextVQA100 prerequisites with zero skips")
    require(p.get("source_sha256") == prerequisite_signature()
            and p.get("setup_log_sha256") == sha256(args.setup_log)
            and p.get("runtime_report_sha256") == sha256(args.runtime_report),
            "Sources or runtime changed since prerequisites; rerun them")
    return p


def validate_bank(observations, image_sha256):
    """Within-run checks independent of the model's answer correctness."""
    full = observations["baseline576"]
    require(set(CONDITIONS) <= set(observations) <= set(CONDITIONS) | {IDENTITY_CONDITION}, "Incomplete input bank")
    reference = observations["native-96-to-32"]["input"]
    for condition, obs in observations.items():
        spec = specification(condition)
        require(obs["source_clip"] == full["source_clip"] and obs["source_patches"] == full["source_patches"],
                "CLIP source changed between conditions: " + condition)
        require(obs["encoded"]["kept_tokens"] == spec["input_tokens"], "Input budget changed: " + condition)
        if spec["method"] == "native":
            require(obs["native_merge_executed"] and obs["merge_calls"] == 1
                    and obs["input"]["errors"] == reference["errors"]
                    and obs["input"]["basis_indices"] == reference["basis_indices"], "Native input E/FPS/merge changed")
        elif spec["method"] == "pure-random":
            require(obs["full_projected"] == full["encoded"]["features"]
                    and obs["pure_input"] == dict(fps=False, E=False, basis_replacement=False, merging=False)
                    and obs["random_plan"] == random_plan(image_sha256, spec["random_seed"], spec["input_tokens"], spec["decoder_tokens"])
                    and obs["encoded"]["selected_indices_sorted"] == obs["random_plan"]["input_original_positions"],
                    "Pure random input vectors, policy or membership changed")
    for seed in RANDOM_SEEDS:
        small = observations[f"pure-random-96-to-32-s{seed}"]["encoded"]["selected_indices_sorted"]
        large = observations[f"pure-random-288-to-96-s{seed}"]["encoded"]["selected_indices_sorted"]
        require(set(small) <= set(large), "Random input masks are not nested")
    if IDENTITY_CONDITION in observations:
        require(observations[IDENTITY_CONDITION]["encoded"] == full["encoded"], "Identity576 input differs")
    return {"status": "passed", "sources_exact": True, "native_input_E_FPS_shared_between_budgets": True,
            "pure_input_original_vectors_and_membership_exact": True, "random_input_nested": True}


def compact_row(row, encoding_ref):
    """Keep shared input diagnostics once per image; retain decoder records per arm."""
    row = dict(row)
    observation = row.pop("input_encoding")
    row["encoded_fingerprint"] = copy.deepcopy(observation["encoded"])
    row["encoding_reference"] = {**encoding_ref, "condition": row["condition"]}
    row["generated_tokens"] = row["generation"]["generated_tokens"]
    return row


def run_panel(args, report, profile, data, reference, loaded):
    from PIL import Image
    from rq2_staged_native import capture_input, generate_condition, generate_raw_native
    tokenizer, model, processor = loaded
    require(len(model.model.layers) == 32, "Staged run requires original 32-block LLaVA1.5-7B")
    ids = [c["case_id"] for c in data["cases"]]
    report.update(expected_case_ids=ids, questions=[], question_sha256={}, feature_sha256={}, encoding_sha256={},
        progress=dict(completed_questions=0, expected_questions=len(ids), completed_primary_predictions=0,
                      expected_primary_predictions=len(ids) * len(CONDITIONS)))
    for name in ("questions", "features", "encodings", "images"):
        (args.output / name).mkdir(exist_ok=True)
    complete = []
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for case in data["cases"]:
            cid, key = case["case_id"], case["image_sha256"]
            seed = case_seed(case, profile["seed"])
            args.manifest = args.inputs / "textvqa/manifest.json"
            image = save_image(args, data, case)
            q = {"case_id": cid, "case": case, "seed": seed, "status": "running", "arms": {}, "controls": {}}
            path = args.output / "questions" / (cid.replace(":", "-") + ".json")
            report["active_case"] = cid
            write_json(path, q)
            try:
                pixels, canvas = prepare_pixels(model, processor, image)
                Image.fromarray(canvas).save(args.output / "images" / (key + "-canvas.png"))
                observations, snapshots, layers = {}, {}, {}
                conditions = (*CONDITIONS, IDENTITY_CONDITION) if cid in DIAGNOSTIC_IDS else tuple(CONDITIONS)
                encoding_path = args.output / "encodings" / (key + ".json")
                for condition in conditions:
                    print(cid + ": capture input " + condition, flush=True)
                    observations[condition], snapshots[condition] = capture_input(model, pixels, seed, condition, key)
                q["controls"]["input_bank"] = validate_bank(observations, key)
                write_json(encoding_path, dict(case_id=cid, image_sha256=key, seed=seed,
                    encodings=observations, controls=q["controls"]["input_bank"]))
                encoding_ref = {"file": str(encoding_path.relative_to(args.output)), "sha256": sha256(encoding_path)}
                report["encoding_sha256"][encoding_ref["file"]] = encoding_ref["sha256"]
                q["encoding_reference"] = encoding_ref
                if cid in TENSOR_IDS:
                    q["input_tensor_bank"] = save_input_bank(args, report, key, snapshots)
                for condition in CONDITIONS:
                    print(cid + ": answer " + condition, flush=True)
                    row, layers[condition] = generate_condition(tokenizer, model, processor, image, case,
                        pixels, condition, snapshots[condition], seed)
                    row = compact_row(row, encoding_ref)
                    if cid in TENSOR_IDS and layers[condition] is not None:
                        row["decoder_tensor"] = save_decoder(args, report, cid, condition, layers[condition])
                    q["arms"][condition] = row
                    write_json(path, q)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    report["progress"]["completed_primary_predictions"] += 1
                    print(f"{cid}: {condition} = {row['response']!r}; consensus={row['scoring']['consensus_score']:.2f}", flush=True)
                full = q["arms"]["baseline576"]
                require(all(row["generation"]["prompt"] == full["generation"]["prompt"] for row in q["arms"].values()),
                        "Condition prompts differ")
                q["historical_comparison"] = historical_comparison(case, q["arms"], reference)
                if cid in DIAGNOSTIC_IDS:
                    row, _ = generate_condition(tokenizer, model, processor, image, case, pixels,
                        IDENTITY_CONDITION, snapshots[IDENTITY_CONDITION], seed)
                    row = compact_row(row, encoding_ref)
                    q["controls"]["identity576"] = compare_replay(full, row)
                if cid in REPEAT_IDS:
                    q["controls"]["fixed_replays"] = {}
                    for condition in CONDITIONS:
                        row, _ = generate_condition(tokenizer, model, processor, image, case, pixels,
                            condition, snapshots[condition], seed, decoder_replay=layers[condition])
                        row = compact_row(row, encoding_ref)
                        row["control_only"] = True
                        q["controls"]["fixed_replays"][condition] = compare_replay(q["arms"][condition], row)
                        write_json(path, q)
                if cid == PROBE_CASE:
                    q["controls"]["cached_decoding_probes"] = []
                    for condition in PROBE_CONDITIONS:
                        row, _ = generate_condition(tokenizer, model, processor, image, case, pixels,
                            condition, snapshots[condition], seed, decoder_replay=layers[condition], probe=True)
                        q["controls"]["cached_decoding_probes"].append(compact_row(row, encoding_ref))
                if cid == FRESH_CASE:
                    q["controls"]["fresh_raw_native"] = []
                    for a, b in SCHEDULES:
                        condition = f"native-{a}-to-{b}"
                        for repeat in range(2):
                            row = generate_raw_native(tokenizer, model, processor, image, case, condition, seed)
                            primary = q["arms"][condition]
                            q["controls"]["fresh_raw_native"].append(dict(repeat=repeat+1, row=row,
                                generated_ids_equal=row["generation_tokens"]["generated_token_ids"] == primary["generation_tokens"]["generated_token_ids"],
                                consensus_score_delta=row["scoring"]["consensus_score"] - primary["scoring"]["consensus_score"],
                                policy="Record fresh whole-native variation; never replace the first primary realization"))
                q["status"] = "passed"
                write_json(path, q)
                report["question_sha256"][str(path.relative_to(args.output))] = sha256(path)
                complete.extend(q["arms"].values())
                report["questions"].append(dict(case_id=cid, status="passed"))
                report["progress"]["completed_questions"] = len(report["questions"])
                write_json(args.output / "report.json", report)
                print(f"Completed {len(report['questions'])}/{len(ids)}: {cid}", flush=True)
                del snapshots, layers, observations, pixels
            except BaseException as error:
                q.update(status="failed", error=f"{type(error).__name__}: {error}")
                write_json(path, q)
                raise
    report["summaries"] = summarize(complete, expected_ids=ids, intervals=True)
    report.pop("active_case", None)


def copy_evidence(args, report):
    root = args.output / "evidence"
    files = {**{str(REPO / p): "sources/" + p for p in (*measurement_hashes(), *AUXILIARY)},
        str(args.inputs / "textvqa/manifest.json"): "inputs/textvqa/manifest.json",
        str(args.setup_log): "notebook/setup.log", str(args.runtime_report): "notebook/runtime.json",
        str(args.prerequisite_report): "notebook/prerequisites.json"}
    for source, relative in files.items():
        dest = root / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
    report["evidence_sha256"] = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*") if p.is_file()}


def finish_run(output, report, started):
    questions = [json.loads(p.read_text()) for p in sorted((output / "questions").glob("*.json"))]
    completed = [row for q in questions if q["status"] == "passed" for row in q.get("arms", {}).values()]
    if completed and "summaries" not in report:
        report["summaries"] = summarize(completed)
    report.update(wall_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
    report["question_sha256"] = {str(p.relative_to(output)): sha256(p) for p in (output / "questions").glob("*.json")}
    write_json(output / "report.json", report)
    rows = [dict(case_id=q["case_id"], case_status=q["status"], question=q["case"]["question"],
        condition=name, response=row["response"], consensus_score=row["scoring"]["consensus_score"],
        full_credit=row["scoring"]["correct"], seed=row["seed"], random_seed=row["random_seed"],
        input_tokens=row["input_tokens"], after_block16_tokens=row["decoder_tokens"] or 576)
        for q in questions for name, row in q.get("arms", {}).items()]
    if rows:
        write_csv(output / "answers.csv", rows)
    write_json(output / "historical-comparison.json", [{"case_id": q["case_id"], **q["historical_comparison"]}
        for q in questions if "historical_comparison" in q])
    esc = lambda x: html.escape(str(x))
    parts = ['<!doctype html><meta charset="utf-8"><title>RQ2 staged TextVQA100</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:1em}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.3em}img{width:336px;max-width:100%}pre{white-space:pre-wrap}article{margin:2em 0}.wrong{background:#fee}.partial{background:#ffe}</style>',
        '<h1>TextVQA100: staged ApET and matched pure random</h1><p>Status: ' + esc(report["status"]) +
        '. Input96→32 /288→96 after block16. Screening/development cohort; all100 retained. No speed claim.</p>',
        '<p><a href="report.json">Report</a> · <a href="answers.csv">Answers</a> · <a href="predictions.jsonl">Predictions</a></p>']
    for label, summary in report.get("summaries", {}).items():
        parts.append('<h2>' + esc(label) + ' (' + str(summary["questions"]) + ' questions)</h2><table><tr><th>Condition</th><th>Consensus mean</th><th>Full credit</th><th>APR</th></tr>')
        for name, v in summary["conditions"].items():
            apr = format(v["APR"], ".1%") if v["APR"] is not None else "n/a"
            parts.append(f'<tr><td>{esc(name)}</td><td>{v["mean_consensus_score"]:.1%}</td><td>{v["full_credit_count"]}/{summary["questions"]}</td><td>{apr}</td></tr>')
        parts.append('</table>')
    for q in questions:
        c = q["case"]
        parts.append('<article><h2>' + esc(q["case_id"]) + '</h2><p>' + esc(c["question"]) + '</p><p>Status: ' + esc(q["status"]) +
            '</p><img src="images/' + c["image_sha256"] + '-canvas.png" alt="Actual model canvas"><p>All ten references: ' +
            esc(c["gold_answers"]) + '</p><table><tr><th>Condition</th><th>Answer</th><th>Consensus</th></tr>')
        for name, row in q.get("arms", {}).items():
            score = row["scoring"]["consensus_score"]
            css = "wrong" if score == 0 else "partial" if score < 1 else ""
            parts.append(f'<tr class="{css}"><td>{esc(name)}</td><td>{esc(row["response"])}</td><td>{score:.2f}</td></tr>')
        parts.append('</table><p><a href="questions/' + q["case_id"].replace(":", "-") + '.json">Case and controls</a></p></article>')
    (output / "index.html").write_text("\n".join(parts), encoding="utf-8")
    write_json(output / "artifact-sha256.json", {str(p.relative_to(output)): sha256(p)
        for p in sorted(output.rglob("*")) if p.is_file()})
    with zipfile.ZipFile(output.with_suffix(".zip"), "x", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob("*")):
            if p.is_file():
                archive.write(p, p.relative_to(output))
    print("Status:", report["status"], "Download:", output.with_suffix(".zip"), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("inputs", "output", "setup-log", "runtime-report", "prerequisite-report"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--run-textvqa100", action="store_true", required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--vision-model", default=VISION_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-original-hf-cache"))
    parser.add_argument("--download-budget-gib", type=float, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-gpus", type=int, default=2)
    args = parser.parse_args(argv)
    for k, v in vars(args).items():
        if isinstance(v, Path):
            setattr(args, k, v.resolve())
    if args.output.exists() or args.output.with_suffix(".zip").exists():
        parser.error("Preserve existing output; choose a fresh path")
    if args.inputs == args.output or args.inputs in args.output.parents:
        parser.error("Output must be outside accepted inputs")
    if args.expected_gpus < 1 or not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        parser.error("Invalid GPU count/download budget")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    signal.signal(signal.SIGTERM, interrupt_run)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(version=VERSION, status="running", execution_passed=False, scientific_benchmark_passed=False,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    try:
        report.update(provenance("Staged TextVQA100: all fixed screening questions; native96→32/288→96 versus matched pure random"))
        report["status"] = "running"
        validate_runtime(report)
        verify_native_reference(report)
        profile, data, reference = validate_inputs(args.inputs)
        report.update(profile=profile, profile_sha256=sha256(PROFILE), conditions=CONDITIONS,
            reviewed_calibration=load_calibration_reference(), measurement_sources_sha256=measurement_hashes(),
            prerequisite=verify_prerequisites(args))
        copy_evidence(args, report)
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == args.expected_gpus, "Enable expected CUDA GPUs")
        write_json(args.output / "report.json", report)
        print("Validated all100 inputs. Loading native LLaVA once for 900 primary predictions plus bounded controls.", flush=True)
        loaded = load_model(args, report)
        run_panel(args, report, profile, data, reference, loaded)
        require([q["case_id"] for q in report["questions"]] == profile["case_ids"]
            and report["progress"]["completed_primary_predictions"] == 900, "Incomplete staged TextVQA100")
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
