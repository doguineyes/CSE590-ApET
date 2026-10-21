"""All pinned MMStar1500 questions: native staged ApET versus matched pure random.

Uses the unchanged, calibrated input/decoder adapter. Generated-answer accuracy
is primary; no teacher-forced intervention scores and no accuracy pass threshold.
"""
import argparse
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
from rq2_staged_protocol import CONDITIONS, SCHEDULES, IDENTITY_CONDITION
from rq2_staged_mmstar_protocol import (VERSION, PROFILE, REFERENCE, CALIBRATION_REFERENCE,
    REPEAT_IDS, TENSOR_IDS, PROBE_CASE, FRESH_CASE, validate_inputs, historical_comparison,
    summarize, load_calibration_reference, selected_ids)
from rq2_staged_textvqa_protocol import PROBE_CONDITIONS
from run_rq2_staged_textvqa100 import (MEASUREMENT as SHARED_SOURCES, validate_bank, compact_row,
    save_input_bank, save_decoder, compare_replay, prepare_pixels, load_model, save_image,
    case_seed, interrupt_run, MODEL_ID, VISION_ID, write_csv)

DIAGNOSTIC_IDS = tuple("mmstar:" + str(i) for i in range(6))
MEASUREMENT = (*SHARED_SOURCES, "scripts/rq2_staged_mmstar_protocol.py",
    "scripts/prepare_rq2_staged_mmstar.py", "scripts/run_rq2_staged_mmstar1500.py",
    "scripts/summarize_rq2_staged_mmstar1500.py", "scripts/eval_original_mmstar.py",
    "research/protocols/rq2_staged_mmstar1500_v1.json",
    "research/protocols/rq2_staged_mmstar1500_reference_v1.json")
TESTS = ("research.tests.test_rq2_native", "research.tests.test_rq2_staged_protocol",
    "research.tests.test_rq2_staged_runner", "research.tests.test_rq2_staged_native",
    "research.tests.test_rq2_staged_mmstar_protocol", "research.tests.test_rq2_staged_mmstar_runner")
AUXILIARY = tuple(m.replace(".", "/") + ".py" for m in TESTS) + (
    "research/notebooks/kaggle_rq2_staged_mmstar1500.ipynb", "research/RQ2_STAGED_MMSTAR1500.md",
    "research/tests/test_rq2_staged_textvqa_runner.py", "research/tests/test_rq2_random_runner.py",
    "research/tests/test_rq2_frozen_native.py")


def measurement_hashes():
    return source_hashes(extra=MEASUREMENT)


def prerequisite_signature():
    return {**measurement_hashes(), **{p: sha256(REPO / p) for p in AUXILIARY}}


def verify_prerequisites(args):
    p = json.loads(args.prerequisite_report.read_text())
    require(p.get("version") == VERSION and p.get("status") == "passed"
            and p.get("test_modules") == list(TESTS) and p.get("skipped") == 0
            and type(p.get("tests_run")) is int and p["tests_run"] > 26,
            "Require all current native/MMStar staged prerequisites with zero skips")
    require(p.get("source_sha256") == prerequisite_signature()
            and p.get("setup_log_sha256") == sha256(args.setup_log)
            and p.get("runtime_report_sha256") == sha256(args.runtime_report),
            "Sources or runtime changed since prerequisites; rerun them")
    return p


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
    begin = time.perf_counter()
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for case in data["cases"]:
            cid, key = case["case_id"], case["image_sha256"]
            seed = case_seed(case, profile["seed"])
            args.manifest = args.inputs / "manifest.json"
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
                encoding_path = args.output / "encodings" / (cid.replace(":", "-") + ".json")
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
                    row.update(index=case["index"], image_id=case["image_id"], category=case["category"], l2_category=case["l2_category"])
                    if cid in TENSOR_IDS and layers[condition] is not None:
                        row["decoder_tensor"] = save_decoder(args, report, cid, condition, layers[condition])
                    q["arms"][condition] = row
                    write_json(path, q)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    report["progress"]["completed_primary_predictions"] += 1
                    print(f"{cid}: {condition} = {row['response']!r}; correct={row['scoring']['correct']}", flush=True)
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
                                correct_equal=row["scoring"]["correct"] == primary["scoring"]["correct"],
                                policy="Record fresh whole-native variation; never replace the first primary realization"))
                q["status"] = "passed"
                write_json(path, q)
                report["question_sha256"][str(path.relative_to(args.output))] = sha256(path)
                complete.extend(q["arms"].values())
                report["questions"].append(dict(case_id=cid, status="passed"))
                report["progress"]["completed_questions"] = len(report["questions"])
                write_json(args.output / "report.json", report)
                n = len(report["questions"])
                eta = (time.perf_counter() - begin) / n * (len(ids) - n)
                print(f"Completed {n}/{len(ids)}: {cid}; rough remaining {eta / 60:.1f} min", flush=True)
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
        str(args.inputs / "manifest.json"): "inputs/mmstar/manifest.json",
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
    rows = [dict(case_id=q["case_id"], case_status=q["status"], index=q["case"]["index"],
        category=q["case"]["category"], l2_category=q["case"]["l2_category"],
        question=q["case"]["question"], gold=q["case"]["canonical_target"], condition=name,
        response=r["response"], **r["scoring"], seed=r["seed"], random_seed=r["random_seed"],
        input_tokens=r["input_tokens"], after_block16_tokens=r["decoder_tokens"] or 576)
        for q in questions for name, r in q.get("arms", {}).items()]
    if rows:
        write_csv(output / "answers.csv", rows)
    write_json(output / "historical-comparison.json", [{"case_id": q["case_id"], **q["historical_comparison"]}
        for q in questions if "historical_comparison" in q])
    report["summary_coverage"] = dict(complete_questions=len(completed) // 9,
        full_mmstar1500=len(completed) == 13500, policy="Only complete nine-arm questions summarized; all partial arms retained")
    write_json(output / "report.json", report)
    esc = lambda x: html.escape(str(x))
    summary = report.get("summaries", {})
    parts = ['<!doctype html><meta charset="utf-8"><title>RQ2 staged MMStar1500</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:1em}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:.3em}img{width:336px;max-width:100%}article{margin:2em 0;border-top:1px solid #bbb}pre{white-space:pre-wrap}</style>',
        '<h1>Staged MMStar: native ApET and matched pure random</h1><p>Status: ' + esc(report["status"]) +
        '. Complete questions in this export: ' + str(summary.get("questions", 0)) +
        '/1500. Native96→32 /288→96 after block16. Project explicit-choice-v2 scorer.</p>',
        '<p><a href="report.json">Report and comparisons</a> · <a href="answers.csv">Answers</a> · <a href="predictions.jsonl">Predictions</a></p>',
        '<table><tr><th>Condition</th><th>Accuracy</th><th>APR</th><th>Unparsed</th></tr>']
    for name, v in summary.get("conditions", {}).items():
        apr = format(v["APR"], ".1%") if v["APR"] is not None else "n/a"
        parts.append(f'<tr><td>{esc(name)}</td><td>{v["accuracy"]:.1%}</td><td>{apr}</td><td>{len(v["unparsed_ids"])}</td></tr>')
    parts.append('</table>')
    for q in questions:
        c = q["case"]
        parts.append('<article><h2>' + esc(q["case_id"]) + ' · ' + esc(c["category"]) +
            '</h2><pre>' + esc(c["question"]) + '</pre><p>Reference: ' + esc(c["canonical_target"]) +
            ' · Status: ' + esc(q["status"]) + '</p><img loading="lazy" src="images/' + c["image_sha256"] +
            '-canvas.png" alt="Actual model canvas"><table><tr><th>Condition</th><th>Answer</th><th>Correct</th><th>Parse method</th></tr>')
        for name, r in q.get("arms", {}).items():
            parts.append('<tr><td>' + esc(name) + '</td><td>' + esc(r["response"]) + '</td><td>' +
                str(r["scoring"]["correct"]) + '</td><td>' + esc(r["scoring"]["parse_method"]) + '</td></tr>')
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
    parser.add_argument("--run-mmstar", action="store_true", required=True)
    parser.add_argument("--shards", nargs="+", type=int, default=list(range(6)))
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
        report.update(provenance("Staged MMStar1500: all pinned questions; native96→32/288→96 versus matched pure random"))
        report["status"] = "running"
        validate_runtime(report)
        verify_native_reference(report)
        profile, data, reference = validate_inputs(args.inputs / "manifest.json")
        ids = selected_ids(profile, args.shards)
        data = {**data, "cases": [c for c in data["cases"] if c["case_id"] in set(ids)]}
        report.update(shards=args.shards, profile=profile, profile_sha256=sha256(PROFILE), conditions=CONDITIONS,
            reviewed_calibration=load_calibration_reference(), measurement_sources_sha256=measurement_hashes(),
            prerequisite=verify_prerequisites(args))
        copy_evidence(args, report)
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == args.expected_gpus, "Enable expected CUDA GPUs")
        write_json(args.output / "report.json", report)
        print(f"Validated full1500 inputs. Requested {len(ids)} questions / {len(ids) * 9} primary predictions; one model load.", flush=True)
        loaded = load_model(args, report)
        run_panel(args, report, profile, data, reference, loaded)
        require([q["case_id"] for q in report["questions"]] == ids
            and report["progress"]["completed_primary_predictions"] == len(ids) * 9, "Incomplete requested staged MMStar parts")
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
