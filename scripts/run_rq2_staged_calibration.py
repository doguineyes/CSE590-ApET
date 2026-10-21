"""Small native staged ApET / matched random calibration: six questions, nine arms."""
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
from rq2_staged_protocol import (VERSION, PROFILE, CONDITIONS, CASE_IDS, SCHEDULES, REPEAT_IDS,
    PROBE_CASE, TENSOR_IDS, IDENTITY_CONDITION, specification, validate_bundle, summarize)
from run_rq1_input import load_model, save_image, interrupt_run
from run_rq2_random_calibration import prepare_pixels
from run_rq2_input_merge import copy_evidence
from smoke_original_apet import MODEL_ID, VISION_ID
from summarize_rq1_pilot import write_csv

MEASUREMENT = ("scripts/rq2_staged_protocol.py", "scripts/rq2_staged_native.py",
    "scripts/run_rq2_staged_calibration.py", "research/protocols/rq2_staged_calibration_v1.json",
    "scripts/rq2_native.py", "scripts/rq2_frozen_protocol.py", "scripts/rq2_parity_diagnostic.py",
    "scripts/rq2_random_native.py", "scripts/rq2_random_protocol.py", "scripts/run_rq2_random_calibration.py",
    "scripts/run_rq2_input_merge.py", "research/protocols/rq2_input_merge_v2.json",
    "environments/original/pyproject.toml", "environments/original/.python-version", "bootstrap/setup_original_linux.sh")
TESTS = ("research.tests.test_rq2_native", "research.tests.test_rq2_staged_protocol",
         "research.tests.test_rq2_staged_runner", "research.tests.test_rq2_staged_native")
AUXILIARY = tuple(m.replace(".", "/") + ".py" for m in TESTS) + (
    "research/notebooks/kaggle_rq2_staged_calibration.ipynb", "research/RQ2_STAGED_CALIBRATION.md")


def measurement_hashes():
    return source_hashes(extra=MEASUREMENT)


def prerequisite_signature():
    return {**measurement_hashes(), **{name: sha256(REPO / name) for name in AUXILIARY}}


def verify_prerequisites(args):
    proof = json.loads(args.prerequisite_report.read_text())
    require(proof.get("version") == VERSION and proof.get("status") == "passed"
            and proof.get("test_modules") == list(TESTS) and proof.get("skipped") == 0
            and type(proof.get("tests_run")) is int and proof["tests_run"] > 10,
            "Staged prerequisites require all actual native tests with zero skips")
    require(proof.get("source_sha256") == prerequisite_signature()
            and proof.get("setup_log_sha256") == sha256(args.setup_log)
            and proof.get("runtime_report_sha256") == sha256(args.runtime_report),
            "Staged code/runtime changed since prerequisites; rerun them")
    return proof


def check_bank(observations, item):
    full, archive = observations["baseline576"], item["reference"]
    require(full["encoded"]["features"]["sha256"] == archive["projected_sha256"], "Archived full features changed")
    for name, obs in observations.items():
        spec = specification(name)
        require(obs["source_clip"]["sha256"] == archive["clip_sha256"]
                and obs["source_patches"] == full["source_patches"]
                and obs["encoded"]["kept_tokens"] == spec["input_tokens"], "Source/budget changed: " + name)
        if spec["method"] == "native":
            require(obs["input"]["errors"] == item["feature"]["diagnostics"]["input"]["errors"]
                    and obs["input"]["basis_indices"] == item["feature"]["diagnostics"]["input"]["basis_indices"],
                    "Archived input E/FPS changed")
        if spec["method"] == "pure-random":
            require(obs["full_projected"] == full["encoded"]["features"]
                    and not any(obs["pure_input"].values()), "Pure input changed vectors or executed compression")
    require(observations[IDENTITY_CONDITION]["encoded"] == full["encoded"], "Full576 random input differs")
    return {"status": "passed", "archived_sources_and_input_E_FPS_exact": True,
            "pure_input_original_vectors_exact": True, "identity576_input_exact": True}


def compare_replay(first, again):
    fields = ("case_id", "image_sha256", "condition", "seed", "response", "scoring", "generation_tokens")
    # Identity576 intentionally has a distinct condition name and a no-op stage.
    if first["condition"] == "baseline576" and again["condition"] == IDENTITY_CONDITION:
        fields = tuple(k for k in fields if k != "condition")
    require(all(first[k] == again[k] for k in fields)
            and first["generation"]["prompt"] == again["generation"]["prompt"],
            "Fixed/no-op replay changed generated IDs, processed scores or answers")
    return {"status": "passed", "generated_ids_scores_and_answers_exact": True, "row": again}


def save_input_bank(args, report, key, snapshots):
    import torch
    from rq2_staged_native import clone_tree
    # Pure masks share the byte-identical original full576 projected tensor.
    native = {name: clone_tree(s, cpu=True) for name, s in snapshots.items() if not name.startswith("pure-random-")}
    pure = {name: {"mask": s["encoded"][1].detach().cpu(), "rng_after_input": clone_tree(s["rng_after_input"], cpu=True),
                   "capture": s["capture"]} for name, s in snapshots.items() if name.startswith("pure-random-")}
    relative = "features/" + key + "-input.pt"
    torch.save({"version": VERSION, "native": native, "pure": pure,
                "pure_feature_reference": "baseline576.encoded[0]"}, args.output / relative)
    report["feature_sha256"][relative] = sha256(args.output / relative)
    return {"file": relative, "sha256": report["feature_sha256"][relative]}


def save_decoder(args, report, cid, condition, snapshot):
    import torch
    from rq2_staged_native import clone_tree
    relative = "features/" + cid.replace(":", "-") + "-" + condition + "-decoder.pt"
    torch.save({"version": VERSION, "case_id": cid, "snapshot": clone_tree(snapshot, cpu=True)}, args.output / relative)
    report["feature_sha256"][relative] = sha256(args.output / relative)
    return {"file": relative, "sha256": report["feature_sha256"][relative]}


def run_panel(args, report, profile, manifests, records, loaded):
    from PIL import Image
    from rq2_staged_native import capture_input, generate_condition, generate_raw_native
    tokenizer, model, processor = loaded
    require(len(model.model.layers) == 32, "Calibration requires the native32-block LLaVA1.5-7B")
    report.update(expected_case_ids=list(CASE_IDS), questions=[], question_sha256={}, feature_sha256={},
        image_bank={}, progress={"completed_questions": 0, "expected_questions": 6,
                                "completed_primary_predictions": 0, "expected_primary_predictions": 54})
    for name in ("questions", "features", "images", "encodings"):
        (args.output / name).mkdir(exist_ok=True)
    bank, rows = {}, []
    last_use = {r["case"]["image_sha256"]: i for i, r in enumerate(records)}
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for index, item in enumerate(records):
            case, archive = item["case"], item["reference"]
            cid, key, seed = case["case_id"], case["image_sha256"], archive["seed"]
            dataset = cid.split(":")[0]
            args.manifest = args.inputs / dataset / "manifest.json"
            image = save_image(args, manifests[dataset], case)
            path = args.output / "questions" / (cid.replace(":", "-") + ".json")
            q = {"case_id": cid, "case": item["rq2"], "status": "running", "arms": {}, "controls": {}}
            report["active_case"] = cid
            write_json(path, q)
            try:
                if key not in bank:
                    pixels, canvas = prepare_pixels(model, processor, image)
                    Image.fromarray(canvas).save(args.output / "images" / (key + "-canvas.png"))
                    observations, snapshots = {}, {}
                    for condition in (*CONDITIONS, IDENTITY_CONDITION):
                        print(cid + ": capture input " + condition, flush=True)
                        observations[condition], snapshots[condition] = capture_input(model, pixels, seed, condition, key)
                        write_json(args.output / "encodings" / (key + ".json"), observations)
                    checks = check_bank(observations, item)
                    feature = save_input_bank(args, report, key, snapshots)
                    bank[key] = {"pixels": pixels, "snapshots": snapshots, "observations": observations,
                                 "seed": seed, "checks": checks, "feature": feature}
                    report["image_bank"][key] = {"encodings": observations, "checks": checks, "feature": feature}
                cache = bank[key]
                require(cache["seed"] == seed, "Same-image seed changed")
                q["controls"]["input_bank"] = check_bank(cache["observations"], item)
                q["input_bank"] = cache["feature"]
                pixels, snapshots = cache["pixels"], cache["snapshots"]
                decoder = {}
                for condition in CONDITIONS:
                    print(cid + ": answer " + condition, flush=True)
                    row, decoder[condition] = generate_condition(tokenizer, model, processor, image, case, pixels,
                        condition, snapshots[condition], seed)
                    if cid in TENSOR_IDS and decoder[condition] is not None:
                        row["decoder_tensor"] = save_decoder(args, report, cid, condition, decoder[condition])
                    q["arms"][condition] = row
                    write_json(path, q)
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    report["progress"]["completed_primary_predictions"] += 1
                    print(f"{cid}: {condition} = {row['response']!r}; correct={row['scoring']['correct']}", flush=True)
                full = q["arms"]["baseline576"]
                require(all(r["generation"]["prompt"] == full["generation"]["prompt"] for r in q["arms"].values()),
                        "Condition prompts differ")
                identity, _ = generate_condition(tokenizer, model, processor, image, case, pixels,
                    IDENTITY_CONDITION, snapshots[IDENTITY_CONDITION], seed)
                q["controls"]["identity576"] = compare_replay(full, identity)
                if cid in REPEAT_IDS:
                    q["controls"]["fixed_replays"] = {}
                    for condition in CONDITIONS:
                        repeated, _ = generate_condition(tokenizer, model, processor, image, case, pixels,
                            condition, snapshots[condition], seed, decoder_replay=decoder[condition])
                        repeated["control_only"] = True
                        q["controls"]["fixed_replays"][condition] = compare_replay(q["arms"][condition], repeated)
                        write_json(path, q)
                if cid == PROBE_CASE:
                    q["controls"]["cached_decoding_probes"] = []
                    for condition in profile["probe_conditions"]:
                        probe, _ = generate_condition(tokenizer, model, processor, image, case, pixels,
                            condition, snapshots[condition], seed, decoder_replay=decoder[condition], probe=True)
                        q["controls"]["cached_decoding_probes"].append(probe)
                        write_json(path, q)
                if cid == "textvqa:37533":
                    q["controls"]["fresh_raw_native"] = []
                    for first, last in SCHEDULES:
                        condition = f"native-{first}-to-{last}"
                        for repeat in range(2):
                            fresh = generate_raw_native(tokenizer, model, processor, image, case, condition, seed)
                            primary = q["arms"][condition]
                            q["controls"]["fresh_raw_native"].append({"repeat": repeat + 1, "row": fresh,
                                "generated_ids_equal": fresh["generation_tokens"]["generated_token_ids"] == primary["generation_tokens"]["generated_token_ids"],
                                "score_delta": float(fresh["scoring"].get("consensus_score", fresh["scoring"]["correct"])) - float(primary["scoring"].get("consensus_score", primary["scoring"]["correct"])),
                                "policy": "Record fresh original pipeline variation; do not replace the first primary realization"})
                            write_json(path, q)
                q["status"] = "passed"
                write_json(path, q)
                report["question_sha256"][str(path.relative_to(args.output))] = sha256(path)
                rows.extend(q["arms"].values())
                report["questions"].append({"case_id": cid, "status": "passed"})
                report["progress"]["completed_questions"] = len(report["questions"])
                report["summaries"] = summarize(rows)
                write_json(args.output / "report.json", report)
                print("Completed " + cid, flush=True)
                del decoder
                if last_use[key] == index:
                    del bank[key]
                    del cache, snapshots, pixels
            except BaseException as error:
                q.update(status="failed", error=f"{type(error).__name__}: {error}")
                write_json(path, q)
                raise
    report.pop("active_case", None)


def finish_run(output, report, started):
    report.update(wall_seconds=time.perf_counter() - started, finished_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "report.json", report)
    questions = [json.loads(p.read_text()) for p in sorted((output / "questions").glob("*.json"))]
    rows = [{"case_id": q["case_id"], "case_status": q["status"], "question": q["case"]["question"],
             "condition": name, "response": r["response"], "correct": r["scoring"]["correct"],
             "consensus_score": r["scoring"].get("consensus_score"), "input_tokens": r["input_tokens"],
             "decoder_tokens": r["decoder_tokens"], "random_seed": r["random_seed"]}
            for q in questions for name, r in q.get("arms", {}).items()]
    if rows:
        write_csv(output / "answers.csv", rows)
    escape = lambda s: html.escape(str(s))
    page = ['<!doctype html><meta charset="utf-8"><title>RQ2 staged calibration</title>',
        '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:1em}td,th{padding:.3em;border:1px solid #aaa}table{border-collapse:collapse}img{width:336px;max-width:100%}pre{white-space:pre-wrap}</style>',
        '<h1>Native staged ApET and matched pure random</h1><p>Status: ' + escape(report["status"]) +
        '. Six selected questions. Input96→32 and288→96 after block16. No population or speed claim.</p>',
        '<p><a href="report.json">Report</a> · <a href="answers.csv">Answers</a> · <a href="predictions.jsonl">Raw predictions</a></p>',
        '<pre>' + escape(json.dumps(report.get("summaries", {}), indent=2)) + '</pre>']
    for q in questions:
        case = q["case"]
        page.append('<h2>' + escape(q["case_id"]) + '</h2><p>' + escape(case["question"]) +
            '</p><img src="images/' + case["image_sha256"] + '-canvas.png" alt="Actual model canvas"><p>References: ' +
            escape(case["gold_answers"]) + '</p><table><tr><th>Condition</th><th>Answer</th><th>Correct / consensus</th></tr>')
        for name, row in q.get("arms", {}).items():
            page.append('<tr><td>' + escape(name) + '</td><td>' + escape(row["response"]) + '</td><td>' +
                        escape(row["scoring"].get("consensus_score", row["scoring"]["correct"])) + '</td></tr>')
        page.append('</table>')
    (output / "index.html").write_text("\n".join(page), encoding="utf-8")
    write_json(output / "artifact-sha256.json", {str(p.relative_to(output)): sha256(p)
               for p in sorted(output.rglob("*")) if p.is_file()})
    with zipfile.ZipFile(output.with_suffix(".zip"), "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    print("Status:", report["status"], "Download:", output.with_suffix(".zip"), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("inputs", "output", "setup-log", "runtime-report", "prerequisite-report"):
        parser.add_argument("--" + field, type=Path, required=True)
    parser.add_argument("--calibration", action="store_true", required=True)
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
        parser.error("Preserve existing outputs; choose a fresh path")
    if args.inputs == args.output or args.inputs in args.output.parents:
        parser.error("Output must be outside accepted inputs")
    if args.expected_gpus < 1 or not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        parser.error("Invalid GPU count/download budget")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    signal.signal(signal.SIGTERM, interrupt_run)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = {"version": VERSION, "status": "running", "calibration_passed": False,
        "scientific_benchmark_passed": False,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    try:
        report.update(provenance("Native staged96→32/288→96 versus matched pure random; six-question execution calibration"))
        report["status"] = "running"
        validate_runtime(report)
        verify_native_reference(report)
        profile, _, manifests, records = validate_bundle(args.inputs)
        report.update(profile=profile, profile_sha256=sha256(PROFILE), measurement_sources_sha256=measurement_hashes(),
                      conditions=CONDITIONS, prerequisite=verify_prerequisites(args))
        copy_evidence(args, report)
        for name in AUXILIARY:
            destination = args.output / "evidence/sources" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, destination)
        shutil.copy2(args.prerequisite_report, args.output / "evidence/notebook/prerequisites.json")
        report["evidence_sha256"] = {str(p.relative_to(args.output / "evidence")): sha256(p)
            for p in (args.output / "evidence").rglob("*") if p.is_file()}
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == args.expected_gpus, "Enable expected CUDA GPUs")
        write_json(args.output / "report.json", report)
        print("Loading native LLaVA once: six questions, nine primary conditions (54 predictions).", flush=True)
        loaded = load_model(args, report)
        run_panel(args, report, profile, manifests, records, loaded)
        require([q["case_id"] for q in report["questions"]] == list(CASE_IDS)
                and report["progress"]["completed_primary_predictions"] == 54, "Incomplete staged calibration")
        report.update(status="passed", calibration_passed=True)
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
