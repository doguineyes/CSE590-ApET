"""Native ApET evaluation on frozen GQA subsets, with resume and saved baselines."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_original_gqa_full as shared
from check_original_env import REPO, provenance, validate_runtime, verify_native_reference
from eval_original_mmstar import atomic_report
from gqa_scoring import INSTRUCTION, metadata as scoring_metadata, score_answer
from prepare_gqa_full import full_questions, QUESTION_COUNT, IMAGE_COUNT
from prepare_gqa_pilot import DATASET_ID, DATASET_REVISION, FILES, image_seed
from prepare_gqa_subsets import STUDY_SCHEMA, SUBSET_SCHEMA, canonical_hash
from smoke_original_apet import (
    CONFIGURATIONS, MODEL_ID, MODEL_REVISION, VISION_ID, VISION_REVISION,
    checkpoint_plan, download, generate_native_answer, load_native_model, validate_trace,
)

RUN_SCHEMA = "original-gqa-fixed-subset-v1"
EXTRA_SOURCES = ["scripts/eval_original_gqa_subset.py", "scripts/prepare_gqa_subsets.py"]


def load_manifest(path):
    from PIL import Image

    manifest = json.loads(path.read_text())
    root = path.parent / "study.json"
    if not root.is_file():
        root = path.parent.parent / "study.json"
    study = json.loads(root.read_text())
    identity = dict(study)
    study_id = identity.pop("id")
    role = manifest["study"]["role"]
    source, selection = manifest["source"], manifest["selection"]
    if (manifest["schema"] != SUBSET_SCHEMA or study["schema"] != STUDY_SCHEMA
            or role not in {"development", "evaluation"} or source != study["source"]
            or source["dataset"] != DATASET_ID or source["revision"] != DATASET_REVISION
            or source["files"] != FILES or source["split"] != "testdev"
            or source["tier"] != "testdev_balanced" or source["total_questions"] != QUESTION_COUNT
            or source["total_images"] != IMAGE_COUNT or canonical_hash(identity) != study_id
            or manifest["study"]["id"] != study_id
            or shared.digest(root) != manifest["study"]["manifest_sha256"]):
        raise ValueError("Expected an unchanged frozen GQA study/subset manifest")
    assignment = study["assignment"]
    dev, evaluation, unused = (set(assignment[name]) for name in ["development", "evaluation", "unused"])
    if (dev & evaluation or dev & unused or evaluation & unused
            or dev | evaluation | unused != set(study["image_sha256"])
            or len(study["image_sha256"]) != IMAGE_COUNT
            or set(study["previously_inspected_pilot_images"]) & evaluation
            or {study["image_sha256"][i] for i in dev} & {study["image_sha256"][i] for i in evaluation}
            or set(study["question_ids"]["development"]) & set(study["question_ids"]["evaluation"])
            or not set(study["question_ids"]["diagnostic"]) <= set(study["question_ids"]["development"])):
        raise ValueError("Development/evaluation leakage or inconsistent study assignment")
    rows = full_questions(manifest["questions"], manifest["images"])
    ids = [row["id"] for row in rows]
    image_ids = sorted(manifest["images"])
    if (rows != manifest["questions"] or ids != selection["question_ids"]
            or ids != study["question_ids"][role] or image_ids != selection["image_ids"]
            or image_ids != assignment[role] or len(rows) != selection["question_count"]
            or len(image_ids) != selection["image_count"]
            or study["counts"][role] != {"images": len(image_ids), "questions": len(rows)}
            or selection["seed"] != study["seed"]
            or dict(Counter(row["imageId"] for row in rows)) != selection["questions_by_image"]
            or any(n != selection["questions_per_image"] for n in selection["questions_by_image"].values())
            or any(canonical_hash(row) != study["question_sha256"][row["id"]] for row in rows)):
        raise ValueError("Subset question IDs, source metadata, image groups or counts changed")
    for image_id, record in manifest["images"].items():
        relative = Path(record["file"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "images":
            raise ValueError("Images must be relative files inside images/")
        file = path.parent / relative
        if shared.digest(file) != record["sha256"] or record["sha256"] != study["image_sha256"][image_id]:
            raise ValueError(f"Image bytes changed: {image_id}")
        with Image.open(file) as image:
            image.load()
            if (list(image.size) != record["image_size"] or
                    hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != record["rgb_pixels_sha256"]):
                raise ValueError(f"Decoded image differs: {image_id}")
        if record["compression_seed"] != image_seed(study["seed"], image_id):
            raise ValueError("Compression seed must depend on image")
    return manifest, root


def contract_for(args, current, manifest):
    contract = shared.run_contract(args, current, manifest)
    contract["schema"] = RUN_SCHEMA
    for name in EXTRA_SOURCES:
        contract["source_sha256"][name] = shared.digest(REPO / name)
    core_sources = {name: contract["source_sha256"][name] for name in current["source_sha256"]}
    core_sources["llava/constants.py"] = contract["source_sha256"]["llava/constants.py"]
    contract["comparison_identity"] = {
        "manifest_sha256": contract["manifest_sha256"], "study": manifest["study"],
        "lock_sha256": contract["lock_sha256"], "model": contract["model"], "vision": contract["vision"],
        "scoring": scoring_metadata(), "inference_source_sha256": core_sources,
        "instruction": INSTRUCTION, "max_new_tokens": args.max_new_tokens,
        "do_sample": False, "use_cache": True, "precision": "float16", "batch_size": 1,
        "baseline_configuration": CONFIGURATIONS["baseline"], "basis_tokens": 10,
    }
    contract["protocol"]["note"] += " Frozen project subset; saved baseline latency is historical, not a same-session timing control."
    return contract


def read_reference(path, contract, manifest):
    """Read and validate baseline evidence without changing any reference file."""
    report = json.loads(path.read_text())
    if (report.get("schema") != RUN_SCHEMA or report.get("status") != "passed"
            or report["contract"]["comparison_identity"] != contract["comparison_identity"]
            or "baseline" not in report["protocol"]["configuration_order"]):
        raise ValueError("Reference must be a completed matching-subset baseline with unchanged inference/scoring/lock/checkpoints")
    prediction_name = report["predictions_file"]
    if Path(prediction_name).name != prediction_name:
        raise ValueError("Reference predictions must be a sibling JSONL file")
    prediction_path = path.parent / prediction_name
    if shared.digest(prediction_path) != report.get("predictions_sha256"):
        raise ValueError("Reference predictions changed or lack a recorded hash")
    rows = [json.loads(line) for line in prediction_path.read_text().splitlines()]
    samples = shared.samples_from(manifest, report["dataset"]["data_directory"])
    if [(row["index"], row["configuration"]) for row in rows] != shared.expected_order(samples, report["protocol"]["configuration_order"]):
        raise ValueError("Reference is incomplete, duplicated or reordered")
    baseline = [row for row in rows if row["configuration"] == "baseline"]
    by_id = {sample["index"]: sample for sample in samples}
    for row in baseline:
        sample = by_id[row["index"]]
        if (any(row.get(key) != value for key, value in sample.items())
                or row["status"] != "passed" or row["seed"] != sample["compression_seed"]
                or row["settings"] != {**CONFIGURATIONS["baseline"], "basis_tokens": 10}
                or not 1 <= row["generated_tokens"] <= contract["protocol"]["max_new_tokens"]
                or not math.isfinite(row["generation_seconds"]) or row["generation_seconds"] < 0
                or any(row.get(key) != value for key, value in score_answer(row["response"], sample["gold"]).items())):
            raise ValueError("Reference baseline settings, samples or scores differ")
        validate_trace("baseline", row["trace"])
    identity = {"report_sha256": shared.digest(path), "predictions_sha256": shared.digest(prediction_path)}
    return [shared.compact(row) for row in baseline], identity, prediction_path


def update_summary(report, predictions, names, reference):
    effective_names = names if not reference else ["baseline", *names]
    shared.update_summary(report, [*reference, *predictions], effective_names)
    report["progress"] = {"completed_predictions": len(predictions),
        "expected_predictions": len(report["dataset"]["question_ids"]) * len(names),
        "saved_reference_predictions": len(reference)}
    report["baseline_origin"] = "saved_reference" if reference else "current_run"


def open_run(args, current):
    manifest, study_path = load_manifest(args.data_manifest)
    contract = contract_for(args, current, manifest)
    data_name, source_name = args.output.stem + "-data", args.output.stem + "-source"
    predictions_path = args.output.with_suffix(".jsonl")
    samples = shared.samples_from(manifest, data_name)
    old = json.loads(args.output.read_text()) if args.resume else None
    reference_path = args.reference_report
    if args.resume and reference_path is None and old.get("reference"):
        reference_path = args.output.parent / old["reference"]["report_file"]
    reference, ref_identity, ref_predictions = [], None, None
    if reference_path is not None:
        if "baseline" in args.configurations:
            raise ValueError("Choose compressed configurations when using a saved baseline")
        reference, ref_identity, ref_predictions = read_reference(reference_path, contract, manifest)
    elif "baseline" not in args.configurations:
        raise ValueError("Include baseline or provide --reference-report")
    contract["reference"] = ref_identity
    if args.resume:
        if old.get("contract") != contract:
            raise ValueError("Resume refused: subset, code, environment, checkpoints, settings, scoring or reference changed")
        copied = args.output.parent / data_name / "manifest.json"
        if shared.digest(copied) != contract["manifest_sha256"]:
            raise ValueError("Saved data manifest differs")
        load_manifest(copied)
        for name, expected in contract["source_sha256"].items():
            if shared.digest(args.output.parent / source_name / name) != expected:
                raise ValueError(f"Saved source differs: {name}")
        predictions = shared.read_completed(predictions_path, samples, args)
        report = old
    else:
        if any(path.exists() for path in [args.output, predictions_path, args.output.parent / data_name, args.output.parent / source_name]):
            raise ValueError("Results exist; use --resume or a new run directory")
        destination = args.output.parent / data_name
        destination.mkdir()
        shutil.copy2(args.data_manifest, destination / "manifest.json")
        shutil.copy2(study_path, destination / "study.json")
        for record in manifest["images"].values():
            target = destination / record["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.data_manifest.parent / record["file"], target)
        for name in contract["source_sha256"]:
            target = args.output.parent / source_name / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)
        report = {**current, "schema": RUN_SCHEMA, "contract": contract,
            "protocol": contract["protocol"], "scoring": contract["scoring"],
            "dataset": {**manifest["source"], **manifest["selection"], "study": manifest["study"],
                "manifest_file": data_name + "/manifest.json", "data_directory": data_name,
                "manifest_sha256": contract["manifest_sha256"]},
            "source_directory": source_name, "predictions_file": predictions_path.name, "sessions": []}
        if reference_path is not None:
            reference_dir = args.output.parent / (args.output.stem + "-reference")
            reference_dir.mkdir()
            shutil.copy2(reference_path, reference_dir / reference_path.name)
            shutil.copy2(ref_predictions, reference_dir / ref_predictions.name)
            source_report = json.loads(reference_path.read_text())
            original_source = reference_path.parent / source_report["source_directory"]
            shutil.copytree(original_source, reference_dir / source_report["source_directory"])
            report["reference"] = {**ref_identity, "report_file": str((reference_dir / reference_path.name).relative_to(args.output.parent)),
                "note": "Saved baseline used for accuracy/APR; reference timings are from its original session."}
        predictions_path.touch(exist_ok=False)
        predictions = []
    report["status"] = "running"
    for key in ["error", "traceback", "current_sample", "current_generation", "predictions_sha256"]:
        report.pop(key, None)
    update_summary(report, predictions, args.configurations, reference)
    atomic_report(args.output, report)
    return report, samples, predictions, reference


def run(args, report, samples, predictions, reference):
    from PIL import Image

    expected = len(samples) * len(args.configurations)
    if len(predictions) == expected:
        report["status"] = "passed"
        return
    session = {"id": len(report["sessions"]) + 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "resumed_from_predictions": len(predictions), "provenance": args.session_provenance}
    report["sessions"].append(session)
    atomic_report(args.output, report)
    plans = [checkpoint_plan(args.model, MODEL_REVISION, args.cache_dir, native=True),
             checkpoint_plan(args.vision_model, VISION_REVISION, args.cache_dir)]
    session["checkpoints"] = {"llava": plans[0], "vision": plans[1]}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    missing = sum(plan["missing_bytes"] for plan in plans)
    if missing and (missing > args.download_budget_gib * 1024 ** 3 or missing + 1024 ** 3 > shutil.disk_usage(args.cache_dir).free):
        raise ValueError("Missing checkpoints exceed budget/free space; allow --download-budget-gib 20 or reuse cache")
    model_path, vision_path = [download(plan, args.cache_dir) for plan in plans]
    tokenizer, model, processor = load_native_model(model_path, vision_path, session)
    initial_count = len(predictions)
    first = samples[initial_count // len(args.configurations)]
    with Image.open(args.output.parent / first["image"]) as source:
        first_image = source.convert("RGB")
    warmup_start = time.perf_counter()
    for name in args.configurations:
        generate_native_answer(tokenizer, model, processor, first_image, first["question"], name,
            seed=first["compression_seed"], max_new_tokens=2, require_nonempty=False)
    session["warmup_seconds"] = round(time.perf_counter() - warmup_start, 3)
    started = time.perf_counter()
    try:
        with args.output.with_suffix(".jsonl").open("a") as stream:
            remaining_in_question = initial_count % len(args.configurations)
            for number in range(initial_count // len(args.configurations), len(samples)):
                sample = samples[number]
                with Image.open(args.output.parent / sample["image"]) as source:
                    image = source.convert("RGB")
                offset = number % len(args.configurations)
                order = args.configurations[offset:] + args.configurations[:offset]
                for name in order[remaining_in_question:]:
                    report["current_sample"] = {"question_id": sample["index"], "image_id": sample["image_id"], "configuration": name}
                    report["current_generation"] = {}
                    generated = generate_native_answer(tokenizer, model, processor, image, sample["question"], name,
                        seed=sample["compression_seed"], max_new_tokens=args.max_new_tokens,
                        require_nonempty=False, record=report["current_generation"])
                    row = {**sample, **generated, **score_answer(generated["response"], sample["gold"]), "session_id": session["id"]}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    predictions.append(shared.compact(row))
                    report.pop("current_generation", None)
                remaining_in_question = 0
                if (number + 1) % args.report_every == 0:
                    update_summary(report, predictions, args.configurations, reference)
                    elapsed = time.perf_counter() - started
                    session["scored_wall_seconds"] = round(elapsed, 3)
                    atomic_report(args.output, report)
                    eta = elapsed / (len(predictions) - initial_count) * (expected - len(predictions))
                    print(f"[{len(predictions)}/{expected}] elapsed={elapsed/60:.1f} min; remaining≈{eta/60:.1f} min", flush=True)
        report.pop("current_sample", None)
        report["status"] = "passed"
    finally:
        session["scored_wall_seconds"] = round(time.perf_counter() - started, 3)
        session["new_predictions"] = len(predictions) - initial_count
        update_summary(report, predictions, args.configurations, reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, help="Completed matching-subset baseline JSON; its sibling JSONL is required")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--vision-model", default=VISION_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-original-hf-cache"))
    parser.add_argument("--download-budget-gib", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--report-every", type=int, default=50)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS)
    args = parser.parse_args()
    if args.configurations is None:
        if args.resume and args.output.exists():
            args.configurations = json.loads(args.output.read_text())["protocol"]["configuration_order"]
        else:
            args.configurations = ([name for name in CONFIGURATIONS if name != "baseline"]
                                   if args.reference_report is not None else list(CONFIGURATIONS))
    if (args.output.suffix != ".json" or args.max_new_tokens < 1 or args.report_every < 1
            or len(set(args.configurations)) != len(args.configurations)
            or not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0):
        parser.error("Use .json output, positive token/report counts, unique configurations and a finite nonnegative budget")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = None
    with args.output.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "Another process owns this report\n")
        try:
            current = provenance("Original ApET frozen GQA subset; project evaluation and image-disjoint sampling")
            validate_runtime(current)
            verify_native_reference(current)
            current["implementation"]["project_owned"].extend(EXTRA_SOURCES)
            args.session_provenance = current
            report, samples, predictions, reference = open_run(args, current)
            run(args, report, samples, predictions, reference)
        except (Exception, KeyboardInterrupt) as error:
            if report is None:
                print(f"Setup/resume refused; existing evidence preserved: {type(error).__name__}: {error}", file=sys.stderr)
                return 1
            report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            report["traceback"] = traceback.format_exc()
        if report["sessions"]:
            report["sessions"][-1]["status"] = report["status"]
        report["predictions_sha256"] = shared.digest(args.output.with_suffix(".jsonl"))
        atomic_report(args.output, report)
        print(json.dumps({"status": report["status"], "progress": report["progress"],
                          "error": report.get("error"), "report": str(args.output)}, indent=2))
        return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
