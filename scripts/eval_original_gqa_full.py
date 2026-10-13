"""Resumable full balanced GQA Test-Dev evaluation using unchanged native ApET."""

import argparse
from collections import Counter, defaultdict
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
from check_original_env import REPO, provenance, validate_runtime, verify_native_reference
from eval_mmstar import summarize
from eval_original_mmstar import atomic_report, paired_summary
from gqa_scoring import INSTRUCTION, metadata as scoring_metadata, score_answer
from prepare_gqa_full import SCHEMA_VERSION, QUESTION_COUNT, IMAGE_COUNT, full_questions
from prepare_gqa_pilot import DATASET_ID, DATASET_REVISION, FILES, image_seed
from smoke_original_apet import (
    CONFIGURATIONS, MODEL_ID, MODEL_REVISION, VISION_ID, VISION_REVISION,
    checkpoint_plan, download, generate_native_answer, load_native_model, validate_trace,
)

RUN_SCHEMA = "original-gqa-testdev-full-v1"
PROJECT_FILES = ["scripts/eval_original_gqa_full.py", "scripts/prepare_gqa_full.py",
    "scripts/prepare_gqa_pilot.py", "scripts/gqa_scoring.py", "scripts/eval_original_mmstar.py",
    "scripts/eval_mmstar.py", "scripts/mmstar_scoring.py", "scripts/smoke_original_apet.py",
    "scripts/check_original_env.py", "llava/constants.py",
    "environments/original/uv.lock", "environments/original/pyproject.toml",
    "environments/original/.python-version", "research/provenance/original_llava_reference.json"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest(path):
    from PIL import Image

    manifest = json.loads(path.read_text())
    source, selection = manifest["source"], manifest["selection"]
    if (manifest["schema"] != SCHEMA_VERSION or source["dataset"] != DATASET_ID
            or source["revision"] != DATASET_REVISION or source["files"] != FILES
            or source["split"] != "testdev" or source["tier"] != "testdev_balanced"
            or source["total_questions"] != QUESTION_COUNT or source["total_images"] != IMAGE_COUNT):
        raise ValueError("Expected the pinned full balanced GQA Test-Dev manifest")
    rows = full_questions(manifest["questions"], manifest["images"])
    if (len(rows) != QUESTION_COUNT or len(manifest["images"]) != IMAGE_COUNT
            or rows != manifest["questions"]
            or [row["id"] for row in rows] != selection["question_ids"]
            or sorted(manifest["images"]) != selection["image_ids"]
            or selection["question_count"] != QUESTION_COUNT or selection["image_count"] != IMAGE_COUNT
            or dict(Counter(row["imageId"] for row in rows)) != selection["questions_by_image"]):
        raise ValueError("Full dataset IDs, order, or counts differ")
    for image_id, record in manifest["images"].items():
        relative = Path(record["file"])
        if relative.is_absolute() or ".." in relative.parts or not relative.parts or relative.parts[0] != "images":
            raise ValueError("Images must be relative files inside images/")
        file = path.parent / relative
        if digest(file) != record["sha256"]:
            raise ValueError(f"Image bytes changed: {image_id}")
        with Image.open(file) as image:
            image.load()
            if (list(image.size) != record["image_size"] or
                    hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != record["rgb_pixels_sha256"]):
                raise ValueError(f"Decoded image differs: {image_id}")
        if record["compression_seed"] != image_seed(selection["seed"], image_id):
            raise ValueError("Seed must depend on image, not question")
    return manifest


def samples_from(manifest, data_directory):
    rows = []
    for source in manifest["questions"]:
        image = manifest["images"][source["imageId"]]
        rows.append({"index": source["id"], "question_id": source["id"], "image_id": source["imageId"],
            "original_question": source["question"], "question": source["question"] + "\n" + INSTRUCTION,
            "gold": source["answer"], "category": source["types"]["semantic"], "types": source["types"],
            "image": str(Path(data_directory) / image["file"]), "image_sha256": image["sha256"],
            "compression_seed": image["compression_seed"]})
    return rows


def protocol(args):
    return {"instruction": INSTRUCTION, "max_new_tokens": args.max_new_tokens, "min_new_tokens": None,
        "do_sample": False, "use_cache": True, "batch_size": 1, "precision": "float16",
        "basis_tokens": 10, "epsilon": 1e-5,
        "configurations": {name: CONFIGURATIONS[name] for name in args.configurations},
        "configuration_order": args.configurations,
        "seed_rule": "SHA256(gqa:selection_seed:image_id), first 8 bytes modulo 2^31; reset RNG per generation",
        "order": "Sorted image ID then question ID; rotate configuration order by global question ordinal.",
        "observation": "Native token-count hooks only; no per-token return tracing or interventions.",
        "warmup": "Unscored max-2-token generation per configuration at each model load.",
        "note": "Question-weighted project short-answer accuracy and APR. Not the official full GQA metric suite. "
                "Generation timing includes vision/compression/prefill/decode and token-count hooks; "
                "output lengths vary. Resumed sessions have separate wall times and device metadata."}


def source_hashes(current):
    names = set(PROJECT_FILES) | set(current["source_sha256"])
    return {name: digest(REPO / name) for name in sorted(names)}


def run_contract(args, current, manifest):
    # Local model paths lack immutable HF revisions: fingerprint actual weights/metadata once
    # per invocation so a changed attachment cannot silently join an existing result.
    def model_identity(model, revision):
        local = Path(model).expanduser()
        if not local.is_dir():
            return {"model": model, "revision": revision}
        files = [file for file in sorted(local.iterdir()) if file.is_file() and
                 (file.suffix in {".json", ".safetensors", ".bin"} or file.name == "tokenizer.model")]
        hashes = {}
        for file in files:
            hasher = hashlib.sha256()
            with file.open("rb") as stream:
                for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
                    hasher.update(block)
            hashes[file.name] = hasher.hexdigest()
        return {"source": "local", "file_sha256": hashes}
    return {"schema": RUN_SCHEMA, "manifest_sha256": digest(args.data_manifest),
        "protocol": protocol(args), "scoring": scoring_metadata(),
        "lock_sha256": current["lock_sha256"], "source_sha256": source_hashes(current),
        "model": model_identity(args.model, MODEL_REVISION),
        "vision": model_identity(args.vision_model, VISION_REVISION)}


def compact(row):
    keys = ["index", "image_id", "configuration", "gold", "category", "types", "prediction", "correct",
            "literal_correct", "generation_seconds", "generated_tokens", "peak_allocated_gib"]
    return {key: row[key] for key in keys}


def expected_order(samples, names):
    return [(sample["index"], name) for number, sample in enumerate(samples)
            for name in names[number % len(names):] + names[:number % len(names)]]


def read_completed(path, samples, args):
    """Validate a complete prefix; preserve a torn final write before removing it."""
    by_id = {row["index"]: row for row in samples}
    expected = expected_order(samples, args.configurations)
    predictions, valid_end, torn, needs_newline = [], 0, None, False
    with path.open("rb") as stream:
        while raw := stream.readline():
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeDecodeError):
                if not raw.endswith(b"\n") and not stream.read(1):
                    torn = raw
                    break
                raise ValueError("Corrupt complete JSONL line; preserve results and inspect before resuming")
            number = len(predictions)
            if number >= len(expected) or (row["index"], row["configuration"]) != expected[number]:
                raise ValueError("Duplicate, reordered, or foreign prediction; expected a completed prefix")
            sample = by_id[row["index"]]
            if any(row.get(key) != value for key, value in sample.items()):
                raise ValueError(f"Saved sample differs: {row['index']}")
            if (row.get("seed") != sample["compression_seed"] or row.get("status") != "passed"
                    or row.get("settings") != {**CONFIGURATIONS[row["configuration"]], "basis_tokens": 10}
                    or not 1 <= row["generated_tokens"] <= args.max_new_tokens
                    or not math.isfinite(row["generation_seconds"]) or row["generation_seconds"] < 0):
                raise ValueError("Saved generation settings/status differ")
            validate_trace(row["configuration"], row["trace"])
            if any(row.get(key) != value for key, value in score_answer(row["response"], sample["gold"]).items()):
                raise ValueError("Saved scoring differs; rescore separately instead of mixing policies")
            predictions.append(compact(row))
            valid_end = stream.tell()
            needs_newline = not raw.endswith(b"\n")
    if torn is not None:
        backup = path.with_name(path.name + ".partial-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        backup.write_bytes(torn)
        with path.open("r+b") as stream:
            stream.truncate(valid_end)
        print(f"Preserved interrupted final write: {backup.name}", flush=True)
    elif needs_newline:
        with path.open("ab") as stream:
            stream.write(b"\n")
    return predictions


def update_summary(report, predictions, names):
    grouped = {name: [] for name in names}
    images = defaultdict(lambda: {name: [] for name in names})
    typed = {name: {field: defaultdict(list) for field in ["semantic", "structural", "detailed"]} for name in names}
    for row in predictions:
        name = row["configuration"]
        grouped[name].append(row)
        images[row["image_id"]][name].append(row)
        for field, groups in typed[name].items():
            groups[row["types"][field]].append(row)
    def counts(rows):
        return {"evaluated": len(rows), "correct": sum(row["correct"] for row in rows),
                "accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else None}
    report["summaries"] = {name: {**summarize(rows),
        "literal_correct": sum(row["literal_correct"] for row in rows),
        "empty_response_ids": [row["index"] for row in rows if row["prediction"] is None],
        "by_question_type": {field: {kind: counts(values) for kind, values in sorted(groups.items())}
                             for field, groups in typed[name].items()}} for name, rows in grouped.items()}
    report["comparisons"] = {name: {**paired_summary(grouped["baseline"], rows),
        "by_question_type": {field: {kind: paired_summary(typed["baseline"][field].get(kind, []), values, False)
                                     for kind, values in sorted(groups.items())}
                             for field, groups in typed[name].items()}}
        for name, rows in grouped.items() if name != "baseline"}
    report["image_groups"] = {image_id: {"summaries": {name: counts(rows) for name, rows in groups.items()},
        "comparisons": {name: paired_summary(groups["baseline"], rows, False)
                        for name, rows in groups.items() if name != "baseline"}}
        for image_id, groups in sorted(images.items())}
    report["progress"] = {"completed_predictions": len(predictions),
        "expected_predictions": len(report["dataset"]["question_ids"]) * len(names)}


def open_run(args, current):
    manifest = load_manifest(args.data_manifest)
    contract = run_contract(args, current, manifest)
    data_name = args.output.stem + "-data"
    samples = samples_from(manifest, data_name)
    predictions_path = args.output.with_suffix(".jsonl")
    if args.resume:
        report = json.loads(args.output.read_text())
        if report.get("contract") != contract:
            raise ValueError("Resume refused: data, native/project code, lock, checkpoints, scoring or settings changed")
        if digest(args.output.parent / data_name / "manifest.json") != contract["manifest_sha256"]:
            raise ValueError("Saved data manifest differs")
        load_manifest(args.output.parent / data_name / "manifest.json")
        for name, expected in contract["source_sha256"].items():
            if digest(args.output.parent / "source" / name) != expected:
                raise ValueError(f"Saved source snapshot differs: {name}")
        predictions = read_completed(predictions_path, samples, args)
    else:
        if args.output.exists() or predictions_path.exists() or (args.output.parent / data_name).exists():
            raise ValueError("Results exist; use --resume or a new run directory")
        destination = args.output.parent / data_name
        destination.mkdir()
        shutil.copy2(args.data_manifest, destination / "manifest.json")
        for record in manifest["images"].values():
            target = destination / record["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(args.data_manifest.parent / record["file"], target)
        for name in contract["source_sha256"]:
            target = args.output.parent / "source" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, target)
        report = {**current, "schema": RUN_SCHEMA, "contract": contract,
            "protocol": contract["protocol"], "scoring": contract["scoring"],
            "dataset": {**manifest["source"], **manifest["selection"],
                "manifest_file": data_name + "/manifest.json", "manifest_sha256": contract["manifest_sha256"]},
            "predictions_file": predictions_path.name, "sessions": []}
        predictions_path.touch(exist_ok=False)
        predictions = []
    report["status"] = "running"
    for key in ["error", "traceback", "current_sample", "current_generation"]:
        report.pop(key, None)
    update_summary(report, predictions, args.configurations)
    atomic_report(args.output, report)
    return report, samples, predictions


def run(args, report, samples, predictions):
    from PIL import Image

    expected = len(samples) * len(args.configurations)
    if len(predictions) == expected:
        report["status"] = "passed"
        return
    session = {"id": len(report["sessions"]) + 1, "started_utc": datetime.now(timezone.utc).isoformat(),
        "resumed_from_predictions": len(predictions), "provenance": args.session_provenance}
    report["sessions"].append(session)
    # Write the session provenance before any download/loading failures.
    atomic_report(args.output, report)
    plans = [checkpoint_plan(args.model, MODEL_REVISION, args.cache_dir, native=True),
             checkpoint_plan(args.vision_model, VISION_REVISION, args.cache_dir)]
    session["checkpoints"] = {"llava": plans[0], "vision": plans[1]}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    missing = sum(plan["missing_bytes"] for plan in plans)
    free = shutil.disk_usage(args.cache_dir).free
    if missing and (missing > args.download_budget_gib * 1024 ** 3 or missing + 1024 ** 3 > free):
        raise ValueError("Missing checkpoints exceed budget/free space; allow --download-budget-gib 20 or reuse cache")
    model_path, vision_path = [download(plan, args.cache_dir) for plan in plans]
    tokenizer, model, processor = load_native_model(model_path, vision_path, session)
    first = samples[len(predictions) // len(args.configurations)]
    with Image.open(args.output.parent / first["image"]) as source:
        first_image = source.convert("RGB")
    start = time.perf_counter()
    for name in args.configurations:
        generate_native_answer(tokenizer, model, processor, first_image, first["question"], name,
            seed=first["compression_seed"], max_new_tokens=2, require_nonempty=False)
    session["warmup_seconds"] = round(time.perf_counter() - start, 3)
    initial_count = len(predictions)
    started = time.perf_counter()
    session["status"] = "running"
    try:
        with args.output.with_suffix(".jsonl").open("a") as stream:
            # A resume may begin in the middle of a question's configuration sequence.
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
                    predictions.append(compact(row))
                    report.pop("current_generation", None)
                remaining_in_question = 0
                if (number + 1) % args.report_every == 0:
                    update_summary(report, predictions, args.configurations)
                    session["scored_wall_seconds"] = round(time.perf_counter() - started, 3)
                    atomic_report(args.output, report)
                    elapsed = time.perf_counter() - started
                    eta = elapsed / (len(predictions) - initial_count) * (expected - len(predictions))
                    print(f"[{len(predictions)}/{expected}] elapsed={elapsed / 60:.1f} min; remaining≈{eta / 60:.1f} min", flush=True)
        report.pop("current_sample", None)
        report["status"] = "passed"
        session["status"] = "passed"
    finally:
        session["scored_wall_seconds"] = round(time.perf_counter() - started, 3)
        session["new_predictions"] = len(predictions) - initial_count
        update_summary(report, predictions, args.configurations)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--vision-model", default=VISION_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-original-hf-cache"))
    parser.add_argument("--download-budget-gib", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--report-every", type=int, default=100, help="Refresh summary after this many questions; JSONL flushes each prediction")
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    args = parser.parse_args()
    if (args.output.suffix != ".json" or args.max_new_tokens < 1 or args.report_every < 1
            or not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0):
        parser.error("Use a .json output, positive token/report counts and a finite nonnegative budget")
    if "baseline" not in args.configurations or len(set(args.configurations)) != len(args.configurations):
        parser.error("Include baseline once and use unique configurations")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = None
    with args.output.with_suffix(".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.exit(1, "Another process owns this report; do not run concurrently\n")
        try:
            current = provenance("Original ApET full balanced GQA Test-Dev; project evaluation code")
            validate_runtime(current)
            verify_native_reference(current)
            current["implementation"]["project_owned"].extend(PROJECT_FILES[:4])
            args.session_provenance = current
            report, samples, predictions = open_run(args, current)
            run(args, report, samples, predictions)
        except (Exception, KeyboardInterrupt) as error:
            if report is None:
                print(f"Setup/resume refused; existing results preserved: {type(error).__name__}: {error}", file=sys.stderr)
                return 1
            report["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            report["error"] = f"{type(error).__name__}: {error}"
            report["traceback"] = traceback.format_exc()
            if report["sessions"]:
                report["sessions"][-1]["status"] = report["status"]
        atomic_report(args.output, report)
        print(json.dumps({"status": report["status"], "progress": report["progress"],
                          "error": report.get("error"), "report": str(args.output)}, indent=2))
        return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
