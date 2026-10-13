"""Same-image GQA diagnostic pilot with native ApET and one shared model load."""

import argparse
from collections import Counter, defaultdict
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
from check_original_env import REPO, provenance, validate_runtime, verify_native_reference, write_report
from eval_mmstar import summarize
from eval_original_mmstar import atomic_report, paired_summary
from gqa_scoring import INSTRUCTION, metadata as scoring_metadata, score_answer
from original_diagnostics import compare_diagnostics, observe_native_values, validate_diagnostics
from prepare_gqa_pilot import DATASET_ID, DATASET_REVISION, FILES, SCHEMA_VERSION, image_seed
from smoke_original_apet import (
    CONFIGURATIONS, MODEL_ID, MODEL_REVISION, VISION_ID, VISION_REVISION,
    checkpoint_plan, download, generate_native_answer, load_native_model,
)


def load_manifest(path):
    from PIL import Image

    manifest = json.loads(path.read_text())
    source = manifest["source"]
    if (manifest["schema"] != SCHEMA_VERSION or source["dataset"] != DATASET_ID
            or source["revision"] != DATASET_REVISION or source["files"] != FILES):
        raise ValueError("Expected a manifest prepared from the pinned GQA Test-Dev sources")
    questions = manifest["questions"]
    ids = [row["id"] for row in questions]
    if len(ids) != len(set(ids)) or ids != manifest["selection"]["question_ids"]:
        raise ValueError("Question IDs differ from the prepared manifest")
    grouped = defaultdict(list)
    for row in questions:
        if not row["question"].strip() or not row["answer"].strip() or "<image" in row["question"]:
            raise ValueError("Invalid or duplicate-image-placeholder GQA question")
        grouped[row["imageId"]].append(row)
    if set(grouped) != set(manifest["images"]) or sorted(grouped) != manifest["selection"]["image_ids"]:
        raise ValueError("Image IDs differ from the prepared manifest")
    if len(grouped) != manifest["selection"]["image_count"]:
        raise ValueError("Image count differs")
    for image_id, rows in grouped.items():
        if len(rows) != manifest["selection"]["questions_per_image"]:
            raise ValueError("Question count per image differs")
        record = manifest["images"][image_id]
        relative = Path(record["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Images must be relative files within the prepared subset")
        file = path.parent / relative
        if hashlib.sha256(file.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Image bytes changed: {image_id}")
        with Image.open(file) as image:
            image.load()
            if (list(image.size) != record["image_size"] or
                    hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != record["rgb_pixels_sha256"]):
                raise ValueError(f"Decoded image differs: {image_id}")
        if record["compression_seed"] != image_seed(manifest["selection"]["seed"], image_id):
            raise ValueError("Compression seed must depend on image, not question")
    return manifest


def prepare_records(args, report):
    manifest = load_manifest(args.data_manifest)
    destination = args.output.parent / (args.output.stem + "-data")
    destination.mkdir()
    shutil.copy2(args.data_manifest, destination / "manifest.json")
    for record in manifest["images"].values():
        path = destination / record["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.data_manifest.parent / record["file"], path)
    report["dataset"] = {
        **manifest["source"], **manifest["selection"],
        "manifest_file": str((destination / "manifest.json").relative_to(args.output.parent)),
        "manifest_sha256": hashlib.sha256(args.data_manifest.read_bytes()).hexdigest(),
        "sample_ids": manifest["selection"]["question_ids"],
        "data_directory": destination.name,
        "note": "Type-diverse same-image diagnostic subset; not an official aggregate GQA benchmark.",
    }
    rows = []
    for source in manifest["questions"]:
        image_id = source["imageId"]
        image = manifest["images"][image_id]
        rows.append({**source, "index": source["id"], "question_id": source["id"], "image_id": image_id,
                     "original_question": source["question"], "question": source["question"] + "\n" + INSTRUCTION,
                     "gold": source["answer"], "category": source["types"]["semantic"],
                     "image": str((destination / image["file"]).relative_to(args.output.parent)),
                     "image_sha256": image["sha256"], "rgb_pixels_sha256": image["rgb_pixels_sha256"],
                     "image_size": image["image_size"], "compression_seed": image["compression_seed"]})
    return rows


def type_summary(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row["types"][field]].append(row["correct"])
    return {key: {"evaluated": len(values), "correct": sum(values), "accuracy": sum(values) / len(values)}
            for key, values in sorted(groups.items())}


def update_summary(report, predictions, configurations):
    groups = {name: [row for row in predictions if row["configuration"] == name] for name in configurations}
    report["summaries"] = {}
    for name, rows in groups.items():
        report["summaries"][name] = {**summarize(rows),
            "literal_correct": sum(row["literal_correct"] for row in rows),
            "empty_response_ids": [row["index"] for row in rows if row["prediction"] is None],
            "by_question_type": {field: type_summary(rows, field) for field in ["semantic", "structural", "detailed"]}}
    report["comparisons"] = {name: paired_summary(groups["baseline"], rows)
                             for name, rows in groups.items() if name != "baseline"}
    report["image_groups"] = {}
    for image_id in report["dataset"]["image_ids"]:
        image_groups = {name: [row for row in rows if row["image_id"] == image_id] for name, rows in groups.items()}
        record = {"summaries": {name: {"evaluated": len(rows), "correct": sum(row["correct"] for row in rows)}
                                 for name, rows in image_groups.items()},
                  "comparisons": {name: paired_summary(image_groups["baseline"], rows)
                                  for name, rows in image_groups.items() if name != "baseline"},
                  "same_image_checks": {}}
        for name, rows in image_groups.items():
            if not rows:
                continue
            first = rows[0]
            record["same_image_checks"][name] = {
                "reference_question_id": first["question_id"], "compression_seed": first["seed"],
                "other_questions": [{"question_id": row["question_id"],
                                     **compare_diagnostics(first["diagnostics"], row["diagnostics"])} for row in rows[1:]],
            }
        report["image_groups"][image_id] = record
    report["progress"] = {"completed_predictions": len(predictions),
                          "expected_predictions": len(report["dataset"]["sample_ids"]) * len(configurations)}


def generate_observed(tokenizer, model, processor, image, sample, name, current=None, max_new_tokens=16):
    with observe_native_values() as diagnostics:
        if current is not None:
            current["diagnostics"] = diagnostics
        generated = generate_native_answer(tokenizer, model, processor, image, sample["question"], name,
                                           seed=sample["compression_seed"], max_new_tokens=max_new_tokens,
                                           require_nonempty=False,
                                           record=current.setdefault("generation", {}) if current is not None else None)
    validate_diagnostics(CONFIGURATIONS[name], diagnostics)
    return {**generated, "diagnostics": diagnostics}


def run(args, report):
    from PIL import Image

    report["protocol"] = {
        "instruction": INSTRUCTION, "max_new_tokens": args.max_new_tokens, "min_new_tokens": None,
        "do_sample": False, "use_cache": True, "batch_size": 1, "precision": "float16",
        "basis_tokens": 10, "epsilon": 1e-5,
        "configurations": {name: CONFIGURATIONS[name] for name in args.configurations},
        "seed_rule": "SHA256(gqa:selection_seed:image_id), first 8 bytes modulo 2^31; reset native RNG each question/configuration",
        "order": "Group by image; prepared question order; rotate configuration order by question ordinal.",
        "observation": "Read-only Python return tracing of native errors/FPS/selection/merge locals. "
                       "No compression reimplementation and no native source edits.",
        "token_identity": "Indices address the stream entering each stage. Basis slots contain substituted FPS features; "
                          "decoder streams may already contain merged features. Indices are not automatically original patch IDs.",
        "limitations": "No answer-contribution interventions yet. Raw error magnitudes at different feature dimensions/layers "
                       "are not directly comparable. Same-image decoder variations may reflect numerical/kernel behavior; "
                       "image-before-question causal states cannot attend to later question tokens.",
    }
    report["scoring"] = scoring_metadata()
    samples = prepare_records(args, report)
    predictions_path = args.output.with_suffix(".jsonl")
    report["predictions_file"] = predictions_path.name
    print(f"Validated {len(samples)} questions on {report['dataset']['image_count']} images; loading native LLaVA once.", flush=True)
    plans = [checkpoint_plan(args.model, MODEL_REVISION, args.cache_dir, native=True),
             checkpoint_plan(args.vision_model, VISION_REVISION, args.cache_dir)]
    report["checkpoints"] = {"llava": plans[0], "vision": plans[1]}
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    missing = sum(plan["missing_bytes"] for plan in plans)
    free = shutil.disk_usage(args.cache_dir).free
    report["storage"] = {"missing_download_gib": missing / 1024 ** 3, "filesystem_free_gib": free / 1024 ** 3,
                         "download_budget_gib": args.download_budget_gib}
    if missing and (missing > args.download_budget_gib * 1024 ** 3 or missing + 1024 ** 3 > free):
        raise ValueError("Checkpoints missing: provide a download budget or reuse the live MMStar cache")
    model_path, vision_path = [download(plan, args.cache_dir) for plan in plans]
    tokenizer, model, processor = load_native_model(model_path, vision_path, report)
    first = samples[0]
    with Image.open(args.output.parent / first["image"]) as source:
        first_image = source.convert("RGB")
    warmup_start = time.perf_counter()
    for name in args.configurations:
        generate_native_answer(tokenizer, model, processor, first_image, first["question"], name,
                               seed=first["compression_seed"], max_new_tokens=2, require_nonempty=False)
    report["timing"] = {"warmup_seconds": round(time.perf_counter() - warmup_start, 3),
                        "note": "Observer-enabled generation includes CPU logging/synchronization overhead; not a speed benchmark. "
                                "Model loading, warmups and logging parity controls are separate from scored wall time."}
    report["observer_parity"] = {}
    parity_start = time.perf_counter()
    for name in args.configurations:
        before = generate_native_answer(tokenizer, model, processor, first_image, first["question"], name,
                                        seed=first["compression_seed"], max_new_tokens=args.max_new_tokens, require_nonempty=False)
        after = generate_observed(tokenizer, model, processor, first_image, first, name,
                                  max_new_tokens=args.max_new_tokens)
        matched = all(before[key] == after[key] for key in ["response", "generated_tokens", "trace"])
        report["observer_parity"][name] = {"matched": matched, "without_logging": before["response"],
                                          "with_logging": after["response"], "question_id": first["question_id"]}
        if not matched:
            raise RuntimeError(f"Observed/unobserved answer or trace differs for {name}; inspect observer parity before analysis")
    report["timing"]["observer_parity_seconds"] = round(time.perf_counter() - parity_start, 3)
    predictions = []
    started = time.perf_counter()
    report["status"] = "running"
    atomic_report(args.output, report)
    try:
        with predictions_path.open("x") as stream:
            for number, sample in enumerate(samples):
                with Image.open(args.output.parent / sample["image"]) as source:
                    image = source.convert("RGB")
                offset = number % len(args.configurations)
                order = args.configurations[offset:] + args.configurations[:offset]
                for name in order:
                    report["current_sample"] = {"question_id": sample["question_id"], "image_id": sample["image_id"], "configuration": name}
                    report["current_observation"] = {}
                    generated = generate_observed(tokenizer, model, processor, image, sample, name,
                                                  current=report["current_observation"], max_new_tokens=args.max_new_tokens)
                    row = {**sample, **generated, **score_answer(generated["response"], sample["gold"])}
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    predictions.append(row)
                    report.pop("current_observation", None)
                    update_summary(report, predictions, args.configurations)
                    atomic_report(args.output, report)
                print(f"[{number + 1}/{len(samples)}] image={sample['image_id']} question={sample['question_id']} "
                      + " | ".join(f"{name}={report['summaries'][name]['correct']}/{report['summaries'][name]['evaluated']}"
                                   for name in args.configurations), flush=True)
        if len(predictions) != len(samples) * len(args.configurations):
            raise RuntimeError("Incomplete prediction count")
        report.pop("current_sample", None)
        report["status"] = "passed"
    finally:
        update_summary(report, predictions, args.configurations)
        report["timing"]["scored_wall_seconds"] = round(time.perf_counter() - started, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--vision-model", default=VISION_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-original-hf-cache"))
    parser.add_argument("--download-budget-gib", type=float, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=list(CONFIGURATIONS))
    args = parser.parse_args()
    if args.max_new_tokens < 1 or not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        parser.error("Use a positive token maximum and a finite nonnegative download budget")
    if "baseline" not in args.configurations or len(set(args.configurations)) != len(args.configurations):
        parser.error("Include baseline exactly once and use unique configurations")
    if args.output.exists() or args.output.with_suffix(".jsonl").exists():
        parser.error("Results already exist; choose a new output name")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = provenance("Original ApET same-image GQA diagnostics; no answer-contribution interventions yet")
    report["project_source_sha256"] = {name: hashlib.sha256((REPO / "scripts" / name).read_bytes()).hexdigest()
                                       for name in ["eval_original_gqa.py", "prepare_gqa_pilot.py", "gqa_scoring.py", "original_diagnostics.py",
                                                    "eval_original_mmstar.py", "eval_mmstar.py"]}
    try:
        validate_runtime(report)
        verify_native_reference(report)
        report["implementation"]["project_owned"].extend(["scripts/" + name for name in
                                                          ["eval_original_gqa.py", "prepare_gqa_pilot.py", "gqa_scoring.py", "original_diagnostics.py"]])
        run(args, report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "Interrupted; completed predictions and available observations are preserved."
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    write_report(report, args.output)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
