"""Reproducible MMStar evaluation using the validated HF LLaVA runtime."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from contextlib import nullcontext
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DATASET_ID = "Lin-Chen/MMStar"
DATASET_REVISION = "bc98d668301da7b14f648724866e57302778ab27"
ANSWER_INSTRUCTION = "Answer with only the option letter (A, B, C, or D)."


def load_mmstar_parquet(parquet, cache_dir):
    from datasets import Image, load_dataset

    dataset = load_dataset(
        "parquet", data_files={"val": str(parquet)}, split="val", cache_dir=str(cache_dir),
    )
    # Loading the parquet directly can leave this as a binary column. Declare
    # its image feature explicitly so rows decode into PIL images before use.
    return dataset.cast_column("image", Image())


def validate_images(dataset, selected):
    from PIL import Image

    for row in selected:
        image = dataset[row]["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a decoded PIL image at row {row}, got {type(image).__name__}")
        image.convert("RGB").load()


def select_rows(categories, per_category, seed):
    groups = defaultdict(list)
    for row, category in enumerate(categories):
        groups[category].append(row)
    rng = random.Random(seed)
    selected = []
    for category in sorted(groups):
        if len(groups[category]) < per_category:
            raise ValueError(f"Not enough samples in category {category}")
        selected.extend(rng.sample(groups[category], per_category))
    return sorted(selected)


def prepare_question(question):
    # The image is already supplied once by the LLaVA prompt wrapper.
    cleaned = re.sub(r"<image(?:\s+\d+)?>", "", question).strip()
    return f"{cleaned}\n{ANSWER_INSTRUCTION}"


def parse_answer(response):
    # Conservative extraction: ambiguous/prose answers remain unparsed.
    match = re.fullmatch(
        r"(?:(?:THE\s+)?(?:CORRECT\s+)?ANSWER(?:\s+IS)?\s*[:=]?\s*|OPTION\s+)?"
        r"\(?([A-D])\)?[.!]?", response.strip().upper(),
    )
    return match.group(1) if match else None


def summarize(predictions):
    total = len(predictions)
    correct = sum(row["correct"] for row in predictions)
    categories = defaultdict(list)
    for row in predictions:
        categories[row["category"]].append(row["correct"])
    latencies = sorted(row["generation_seconds"] for row in predictions)
    summary = {
        "evaluated": total, "correct": correct,
        "accuracy": correct / total if total else None,
        "unparsed": sum(row["prediction"] is None for row in predictions),
        "per_category": {key: {"evaluated": len(values), "correct": sum(values),
                                "accuracy": sum(values) / len(values)}
                         for key, values in sorted(categories.items())},
        "median_generation_seconds": statistics.median(latencies) if total else None,
        "total_generation_seconds": round(sum(latencies), 3),
        "mean_generation_seconds": statistics.mean(latencies) if total else None,
        # Nearest-rank percentile; meaningful even for small diagnostic subsets.
        "p95_generation_seconds": latencies[math.ceil(0.95 * total) - 1] if total else None,
        "max_generation_seconds": latencies[-1] if total else None,
    }
    if total and all("generated_tokens" in row for row in predictions):
        tokens = [row["generated_tokens"] for row in predictions]
        summary["total_generated_tokens"] = sum(tokens)
        summary["mean_generated_tokens"] = statistics.mean(tokens)
        summary["median_generated_tokens"] = statistics.median(tokens)
        summary["max_generated_tokens"] = max(tokens)
    peak_memory = {}
    for row in predictions:
        for device, gib in row.get("peak_allocated_gib", {}).items():
            peak_memory[device] = max(peak_memory.get(device, 0), gib)
    summary["peak_allocated_gib"] = peak_memory
    return summary


def save_report(path, report):
    """Replace the JSON atomically; prediction rows are flushed independently."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def update_progress(report, predictions, selected_count, started):
    elapsed = time.perf_counter() - started
    completed = len(predictions)
    report["summary"] = summarize(predictions)
    report["timing"]["scored_wall_seconds"] = round(elapsed, 3)
    report["progress"] = {
        "completed": completed, "selected": selected_count,
        "estimated_remaining_seconds": round(elapsed / completed * (selected_count - completed), 1)
        if completed else None,
        "note": "Rough estimate from completed samples; later categories and answer lengths can differ.",
    }


def run(args, report):
    from smoke_llava import checkpoint
    from llava_runtime import generate_answer, load_llava

    preparation_started = time.perf_counter()
    report["timing"] = {
        "note": "Generation timings synchronize all GPUs and include vision, compression, LLM prefill "
                "and variable-length decode. They exclude CPU image processing and file writes. "
                "Scored wall time includes those operations; it excludes model loading and warmup.",
    }
    # Budget zero reuses existing weights and refuses to download a missing model.
    args.preflight_only = False
    args.download_budget_gib = 0
    model_path = checkpoint(args, report)
    args.dataset_cache.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.dataset_cache).free < 2 * 1024 ** 3:
        raise RuntimeError("Need at least 2 GiB free at the temporary dataset cache")
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    from huggingface_hub import hf_hub_download

    # Download just the pinned ~42 MB parquet file, not the duplicate TSV.
    parquet = hf_hub_download(
        DATASET_ID, "mmstar.parquet", repo_type="dataset", revision=args.dataset_revision,
        cache_dir=str(args.dataset_cache / "hub"),
    )
    dataset = load_mmstar_parquet(parquet, args.dataset_cache / "arrow")
    required = {"index", "image", "question", "answer", "category", "l2_category"}
    if not required.issubset(dataset.column_names):
        raise ValueError(f"Unexpected dataset columns: {dataset.column_names}")
    categories = dataset["category"]
    selected = select_rows(categories, args.per_category, args.seed)
    indices = dataset["index"]
    sample_ids = [int(indices[row]) for row in selected]
    report["dataset"] = {
        "id": DATASET_ID, "revision": args.dataset_revision, "split": "val",
        "parquet_sha256": hashlib.sha256(Path(parquet).read_bytes()).hexdigest(),
        "total_rows": len(dataset), "seed": args.seed, "per_category": args.per_category,
        "selected_rows": selected, "sample_ids": sample_ids,
        "category_counts": dict(Counter(categories[row] for row in selected)),
        "image_feature": str(dataset.features["image"]),
    }
    report["protocol"] = {
        "instruction": ANSWER_INSTRUCTION, "max_new_tokens": args.max_new_tokens,
        "do_sample": False, "precision": "float16", "batch_size": 1,
        "scoring": "Conservative single-letter parsing; unparsed answers count as incorrect.",
        "warmup": "One unscored generation of up to 2 tokens on the first selected image.",
        "note": "Diagnostic scoring, not an official MMStar score or fixed-work speed benchmark.",
    }
    report["compression"] = {
        "enabled": args.apet, "stage": "vision input only" if args.apet else "none",
        "original_tokens": 576, "keep_tokens": args.keep_tokens if args.apet else 576,
        "basis_tokens": args.basis_tokens if args.apet else None,
        "epsilon": 1e-5 if args.apet else None, "merge": args.apet,
        "fps_seed_rule": "seed + sample index, independent generator" if args.apet else None,
        "decoder_pruning": False,
    }
    baseline = baseline_rows = None
    if args.compare_to:
        from compare_mmstar import read_predictions, validate_compatible

        baseline = json.loads(args.compare_to.read_text())
        validate_compatible(baseline, report)
        baseline_rows = read_predictions(baseline, args.compare_to)
    # Catch image decoding problems on CPU before loading the 7B model.
    validate_images(dataset, selected)
    report["dataset"]["validated_images"] = len(selected)
    report["timing"]["preparation_seconds"] = round(time.perf_counter() - preparation_started, 3)
    print(f"Decoded and validated {len(selected)} MMStar images; loading LLaVA.", flush=True)
    model_started = time.perf_counter()
    model, processor = load_llava(model_path, report)
    report["timing"]["model_loading_seconds"] = round(time.perf_counter() - model_started, 3)
    import torch

    report["hardware"] = {
        "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                  "total_memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 3)}
                 for i in range(torch.cuda.device_count())],
        "attention_implementation": "sdpa",
    }
    if args.apet:
        sys.path.insert(0, str(REPO))
        from apet_compression import CompressionConfig
        from apet_compression.adapters.hf_llava import HFInputCompressionAdapter

        context = HFInputCompressionAdapter(model, CompressionConfig(args.keep_tokens, args.basis_tokens))
    else:
        context = nullcontext(None)
    predictions = []
    predictions_path = args.output.with_suffix(".jsonl")
    report["predictions_file"] = str(predictions_path)
    with context as adapter, predictions_path.open("w") as stream:
        first = dataset[selected[0]]
        warmup_started = time.perf_counter()
        generate_answer(model, processor, first["image"], prepare_question(first["question"]), 2,
                        adapter=adapter, compression_seed=args.seed + int(first["index"]))
        report["timing"]["warmup_seconds"] = round(time.perf_counter() - warmup_started, 3)
        if adapter is not None:
            report["compression"]["legacy_check"] = adapter.legacy_check
        scored_started = time.perf_counter()
        update_progress(report, predictions, len(selected), scored_started)
        save_report(args.output, report)
        try:
            for number, row in enumerate(selected, 1):
                sample = dataset[row]
                gold = sample["answer"].strip().upper()
                if gold not in {"A", "B", "C", "D"}:
                    raise ValueError(f"Invalid ground truth for sample {sample['index']}")
                generated = generate_answer(
                    model, processor, sample["image"], prepare_question(sample["question"]),
                    args.max_new_tokens,
                    adapter=adapter, compression_seed=args.seed + int(sample["index"]),
                )
                prediction = parse_answer(generated["response"])
                record = {
                    **generated, "index": int(sample["index"]), "row": row,
                    "original_question": sample["question"], "category": sample["category"],
                    "l2_category": sample["l2_category"], "gold": gold,
                    "prediction": prediction, "correct": prediction == gold,
                }
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                predictions.append(record)
                if number % args.log_every == 0 or number == len(selected):
                    update_progress(report, predictions, len(selected), scored_started)
                    save_report(args.output, report)
                    remaining = report["progress"]["estimated_remaining_seconds"]
                    print(f"[{number}/{len(selected)}] accuracy={report['summary']['accuracy']:.3f} "
                          f"scored_wall={report['timing']['scored_wall_seconds']:.1f}s "
                          f"remaining~{remaining / 60:.1f}min", flush=True)
        finally:
            # Includes completed rows even after a generation failure or Ctrl-C.
            update_progress(report, predictions, len(selected), scored_started)
    if baseline is not None:
        from compare_mmstar import compare_runs

        report["comparison"] = compare_runs(baseline, {**report, "status": "passed"}, baseline_rows,
                                             {row["index"]: row for row in predictions})
        report["comparison"]["baseline_report"] = str(args.compare_to)
    report["status"] = "passed"


def main():
    from smoke_llava import MODEL_ID, MODEL_REVISION

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-hf-cache"))
    parser.add_argument("--dataset-cache", type=Path, default=Path("/tmp/apet-mmstar-cache"))
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=1,
                        help="Print progress and checkpoint the JSON report every N answers")
    parser.add_argument("--apet", action="store_true", help="Enable extracted input-stage ApET")
    parser.add_argument("--keep-tokens", type=int, default=96)
    parser.add_argument("--basis-tokens", type=int, default=10)
    parser.add_argument("--compare-to", type=Path, help="Saved uncompressed baseline JSON report")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        filename = f"mmstar-apet-input-{args.keep_tokens}.json" if args.apet else "mmstar-baseline.json"
        args.output = REPO / "research/results" / filename
    if args.per_category < 1 or args.max_new_tokens < 1 or args.log_every < 1:
        parser.error("--per-category, --max-new-tokens and --log-every must be positive")
    if args.apet and not 1 <= args.basis_tokens < args.keep_tokens <= 576:
        parser.error("Require 1 <= basis-tokens < keep-tokens <= 576")
    if args.compare_to and args.output.resolve() == args.compare_to.resolve():
        parser.error("Use a different output filename; preserve the uncompressed baseline")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "HF LLaVA + extracted input-stage ApET" if args.apet else "HF LLaVA baseline; ApET disabled",
              "executable": sys.executable}
    started = time.perf_counter()
    try:
        if Path(sys.prefix).resolve() != (REPO / ".venv").resolve():
            raise RuntimeError("Launch with the project's .venv/bin/python")
        report["packages"] = {name: importlib.metadata.version(name) for name in
                              ["torch", "transformers", "accelerate", "datasets", "pillow"]}
        report["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        report["lock_sha256"] = hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest()
        run(args, report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "Interrupted; completed prediction rows are preserved, but this run is incomplete."
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
    report.setdefault("timing", {})["total_wall_seconds"] = round(time.perf_counter() - started, 3)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_report(args.output, report)
    # Keep full sample IDs/provenance in the file, rather than flooding notebook output.
    print(json.dumps({key: report[key] for key in
                      ["status", "summary", "timing", "comparison", "error"] if key in report}, indent=2),
          flush=True)
    print(f"Report: {args.output}", flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
