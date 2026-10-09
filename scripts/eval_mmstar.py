"""Small reproducible MMStar baseline using the validated HF LLaVA runtime."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
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
    return {
        "evaluated": total, "correct": correct,
        "accuracy": correct / total if total else None,
        "unparsed": sum(row["prediction"] is None for row in predictions),
        "per_category": {key: {"evaluated": len(values), "correct": sum(values),
                                "accuracy": sum(values) / len(values)}
                         for key, values in sorted(categories.items())},
        "median_generation_seconds": statistics.median(
            row["generation_seconds"] for row in predictions) if total else None,
    }


def run(args, report):
    from smoke_llava import checkpoint
    from llava_runtime import generate_answer, load_llava

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
        "note": "Subset diagnostic, not an official MMStar score or speed benchmark.",
    }
    # Catch image decoding problems on CPU before loading the 7B model.
    validate_images(dataset, selected)
    report["dataset"]["validated_images"] = len(selected)
    print(f"Decoded and validated {len(selected)} MMStar images; loading LLaVA.", flush=True)
    model, processor = load_llava(model_path, report)
    first = dataset[selected[0]]
    generate_answer(model, processor, first["image"], prepare_question(first["question"]), 2)

    predictions = []
    predictions_path = args.output.with_suffix(".jsonl")
    report["predictions_file"] = str(predictions_path)
    with predictions_path.open("w") as stream:
        for number, row in enumerate(selected, 1):
            sample = dataset[row]
            gold = sample["answer"].strip().upper()
            if gold not in {"A", "B", "C", "D"}:
                raise ValueError(f"Invalid ground truth for sample {sample['index']}")
            generated = generate_answer(
                model, processor, sample["image"], prepare_question(sample["question"]),
                args.max_new_tokens,
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
            report["summary"] = summarize(predictions)
            print(f"[{number}/{len(selected)}] index={record['index']} "
                  f"prediction={prediction} gold={gold} correct={record['correct']}", flush=True)
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
    parser.add_argument("--output", type=Path, default=REPO / "research/results/mmstar-baseline.json")
    args = parser.parse_args()
    if args.per_category < 1 or args.max_new_tokens < 1:
        parser.error("--per-category and --max-new-tokens must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "failed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "HF LLaVA baseline; ApET disabled", "executable": sys.executable}
    try:
        if Path(sys.prefix).resolve() != (REPO / ".venv").resolve():
            raise RuntimeError("Launch with the project's .venv/bin/python")
        report["packages"] = {name: importlib.metadata.version(name) for name in
                              ["torch", "transformers", "accelerate", "datasets", "pillow"]}
        report["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        report["lock_sha256"] = hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest()
        run(args, report)
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    serialized = json.dumps(report, indent=2)
    print(serialized, flush=True)
    args.output.write_text(serialized + "\n")
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
