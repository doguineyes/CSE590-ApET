"""Paired MMStar pilot using original ApET, with one model load per invocation."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_original_env import REPO, provenance, validate_runtime, write_report
from eval_mmstar import DATASET_ID, DATASET_REVISION, ANSWER_INSTRUCTION, parse_answer, prepare_question, select_rows, summarize
from smoke_original_apet import (
    CONFIGURATIONS, MODEL_ID, MODEL_REVISION, VISION_ID, VISION_REVISION,
    checkpoint_plan, download, generate_native_answer, load_native_model,
)

PARQUET_SHA256 = "29afd74b0134cfab083a8909b5358577ab18fd41c1e612031577cfb3635531c2"


def atomic_report(output, report):
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(output)


def paired_summary(baseline, current, include_categories=True):
    before = {row["index"]: row for row in baseline}
    after = {row["index"]: row for row in current}
    if len(before) != len(baseline) or len(after) != len(current):
        raise ValueError("Duplicate sample IDs within a configuration")
    ids = sorted(before.keys() & after.keys())
    for index in ids:
        if (before[index]["gold"], before[index]["category"]) != (after[index]["gold"], after[index]["category"]):
            raise ValueError(f"Paired labels/category differ for sample {index}")
    both = [index for index in ids if before[index]["correct"] and after[index]["correct"]]
    lost = [index for index in ids if before[index]["correct"] and not after[index]["correct"]]
    gained = [index for index in ids if not before[index]["correct"] and after[index]["correct"]]
    neither = [index for index in ids if not before[index]["correct"] and not after[index]["correct"]]
    denominator = len(both) + len(lost)
    return {
        "paired_samples": len(ids), "baseline_correct_in_pairs": denominator,
        "both_correct": len(both), "apr": len(both) / denominator if denominator else None,
        "both_correct_ids": both, "lost_correct_ids": lost, "gained_correct_ids": gained,
        "both_wrong_ids": neither,
        "per_category": {
            category: paired_summary(
                [before[index] for index in ids if before[index]["category"] == category],
                [after[index] for index in ids if after[index]["category"] == category],
                include_categories=False,
            ) for category in sorted({before[index]["category"] for index in ids})
        } if include_categories else {},
        "note": "APR uses baseline-correct paired samples only; null means a zero denominator. "
                "Partial runs compare only completed pairs.",
    }


def prepare_samples(args, report):
    import pyarrow.parquet as pq
    from PIL import Image
    from huggingface_hub import hf_hub_download

    parquet = Path(hf_hub_download(
        DATASET_ID, "mmstar.parquet", repo_type="dataset", revision=DATASET_REVISION,
        cache_dir=str(args.dataset_cache),
    ))
    digest = hashlib.sha256(parquet.read_bytes()).hexdigest()
    if digest != PARQUET_SHA256:
        raise ValueError(f"Pinned MMStar parquet hash differs: {digest}")
    rows = pq.read_table(parquet).to_pylist()
    required = {"index", "image", "question", "answer", "category", "l2_category"}
    if not rows or any(not required.issubset(row) for row in rows):
        raise ValueError("Unexpected MMStar parquet schema")
    if len({int(row["index"]) for row in rows}) != len(rows):
        raise ValueError("Duplicate MMStar sample IDs")
    selected = select_rows([row["category"] for row in rows], args.per_category, args.seed)
    image_dir = args.output.parent / (args.output.stem + "-images")
    image_dir.mkdir(parents=True)
    samples = []
    for row_number in selected:
        row = rows[row_number]
        gold = row["answer"].strip().upper()
        if gold not in {"A", "B", "C", "D"}:
            raise ValueError(f"Invalid answer for sample {row['index']}")
        raw = row["image"]
        if isinstance(raw, dict):
            raw = raw.get("bytes")
        if not isinstance(raw, bytes) or not raw:
            raise ValueError(f"Expected embedded image bytes for sample {row['index']}")
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            extension = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}.get(image.format, ".img")
            size = list(image.size)
            pixels = hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()
        path = image_dir / (str(int(row["index"])) + extension)
        path.write_bytes(raw)
        samples.append({
            "index": int(row["index"]), "row": row_number, "original_question": row["question"],
            "question": prepare_question(row["question"]), "gold": gold,
            "category": row["category"], "l2_category": row["l2_category"],
            "image": str(path.relative_to(args.output.parent)), "image_size": size,
            "image_sha256": hashlib.sha256(raw).hexdigest(), "rgb_pixels_sha256": pixels,
        })
    report["dataset"] = {
        "id": DATASET_ID, "revision": DATASET_REVISION, "split": "val",
        "parquet_sha256": digest, "total_rows": len(rows), "seed": args.seed,
        "per_category": args.per_category, "sample_ids": [row["index"] for row in samples],
        "selected_rows": selected, "category_counts": dict(Counter(row["category"] for row in samples)),
        "image_directory": image_dir.name, "pyarrow": importlib.metadata.version("pyarrow"),
    }
    return samples


def update_summary(report, predictions, configurations):
    groups = {name: [row for row in predictions if row["configuration"] == name] for name in configurations}
    report["summaries"] = {name: summarize(rows) for name, rows in groups.items()}
    report["comparisons"] = {name: paired_summary(groups["baseline"], rows)
                             for name, rows in groups.items() if name != "baseline"}
    report["progress"] = {"completed_predictions": len(predictions),
                          "expected_predictions": len(report["dataset"]["sample_ids"]) * len(configurations)}


def run(args, report):
    from PIL import Image

    report["protocol"] = {
        "instruction": ANSWER_INSTRUCTION, "max_new_tokens": args.max_new_tokens,
        "min_new_tokens": None, "do_sample": False, "use_cache": True, "precision": "float16",
        "batch_size": 1, "scoring": "Conservative single-letter parsing; unparsed answers count as incorrect.",
        "fps_seed_rule": "seed + MMStar sample index; native global Torch/CUDA RNG reset per generation",
        "configurations": {name: CONFIGURATIONS[name] for name in args.configurations}, "basis_tokens": 10,
        "order": "Sorted selected rows; configuration order rotates by sample.",
        "warmup": "One unscored max-2-token generation per configuration on the first image.",
        "note": "Native diagnostic pilot, not official MMStar scoring or a controlled timing benchmark. "
                "Token-level reconstruction errors/intervention traces are not yet collected.",
    }
    samples = prepare_samples(args, report)
    print(f"Decoded and validated {len(samples)} MMStar images; loading original LLaVA once.", flush=True)
    predictions_path = args.output.with_suffix(".jsonl")
    report["predictions_file"] = predictions_path.name
    plans = [checkpoint_plan(args.model, MODEL_REVISION, args.cache_dir, native=True),
             checkpoint_plan(args.vision_model, VISION_REVISION, args.cache_dir)]
    report["checkpoints"] = {"llava": plans[0], "vision": plans[1]}
    # The pilot follows the passed smoke; missing large checkpoints should not download silently.
    if any(plan["missing_bytes"] for plan in plans):
        raise ValueError("Pinned checkpoints are missing; run the original smoke's preflight/download first")
    model_path, vision_path = [download(plan, args.cache_dir) for plan in plans]
    tokenizer, model, processor = load_native_model(model_path, vision_path, report)
    first = samples[0]
    with Image.open(args.output.parent / first["image"]) as source:
        first_image = source.convert("RGB")
    warmup_start = time.perf_counter()
    for name in args.configurations:
        generate_native_answer(tokenizer, model, processor, first_image, first["question"], name,
                               seed=args.seed + first["index"], max_new_tokens=2, require_nonempty=False)
    report["timing"] = {
        "warmup_seconds": round(time.perf_counter() - warmup_start, 3),
        "note": "Generation includes vision/compression/prefill/decode and diagnostic hooks, "
                "excluding CPU image preparation and writes. Scored wall time includes those; "
                "model loading and warmup are separate. Scored wall time also includes one unscored "
                "baseline state check. Output lengths can differ.",
    }
    predictions = []
    scored_start = time.perf_counter()
    report["status"] = "running"
    atomic_report(args.output, report)
    try:
        with predictions_path.open("x") as output:
            for sample_number, sample in enumerate(samples):
                with Image.open(args.output.parent / sample["image"]) as source:
                    image = source.convert("RGB")
                offset = sample_number % len(args.configurations)
                order = args.configurations[offset:] + args.configurations[:offset]
                for name in order:
                    report["current_sample"] = {"index": sample["index"], "configuration": name}
                    report["current_generation"] = {}
                    generated = generate_native_answer(
                        tokenizer, model, processor, image, sample["question"], name,
                        seed=args.seed + sample["index"], max_new_tokens=args.max_new_tokens,
                        require_nonempty=False,
                        record=report["current_generation"],
                    )
                    prediction = parse_answer(generated["response"])
                    row = {**sample, **generated, "prediction": prediction,
                           "correct": prediction == sample["gold"]}
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                    predictions.append(row)
                    report.pop("current_generation", None)
                    update_summary(report, predictions, args.configurations)
                    atomic_report(args.output, report)
                if sample_number == 0 and len(args.configurations) > 1:
                    before = next(row for row in predictions if row["configuration"] == "baseline")
                    repeated = generate_native_answer(
                        tokenizer, model, processor, image, sample["question"], "baseline",
                        seed=args.seed + sample["index"], max_new_tokens=args.max_new_tokens,
                        require_nonempty=False,
                    )
                    report["baseline_state_check"] = {
                        "index": sample["index"], "before": before["response"], "after": repeated["response"],
                        "matched": before["response"] == repeated["response"],
                        "generation_seconds": repeated["generation_seconds"], "scored": False,
                    }
                    if not report["baseline_state_check"]["matched"]:
                        raise RuntimeError("Repeated baseline answer changed after switching configurations")
                metrics = " | ".join(f"{name}={report['summaries'][name]['correct']}/"
                                     f"{report['summaries'][name]['evaluated']}" for name in args.configurations)
                print(f"[{sample_number + 1}/{len(samples)}] {metrics}", flush=True)
        if len(predictions) != len(samples) * len(args.configurations):
            raise RuntimeError("Incomplete prediction count")
        report.pop("current_sample", None)
        report["status"] = "passed"
    finally:
        update_summary(report, predictions, args.configurations)
        report["timing"]["scored_wall_seconds"] = round(time.perf_counter() - scored_start, 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--vision-model", default=VISION_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-original-hf-cache"))
    parser.add_argument("--dataset-cache", type=Path, default=Path("/tmp/apet-original-mmstar-cache"))
    parser.add_argument("--per-category", type=int, default=4)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=["baseline", "combined"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_category < 1 or args.max_new_tokens < 1:
        parser.error("--per-category and --max-new-tokens must be positive")
    if "baseline" not in args.configurations or len(set(args.configurations)) != len(args.configurations):
        parser.error("Include baseline exactly once and use unique configurations")
    if args.output.exists() or args.output.with_suffix(".jsonl").exists():
        parser.error("Results already exist; choose a new output name to preserve previous evidence")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = provenance("Original ApET paired MMStar pilot; one model load, native input and decoder code")
    report["matplotlib_backend"] = "Agg"
    report["runner_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    try:
        validate_runtime(report)
        run(args, report)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "Interrupted; completed prediction rows are preserved."
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    write_report(report, args.output)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
