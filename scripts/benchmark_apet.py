"""Compare input-stage ApET at fixed generated lengths on balanced MMStar images.

This is a timing workload, with EOS suppressed until the requested length.
Forced continuations are saved for traceability and are never scored as answers.
"""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from eval_mmstar import DATASET_ID, DATASET_REVISION, load_mmstar_parquet, prepare_question
from eval_mmstar import save_report, select_rows, validate_images
from smoke_llava import MODEL_ID, MODEL_REVISION, checkpoint


REPO = Path(__file__).resolve().parents[1]


def trial_plan(rows, output_lengths, budgets, repeats, seed):
    """Keep each image/repeat/length paired; rotate which budget goes first."""
    blocks = [(row, length, repeat) for row in rows for length in output_lengths
              for repeat in range(repeats)]
    random.Random(seed).shuffle(blocks)
    plan = []
    for number, (row, length, repeat) in enumerate(blocks):
        offset = number % len(budgets)
        for budget in budgets[offset:] + budgets[:offset]:
            plan.append({"row": row, "output_tokens": length, "repeat": repeat,
                         "visual_tokens": budget})
    return plan


def summarize_trials(trials):
    groups = defaultdict(list)
    baseline = {}
    seen = set()
    for trial in trials:
        key = (trial["index"], trial["output_tokens"], trial["repeat"])
        if (*key, trial["visual_tokens"]) in seen:
            raise ValueError("Duplicate benchmark trial")
        seen.add((*key, trial["visual_tokens"]))
        if trial["generated_tokens"] != trial["output_tokens"]:
            raise ValueError("Benchmark generation did not reach the fixed output length")
        groups[(trial["output_tokens"], trial["visual_tokens"])].append(trial)
        if trial["visual_tokens"] == 576:
            baseline[key] = trial
    summary = {}
    for (length, budget), records in sorted(groups.items()):
        latencies = sorted(row["generation_seconds"] for row in records)
        total = sum(latencies)
        pairs = [(baseline[key], row) for row in records
                 if (key := (row["index"], length, row["repeat"])) in baseline]
        memory = {}
        for row in records:
            for device, gib in row["peak_allocated_gib"].items():
                memory[device] = max(memory.get(device, 0), gib)
        summary.setdefault(str(length), {})[str(budget)] = {
            "requests": len(records), "images": len({row["index"] for row in records}),
            "mean_generation_seconds": statistics.mean(latencies),
            "median_generation_seconds": statistics.median(latencies),
            "p95_generation_seconds": latencies[math.ceil(0.95 * len(records)) - 1],
            "max_generation_seconds": latencies[-1],
            "total_generation_seconds": round(total, 3),
            "generated_tokens_per_generation_second": len(records) * length / total if total else None,
            "peak_allocated_gib": memory,
            "paired_requests": len(pairs),
            "paired_generation_time_ratio_baseline_over_current":
                sum(before["generation_seconds"] for before, _ in pairs)
                / sum(after["generation_seconds"] for _, after in pairs) if pairs else None,
        }
    return summary


def run(args, report):
    import torch
    from huggingface_hub import hf_hub_download
    from llava_runtime import generate_answer, load_llava

    sys.path.insert(0, str(REPO))
    from apet_compression import CompressionConfig
    from apet_compression.adapters.hf_llava import HFInputCompressionAdapter

    args.preflight_only, args.download_budget_gib = False, 0
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    model_path = checkpoint(args, report)
    parquet = hf_hub_download(DATASET_ID, "mmstar.parquet", repo_type="dataset",
                              revision=args.dataset_revision, cache_dir=str(args.dataset_cache / "hub"))
    dataset = load_mmstar_parquet(parquet, args.dataset_cache / "arrow")
    selected = select_rows(dataset["category"], args.per_category, args.seed)
    validate_images(dataset, selected)
    report["dataset"] = {
        "id": DATASET_ID, "revision": args.dataset_revision, "split": "val",
        "parquet_sha256": hashlib.sha256(Path(parquet).read_bytes()).hexdigest(),
        "seed": args.seed, "per_category": args.per_category,
        "selected_rows": selected, "sample_ids": [int(dataset[row]["index"]) for row in selected],
    }
    print(f"Validated {len(selected)} images; loading the model once for all conditions.", flush=True)
    model, processor = load_llava(model_path, report)
    report["hardware"] = {
        "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                  "total_memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 3)}
                 for i in range(torch.cuda.device_count())],
        "attention_implementation": "sdpa",
    }
    # Instances retain their one-time parity result; hooks exist only during a request.
    adapters = {budget: HFInputCompressionAdapter(model, CompressionConfig(budget, args.basis_tokens))
                for budget in args.visual_tokens if budget != 576}
    first = dataset[selected[0]]
    warmup_started = time.perf_counter()
    for length in args.output_tokens:
        for budget in args.visual_tokens:
            print(f"Unscored warmup: visual={budget}, output={length}", flush=True)
            with adapters.get(budget, nullcontext(None)) as adapter:
                generated = generate_answer(
                    model, processor, first["image"], prepare_question(first["question"]), length,
                    adapter=adapter, compression_seed=args.seed + int(first["index"]), min_new_tokens=length,
                )
                if generated["generated_tokens"] != length:
                    raise RuntimeError("Warmup did not reach the fixed output length")
    report["warmup_seconds"] = round(time.perf_counter() - warmup_started, 3)
    report["legacy_feature_checks"] = {str(budget): adapter.legacy_check
                                       for budget, adapter in adapters.items()}
    plan = trial_plan(selected, args.output_tokens, args.visual_tokens, args.repeats, args.seed)
    trials = []
    records_path = args.output.with_suffix(".jsonl")
    report["measurements_file"] = str(records_path)
    measurements_started = time.perf_counter()

    def progress():
        elapsed = time.perf_counter() - measurements_started
        report["summary"] = summarize_trials(trials)
        report["measurements_wall_seconds"] = round(elapsed, 3)
        report["progress"] = {
            "completed": len(trials), "planned": len(plan),
            "estimated_remaining_seconds": round(elapsed / len(trials) * (len(plan) - len(trials)), 1)
            if trials else None,
        }

    progress()
    save_report(args.output, report)
    with records_path.open("w") as stream:
        try:
            for number, trial in enumerate(plan, 1):
                sample = dataset[trial["row"]]
                budget, length = trial["visual_tokens"], trial["output_tokens"]
                with adapters.get(budget, nullcontext(None)) as adapter:
                    generated = generate_answer(
                        model, processor, sample["image"], prepare_question(sample["question"]), length,
                        adapter=adapter, compression_seed=args.seed + int(sample["index"]), min_new_tokens=length,
                    )
                if generated["generated_tokens"] != length or generated["image_tokens"] != budget:
                    raise RuntimeError("Measured request did not match its fixed token budgets")
                record = {**generated, **trial, "index": int(sample["index"]),
                          "category": sample["category"]}
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                trials.append(record)
                if number % args.log_every == 0 or number == len(plan):
                    progress()
                    save_report(args.output, report)
                    remaining = report["progress"]["estimated_remaining_seconds"]
                    print(f"[{number}/{len(plan)}] visual={budget} output={length} "
                          f"latency={record['generation_seconds']:.3f}s remaining~{remaining / 60:.1f}min",
                          flush=True)
        finally:
            progress()
    report["status"] = "passed"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-hf-cache"))
    parser.add_argument("--dataset-cache", type=Path, default=Path("/tmp/apet-mmstar-cache"))
    parser.add_argument("--dataset-revision", default=DATASET_REVISION)
    parser.add_argument("--per-category", type=int, default=2)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--visual-tokens", type=int, nargs="+", default=[576, 288, 96])
    parser.add_argument("--basis-tokens", type=int, default=10)
    parser.add_argument("--output-tokens", type=int, nargs="+", default=[32, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=6)
    parser.add_argument("--output", type=Path, default=REPO / "research/results/apet-fixed-output.json")
    args = parser.parse_args()
    if min(args.per_category, args.repeats, args.log_every, *args.output_tokens) < 1:
        parser.error("Sample counts, repeats, logging interval and output lengths must be positive")
    if args.basis_tokens < 1 or any(not args.basis_tokens < budget <= 576 for budget in args.visual_tokens):
        parser.error("Require 1 <= basis-tokens < each visual token budget <= 576")
    if 576 not in args.visual_tokens:
        parser.error("Include 576 visual tokens as the uncompressed timing reference")
    if len(set(args.visual_tokens)) != len(args.visual_tokens) or len(set(args.output_tokens)) != len(args.output_tokens):
        parser.error("Token budgets must be unique")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Fixed-output inference timing; input-stage ApET, no decoder pruning, no accuracy scoring",
        "executable": sys.executable,
        "protocol": {
            "visual_tokens": args.visual_tokens, "basis_tokens": args.basis_tokens,
            "output_tokens": args.output_tokens, "repeats": args.repeats, "batch_size": 1,
            "precision": "float16", "do_sample": False, "use_cache": True,
            "length_control": "min_new_tokens = max_new_tokens; each generated length is checked",
            "warmup": "One unscored request per visual/output budget, including one-time legacy parity",
            "order": "Seeded shuffled image/length/repeat blocks; rotating visual-budget order within each block",
            "timing": "Wall clock with all GPUs synchronized; vision + compression + LLM prefill + decode; "
                      "excludes CPU preprocessing, result decoding/statistics and file writes",
            "note": "Synthetic forced continuations are not scored as MMStar answers. Token throughput includes "
                    "prefill; no isolated prefill/decode timings or concurrent serving throughput are measured.",
        },
    }
    started = time.perf_counter()
    try:
        if Path(sys.prefix).resolve() != (REPO / ".venv").resolve():
            raise RuntimeError("Launch with the project's .venv/bin/python")
        report["packages"] = {name: importlib.metadata.version(name) for name in
                              ["torch", "transformers", "accelerate", "datasets", "pillow"]}
        report["commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        report["lock_sha256"] = hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest()
        run(args, report)
    except KeyboardInterrupt:
        report.update(status="interrupted", error="Interrupted; completed trials are preserved, run incomplete.")
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
    report["total_wall_seconds"] = round(time.perf_counter() - started, 3)
    report["finished_utc"] = datetime.now(timezone.utc).isoformat()
    save_report(args.output, report)
    print(json.dumps({key: report[key] for key in ["status", "summary", "error", "total_wall_seconds"]
                      if key in report}, indent=2), flush=True)
    print(f"Report: {args.output}", flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
