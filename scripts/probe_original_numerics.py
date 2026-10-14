"""Trace native input compression and replay fixed merging inputs on Kaggle.

Run each numerical policy in a fresh process. This diagnostic uses unchanged
native ApET and the original locked environment; it does not replace a baseline.
"""

import argparse
from collections import Counter, defaultdict
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
from eval_original_mmstar import atomic_report
from gqa_scoring import INSTRUCTION, score_answer
from numerical_diagnostics import POLICIES, STAGE_ORDER, TensorStore, observe_numerical_values, policy_environment
from original_diagnostics import validate_diagnostics
from repeat_original_gqa import CASE_SPECS, SOURCE_FILES, digest, prepare_cases
from smoke_original_apet import (
    CONFIGURATIONS, GIB, MODEL_ID, MODEL_REVISION, VISION_ID, VISION_REVISION,
    checkpoint_plan, download, generate_native_answer, load_native_model,
)

EXTRA_SOURCES = ["scripts/probe_original_numerics.py", "scripts/numerical_diagnostics.py",
                 "research/GQA_NUMERICS.md", "research/notebooks/kaggle_original_gqa_numerics.ipynb"]


def plan(repeats):
    return [{"trial_id": i + 1, "question_id": qid, "configuration": name, "capture": capture, "repeat": repeat + 1}
            for i, (qid, name, capture, repeat) in enumerate(
                (qid, name, capture, repeat) for qid in CASE_SPECS
                for name in ["baseline", "input-only"] for capture in [False, True] for repeat in range(repeats))]


def summarize(rows, store, expected):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["question_id"], row["configuration"])].append(row)
    result = {}
    for (qid, name), records in grouped.items():
        observed = [row for row in records if row.get("numerics")]
        comparisons = []
        for row in observed[1:]:
            a, b = observed[0]["numerics"]["stages"], row["numerics"]["stages"]
            stages = {stage: store.compare(a[stage], b[stage]) for stage in STAGE_ORDER if stage in a and stage in b}
            comparisons.append({"reference_trial": observed[0]["trial_id"], "trial": row["trial_id"],
                                "first_differing_observed_stage": next((s for s in STAGE_ORDER
                                    if s in stages and not stages[s]["exact_equal"]), None), "stages": stages})
        result.setdefault(qid, {})[name] = {
            "raw_answer_counts": dict(Counter(row["response"] for row in records)),
            "by_capture": {str(enabled): dict(Counter(row["response"] for row in records if row["capture"] == enabled))
                           for enabled in [False, True]},
            "stage_distinct_values": {stage: len({r["numerics"]["stages"][stage]["content_id"] for r in observed
                                                  if stage in r["numerics"]["stages"]}) for stage in STAGE_ORDER
                                      if any(stage in r["numerics"]["stages"] for r in observed)},
            "first_token_scores": [{"trial": row["trial_id"], "response": row["response"],
                                    **row["numerics"]["generation_scores"]} for row in observed],
            "comparisons": comparisons,
        }
    return {"progress": {"completed_trials": len(rows), "expected_trials": expected}, "summaries": result,
            "answer_variation_detected": any(len(group["raw_answer_counts"]) > 1 for configurations in result.values()
                                             for group in configurations.values())}


def snapshot(report, output):
    for name in sorted(set(SOURCE_FILES + EXTRA_SOURCES) | set(report["source_sha256"])):
        report.setdefault("project_source_sha256", {})[name] = digest(REPO / name)
        target = output.parent / (output.stem + "-source") / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)


def fixed_input(args, report, rows):
    if not args.fixed_input_report:
        origin = "first observed input-only airplane trial of this policy"
        reference_rows, root = rows, args.output.parent
    else:
        previous = json.loads(args.fixed_input_report.read_text())
        if (previous["status"] != "passed" or previous["policy"] != "current"
                or previous["lock_sha256"] != report["lock_sha256"]
                or previous["project_source_sha256"] != report["project_source_sha256"]
                or not previous["implementation"]["native_matches_reference"]):
            raise ValueError("Fixed-input reference must be a completed matching current-policy run")
        relative = Path(previous["predictions_file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Reference predictions must stay within the run directory")
        path = args.fixed_input_report.parent / relative
        if digest(path) != previous["predictions_sha256"]:
            raise ValueError("Fixed-input reference predictions changed")
        reference_rows = [json.loads(line) for line in path.read_text().splitlines()]
        root = args.fixed_input_report.parent
        origin = {"current_policy_report_sha256": digest(args.fixed_input_report)}
    row = next(r for r in reference_rows if r["question_id"] == "2058428"
               and r["configuration"] == "input-only" and r.get("numerics"))
    if row["image_sha256"] != CASE_SPECS["2058428"]["image_sha256"]:
        raise ValueError("Fixed-input reference image differs")
    return row["numerics"], TensorStore(root), origin


def replay_merging(args, report, rows, model, store):
    import torch

    numerics, source, origin = fixed_input(args, report, rows)
    merge, scatter = dict(numerics["merge_input"]), dict(numerics["scatter_input"])
    # Keep the new bundle self-contained even when the reference lives elsewhere.
    for record, keys in [(merge, ["features", "selected_indices"]), (scatter, ["initial", "sources", "targets"])]:
        for key in keys:
            original = record[key]
            record[key] = store.save_array(source.load(original), original["device"])

    def tensor(record):
        return torch.from_numpy(store.load(record)).to(record["device"])

    features, selected = tensor(merge["features"]), tensor(merge["selected_indices"])
    initial, sources, targets = [tensor(scatter[key]) for key in ["initial", "sources", "targets"]]
    report["fixed_input_replay"] = {"origin": origin, "merge_input": merge, "scatter_input": scatter,
                                    "native_merge": [], "scatter_add_only": []}
    for name in ["native_merge", "scatter_add_only"]:
        records = report["fixed_input_replay"][name]
        first = None
        for number in range(args.replay_repeats):
            report["current_operation"] = {"phase": "fixed-input replay", "operation": name, "repeat": number + 1}
            atomic_report(args.output, report)
            with torch.inference_mode():
                if name == "native_merge":
                    output = model.token_merging(features.clone(), selected, merge["basis_tokens"], scaling=merge["scaling"])
                else:
                    output = initial.clone()
                    output.scatter_add_(scatter["dim"], targets.unsqueeze(-1).expand_as(sources), sources)
            saved = store.save_tensor(output)
            first = saved if first is None else first
            records.append({"repeat": number + 1, "output": saved, "versus_first": store.compare(first, saved)})
            report["fixed_input_replay"][name + "_distinct_outputs"] = len({r["output"]["content_id"] for r in records})
            atomic_report(args.output, report)
    report.pop("current_operation", None)


def run(args, report, samples, store):
    import torch
    from PIL import Image

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Use T4 x2 to match the reference runs")
    torch.use_deterministic_algorithms(POLICIES[args.policy]["deterministic"], warn_only=False)
    report["runtime_flags"] = {
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cuda_runtime": torch.version.cuda,
        "gpus": [{"name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i))}
                 for i in range(2)],
    }
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    plans = [checkpoint_plan(args.model, MODEL_REVISION, args.cache_dir, native=True),
             checkpoint_plan(args.vision_model, VISION_REVISION, args.cache_dir)]
    report["checkpoints"] = {"llava": plans[0], "vision": plans[1]}
    missing, free = sum(p["missing_bytes"] for p in plans), shutil.disk_usage(args.cache_dir).free
    if missing and (missing > args.download_budget_gib * GIB or missing + GIB > free):
        raise ValueError("Missing checkpoints exceed download budget/free space")
    paths = [download(p, args.cache_dir) for p in plans]
    tokenizer, model, processor = load_native_model(*paths, report)
    images = {}
    for sample in samples:
        with Image.open(args.output.parent / sample["image"]) as image:
            images[sample["question_id"]] = image.convert("RGB")
    by_id = {row["question_id"]: row for row in samples}
    control = by_id["201882614"]
    report["warmups"] = [generate_native_answer(tokenizer, model, processor, images[control["question_id"]],
        control["question"], name, seed=control["seed"], max_new_tokens=2, require_nonempty=False)
        for name in ["baseline", "input-only"]]
    report["predictions_file"] = args.output.with_suffix(".jsonl").name
    rows, started = [], time.perf_counter()
    try:
        with args.output.with_suffix(".jsonl").open("x") as stream:
            for trial in report["plan"]:
                sample = by_id[trial["question_id"]]
                report["current_trial"] = dict(trial)
                atomic_report(args.output, report)
                generation = report["current_trial"].setdefault("generation", {})
                if trial["capture"]:
                    with observe_numerical_values(model, tokenizer, store) as numerics:
                        report["current_trial"]["numerics"] = numerics
                        generate_native_answer(tokenizer, model, processor, images[sample["question_id"]], sample["question"],
                            trial["configuration"], seed=sample["seed"], max_new_tokens=16, require_nonempty=False, record=generation)
                    validate_diagnostics(generation["settings"], numerics["diagnostics"])
                    required = {"vision_patches", "projected_selected", "first_step_scores"}
                    if trial["configuration"] == "input-only":
                        required |= {"merge_before", "merge_after", "scatter_initial", "scatter_sources", "scatter_targets", "scatter_after_add"}
                    if not required <= set(numerics["stages"]):
                        raise RuntimeError("Required feature observations missing")
                else:
                    numerics = None
                    generate_native_answer(tokenizer, model, processor, images[sample["question_id"]], sample["question"],
                        trial["configuration"], seed=sample["seed"], max_new_tokens=16, require_nonempty=False, record=generation)
                row = {**trial, **generation, "policy": args.policy, "image_id": sample["image_id"],
                       "image_sha256": sample["image_sha256"], "gold": sample["gold"], "numerics": numerics,
                       **score_answer(generation["response"], sample["gold"])}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                report.update(summarize(rows, store, len(report["plan"])))
                report.pop("current_trial", None)
                atomic_report(args.output, report)
                print(f"[{args.policy} {len(rows)}/{len(report['plan'])}] {row['question_id']} "
                      f"{row['configuration']} capture={row['capture']} -> {row['response']!r}", flush=True)
        report["predictions_sha256"] = digest(args.output.with_suffix(".jsonl"))
        replay_merging(args, report, rows, model, store)
        report["status"] = "passed"
    finally:
        report.update(summarize(rows, store, len(report["plan"])))
        report["timing"] = {"trial_and_replay_seconds": round(time.perf_counter() - started, 3),
                            "note": "Tensor copies, tracing and disk recording included; not a speed benchmark."}
        if args.output.with_suffix(".jsonl").exists():
            report["predictions_sha256"] = digest(args.output.with_suffix(".jsonl"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--replay-repeats", type=int, default=10)
    parser.add_argument("--fixed-input-report", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--dataset-cache", type=Path, default=Path("/tmp/apet-original-gqa-cache"))
    parser.add_argument("--download-budget-mib", type=float, default=100)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--vision-model", default=VISION_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-original-hf-cache"))
    parser.add_argument("--download-budget-gib", type=float, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (not 2 <= args.repeats <= 10 or not 2 <= args.replay_repeats <= 30 or
            any(not math.isfinite(x) or x < 0 for x in [args.download_budget_gib, args.download_budget_mib])):
        parser.error("Use 2–10 trial repeats, 2–30 replays and finite nonnegative budgets")
    if args.data_manifest and args.source_dir:
        parser.error("Choose a prepared manifest or source parquets")
    if args.output.suffix != ".json" or args.output.exists() or args.output.with_suffix(".jsonl").exists():
        parser.error("Choose a new .json output path; existing results are not overwritten")
    # All module-level project imports are Torch-free. Set the policy before CUDA/PyTorch initialization.
    if "torch" in sys.modules:
        parser.error("Run in a fresh process so workspace configuration precedes Torch initialization")
    environment = policy_environment(args.policy, os.environ)
    if "CUBLAS_WORKSPACE_CONFIG" in environment:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = environment["CUBLAS_WORKSPACE_CONFIG"]
    else:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = provenance("Native ApET numerical diagnostic; explicit runtime policy, unchanged compression math")
    report.update(schema="gqa-native-numerics-v1", policy=args.policy, plan=plan(args.repeats),
                  protocol={"policy": POLICIES[args.policy], "repeats": args.repeats, "replay_repeats": args.replay_repeats,
                            "instruction": INSTRUCTION, "max_new_tokens": 16, "do_sample": False, "use_cache": True,
                            "configurations": {n: CONFIGURATIONS[n] for n in ["baseline", "input-only"]},
                            "note": "Current/default, workspace-only and deterministic are distinct diagnostic policies. "
                                    "Feature capture affects scheduling; counts-only controls are included. "
                                    "Fixed scatter replay is an isolated operator probe, not an inference replacement."})
    store = TensorStore(args.output.parent)
    report.update(summarize([], store, len(report["plan"])))
    try:
        verify_native_reference(report)
        snapshot(report, args.output)
        samples = prepare_cases(args, report)
        if args.prepare_only:
            report["status"] = "data_prepared"
        else:
            validate_runtime(report)
            report["status"] = "running"
            run(args, report, samples, store)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "Interrupted; completed trials, tensors and available current-operation evidence retained."
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    atomic_report(args.output, report)
    print(json.dumps({"policy": args.policy, "status": report["status"], "progress": report["progress"],
                      "answer_variation_detected": report["answer_variation_detected"], "error": report.get("error"),
                      "report": str(args.output)}, indent=2), flush=True)
    return int(report["status"] not in {"passed", "data_prepared"})


if __name__ == "__main__":
    raise SystemExit(main())
