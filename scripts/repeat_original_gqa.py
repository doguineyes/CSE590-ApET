"""Repeat two inspected GQA cases with native ApET; no algorithm changes.

Use the original Linux runtime for inference. --prepare-only also works in the
Mac editor environment. Completed trials are flushed to JSONL, including native
diagnostics when enabled. Answer variation is a finding, not an execution error.
"""

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import io
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
from gqa_scoring import INSTRUCTION, metadata as scoring_metadata, score_answer
from original_diagnostics import compare_diagnostics, observe_native_values, validate_diagnostics
from prepare_gqa_pilot import DATASET_ID, DATASET_REVISION, FILES, fetch_sources, image_seed
from smoke_original_apet import (
    CONFIGURATIONS, GIB, MODEL_ID, MODEL_REVISION, VISION_ID, VISION_REVISION,
    checkpoint_plan, download, generate_native_answer, load_native_model,
)


CASE_SPECS = {
    "2058428": {
        "role": "repeatability-case", "image_id": "n497658", "gold": "yes",
        "question": "Are there airplanes in this scene that are huge?",
        "image_sha256": "c0965ec08a2a947adaf16c20b2a402013d0766b85782f69abf555a2c2d1dfd2c",
        "rgb_pixels_sha256": "67025900ffd3e0864c74afc9b7895c7f228acc79b00e5a1ee4e39e6bb02f04ef",
        "size": [640, 427], "seed": 334882464,
        "pilot_answers": {"baseline": "Yes", "input-only": "No", "decoder-only": "Yes", "combined": "No"},
        "development_answers": {name: "Yes" for name in CONFIGURATIONS},
    },
    "201882614": {
        "role": "historically-stable-control", "image_id": "n125122", "gold": "left",
        "question": "Is the lamp on the right or on the left of the photo?",
        "image_sha256": "d734bac2630aef0af7036ee7b878b17734cb2886e170e0460d142d1994de620c",
        "rgb_pixels_sha256": "e35e461b6521725c68e33eafc6325c424ad7ce14cb097b88bcd35caf16626f28",
        "size": [640, 480], "seed": 413006391,
        "pilot_answers": {"baseline": "Left", "input-only": "Right", "decoder-only": "Left", "combined": "Right"},
        "development_answers": {"baseline": "Left", "input-only": "Right", "decoder-only": "Left", "combined": "Right"},
    },
}
SOURCE_FILES = [
    "scripts/repeat_original_gqa.py", "scripts/original_diagnostics.py", "scripts/gqa_scoring.py",
    "scripts/prepare_gqa_pilot.py", "scripts/eval_original_mmstar.py", "scripts/eval_mmstar.py",
    "scripts/mmstar_scoring.py", "llava/constants.py", "environments/original/pyproject.toml",
    "environments/original/uv.lock", "environments/original/.python-version",
    "bootstrap/setup_original_linux.sh", "research/provenance/original_llava_reference.json",
    "research/IMPLEMENTATION_SCOPE.md", "research/GQA_REPEATABILITY.md",
    "research/notebooks/kaggle_original_gqa_repeatability.ipynb",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_inputs(args):
    """Obtain only previously inspected cases; never evaluate reserved questions."""
    if args.data_manifest:
        manifest = json.loads(args.data_manifest.read_text())
        source = manifest["source"]
        if (source.get("dataset") != DATASET_ID or source.get("revision") != DATASET_REVISION
                or source.get("files") != FILES or manifest.get("study", {}).get("role") == "evaluation"):
            raise ValueError("Use pinned GQA pilot/development data, not reserved evaluation")
        selected = [row for row in manifest["questions"] if row["id"] in CASE_SPECS]
        raw_images = {}
        for row in selected:
            relative = Path(manifest["images"][row["imageId"]]["file"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Prepared image paths must stay within the data directory")
            raw_images[row["imageId"]] = (args.data_manifest.parent / relative).read_bytes()
        origin = {"prepared_manifest_sha256": digest(args.data_manifest)}
    else:
        import pyarrow.parquet as pq

        paths = fetch_sources(args.dataset_cache, args.source_dir, args.download_budget_mib)
        questions = pq.read_table(paths[next(iter(FILES))]).to_pylist()
        selected = [row for row in questions if row["id"] in CASE_SPECS]
        images = pq.read_table(paths[list(FILES)[1]]).to_pylist()
        wanted_images = {spec["image_id"] for spec in CASE_SPECS.values()}
        raw_images = {row["id"]: row["image"]["bytes"] for row in images if row["id"] in wanted_images}
        origin = {"pinned_source_files": FILES}
    if Counter(row["id"] for row in selected) != Counter({qid: 1 for qid in CASE_SPECS}):
        raise ValueError("Each fixed question ID must occur exactly once")
    return {row["id"]: row for row in selected}, raw_images, origin


def prepare_cases(args, report):
    from PIL import Image

    questions, raw_images, origin = case_inputs(args)
    destination = args.output.parent / (args.output.stem + "-data")
    destination.mkdir()
    (destination / "images").mkdir()
    samples = []
    for qid, spec in CASE_SPECS.items():
        source = questions[qid]
        if (source["imageId"] != spec["image_id"] or source["question"] != spec["question"]
                or source["answer"] != spec["gold"] or not source["isBalanced"]):
            raise ValueError(f"Question/label/image identity differs from historical case: {qid}")
        raw = raw_images[spec["image_id"]]
        if hashlib.sha256(raw).hexdigest() != spec["image_sha256"]:
            raise ValueError(f"Historical image bytes differ: {qid}")
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if (list(image.size) != spec["size"] or
                    hashlib.sha256(image.convert("RGB").tobytes()).hexdigest() != spec["rgb_pixels_sha256"]):
                raise ValueError(f"Historical decoded image differs: {qid}")
        if image_seed(590, spec["image_id"]) != spec["seed"]:
            raise ValueError("Historical image-seed rule differs")
        relative = Path("images") / (spec["image_id"] + ".jpg")
        (destination / relative).write_bytes(raw)
        samples.append({"question_id": qid, **spec, "source_question": source,
                        "original_question": spec["question"], "question": spec["question"] + "\n" + INSTRUCTION,
                        "image": str((destination / relative).relative_to(args.output.parent))})
    manifest = {"schema": "gqa-repeatability-cases-v1", "dataset": DATASET_ID,
                "revision": DATASET_REVISION, "origin": origin, "cases": samples,
                "note": "Two previously inspected development cases; no reserved evaluation inference."}
    manifest_path = destination / "manifest.json"
    atomic_report(manifest_path, manifest)
    report["dataset"] = {**manifest, "manifest_file": str(manifest_path.relative_to(args.output.parent)),
                         "manifest_sha256": digest(manifest_path)}
    return samples


def trial_plan(repeats=3):
    """Default: 36 trials per question, 72 total; all blocks are explicit."""
    plan = []

    def add(qid, phase, block, name, observed):
        plan.append({"trial_id": len(plan) + 1, "question_id": qid, "phase": phase,
                     "block": f"{qid}:{block}", "configuration": name,
                     "logging": "native-values" if observed else "counts-only"})

    for qid in CASE_SPECS:
        for name in CONFIGURATIONS:
            for _ in range(repeats):
                add(qid, "identical-repeats", f"repeat:{name}", name, False)
            for observed in [False, True, False]:
                add(qid, "logging-sandwich", f"logging:{name}", name, observed)
        for observed in [False, True]:
            for label, sequence in [
                ("input-decoder-input", ["input-only", "decoder-only", "input-only"]),
                ("combined-baseline-combined", ["combined", "baseline", "combined"]),
            ]:
                for name in sequence:
                    add(qid, "configuration-switch", f"switch:{label}:{observed}", name, observed)
    return plan


def equal_generation(first, second):
    return {key: first[key] == second[key] for key in
            ["response", "generated_tokens", "prompt", "settings", "seed", "trace"]}


def summarize_trials(rows, expected_count):
    grouped, blocks = defaultdict(list), defaultdict(list)
    for row in rows:
        grouped[(row["question_id"], row["configuration"])].append(row)
        blocks[row["block"]].append(row)
    summary = {}
    for (qid, name), records in grouped.items():
        observed = [row for row in records if row.get("diagnostics") is not None]
        summary.setdefault(qid, {})[name] = {
            "trials": len(records), "raw_answer_counts": dict(Counter(row["response"] for row in records)),
            "raw_answer_stable": len({row["response"] for row in records}) == 1,
            "by_logging": {mode: dict(Counter(row["response"] for row in records if row["logging"] == mode))
                           for mode in ["counts-only", "native-values"]},
            "correct": sum(row["correct"] for row in records),
            "pilot_answer": CASE_SPECS[qid]["pilot_answers"][name],
            "development_answer": CASE_SPECS[qid]["development_answers"][name],
            "diagnostic_comparisons": [
                {"reference_trial": observed[0]["trial_id"], "trial": row["trial_id"],
                 "fingerprint_equal": observed[0]["diagnostics"]["fingerprint"] == row["diagnostics"]["fingerprint"],
                 **compare_diagnostics(observed[0]["diagnostics"], row["diagnostics"])}
                for row in observed[1:]],
        }
    controls = []
    for block, records in blocks.items():
        if len(records) != 3 or records[0]["phase"] == "identical-repeats":
            continue
        if records[0]["phase"] == "logging-sandwich":
            pairs = [(records[0], records[1]), (records[1], records[2]), (records[0], records[2])]
        else:
            pairs = [(records[0], records[2])]
        controls.append({"block": block, "phase": records[0]["phase"],
                         "comparisons": [{"first_trial": a["trial_id"], "second_trial": b["trial_id"],
                                          **equal_generation(a, b)} for a, b in pairs]})
    return {"progress": {"completed_trials": len(rows), "expected_trials": expected_count},
            "summaries": summary, "controls": controls,
            "findings": {
                "answer_variation_detected": any(not result["raw_answer_stable"]
                                                 for group in summary.values() for result in group.values()),
                "native_value_difference_detected": any(not comparison["fingerprint_equal"]
                                                        for group in summary.values() for result in group.values()
                                                        for comparison in result["diagnostic_comparisons"]),
                "control_difference_detected": any(not all(value for key, value in comparison.items()
                                                            if key not in {"first_trial", "second_trial"})
                                                   for control in controls for comparison in control["comparisons"]),
                "interpretation": "Execution status covers completion and valid native traces. "
                                  "Variation is saved as evidence. Agreement here does not explain the historical drift.",
            }}


def snapshot_sources(report, output):
    for name in sorted(set(SOURCE_FILES) | set(report["source_sha256"])):
        report.setdefault("project_source_sha256", {})[name] = digest(REPO / name)
        target = output.parent / (output.stem + "-source") / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)


def run(args, report, samples):
    import torch
    from PIL import Image

    if torch.cuda.device_count() != 2:
        raise RuntimeError("Use T4 x2 to match the historical two-device execution")
    report["runtime_flags"] = {
        "cuda_runtime": torch.version.cuda,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "environment": {name: os.environ.get(name) for name in
                        ["CUBLAS_WORKSPACE_CONFIG", "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING",
                         "PYTHONHASHSEED", "NVIDIA_TF32_OVERRIDE"]},
        "gpus": [{"index": i, "name": torch.cuda.get_device_name(i),
                  "capability": list(torch.cuda.get_device_capability(i))} for i in range(2)],
    }
    # Record numerical policies as found; do not enable new deterministic settings.
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    plans = [checkpoint_plan(args.model, MODEL_REVISION, args.cache_dir, native=True),
             checkpoint_plan(args.vision_model, VISION_REVISION, args.cache_dir)]
    report["checkpoints"] = {"llava": plans[0], "vision": plans[1]}
    missing = sum(plan["missing_bytes"] for plan in plans)
    free = shutil.disk_usage(args.cache_dir).free
    report["storage"] = {"missing_download_gib": missing / GIB, "filesystem_free_gib": free / GIB,
                         "download_budget_gib": args.download_budget_gib}
    if missing and (missing > args.download_budget_gib * GIB or missing + GIB > free):
        raise ValueError("Missing checkpoints exceed download budget/free space; reuse cache or attached weights")
    model_path, vision_path = [download(plan, args.cache_dir) for plan in plans]
    tokenizer, model, processor = load_native_model(model_path, vision_path, report)
    images = {}
    for sample in samples:
        with Image.open(args.output.parent / sample["image"]) as source:
            images[sample["question_id"]] = source.convert("RGB")
    by_id = {sample["question_id"]: sample for sample in samples}
    control = by_id["201882614"]
    report["warmups"] = []
    for name in CONFIGURATIONS:
        row = generate_native_answer(tokenizer, model, processor, images[control["question_id"]],
                                     control["question"], name, seed=control["seed"],
                                     max_new_tokens=2, require_nonempty=False)
        report["warmups"].append({"question_id": control["question_id"], **row})
    predictions_path = args.output.with_suffix(".jsonl")
    report["predictions_file"] = predictions_path.name
    rows = []
    started = time.perf_counter()
    try:
        with predictions_path.open("x") as stream:
            for trial in report["plan"]:
                sample = by_id[trial["question_id"]]
                report["current_trial"] = dict(trial)
                atomic_report(args.output, report)
                context = observe_native_values() if trial["logging"] == "native-values" else nullcontext(None)
                with context as diagnostics:
                    if diagnostics is not None:
                        report["current_trial"]["diagnostics"] = diagnostics
                    generated = generate_native_answer(
                        tokenizer, model, processor, images[sample["question_id"]], sample["question"],
                        trial["configuration"], seed=sample["seed"], max_new_tokens=16, require_nonempty=False,
                        record=report["current_trial"].setdefault("generation", {}),
                    )
                if diagnostics is not None:
                    validate_diagnostics(CONFIGURATIONS[trial["configuration"]], diagnostics)
                row = {**trial, **generated, "image_id": sample["image_id"], "image": sample["image"],
                       "image_sha256": sample["image_sha256"], "gold": sample["gold"],
                       "original_question": sample["original_question"], "diagnostics": diagnostics,
                       **score_answer(generated["response"], sample["gold"])}
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                report.update(summarize_trials(rows, len(report["plan"])))
                report.pop("current_trial", None)
                atomic_report(args.output, report)
                print(f"[{len(rows)}/{len(report['plan'])}] q={row['question_id']} {row['phase']} "
                      f"{row['configuration']} {row['logging']} -> {row['response']!r}", flush=True)
        report["status"] = "passed"
    finally:
        report.update(summarize_trials(rows, len(report["plan"])))
        report["timing"] = {"trial_wall_seconds": round(time.perf_counter() - started, 3),
                            "note": "Includes diagnostics/recording; not a performance benchmark."}
        if predictions_path.exists():
            report["predictions_sha256"] = digest(predictions_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--data-manifest", type=Path, help="Optional existing pilot/development manifest")
    parser.add_argument("--source-dir", type=Path, help="Optional directory with both pinned GQA parquets")
    parser.add_argument("--dataset-cache", type=Path, default=Path("/tmp/apet-original-gqa-cache"))
    parser.add_argument("--download-budget-mib", type=float, default=100)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--vision-model", default=VISION_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-original-hf-cache"))
    parser.add_argument("--download-budget-gib", type=float, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if (not 2 <= args.repeats <= 10 or
            any(not math.isfinite(value) or value < 0 for value in [args.download_budget_gib, args.download_budget_mib])):
        parser.error("Use 2–10 repeats and finite nonnegative download budgets")
    if args.data_manifest and args.source_dir:
        parser.error("Choose either a prepared manifest or source parquets")
    if (args.output.suffix != ".json" or args.output.exists() or args.output.with_suffix(".jsonl").exists()
            or (args.output.parent / (args.output.stem + "-data")).exists()):
        parser.error("Choose a new .json output path; previous evidence is not overwritten")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = provenance("Native ApET repeated-case diagnostic; no new compression algorithm")
    report["schema"] = "gqa-native-repeatability-v1"
    report["plan"] = trial_plan(args.repeats)
    report["protocol"] = {
        "instruction": INSTRUCTION, "max_new_tokens": 16, "min_new_tokens": None,
        "do_sample": False, "use_cache": True, "batch_size": 1, "basis_tokens": 10, "epsilon": 1e-5,
        "configurations": CONFIGURATIONS, "repeats": args.repeats,
        "logging": "counts-only uses the existing generation/count helper; native-values also enables read-only return tracing.",
        "seed_rule": "Historical image seed from selection seed 590; reset Torch CPU/CUDA seeds on every generation.",
        "limits": "No question perturbations, seed sweep, contribution interventions or timing claims. "
                  "This shorter execution history does not reproduce either full historical session.",
    }
    report["scoring"] = scoring_metadata()
    report.update(summarize_trials([], len(report["plan"])))
    try:
        verify_native_reference(report)
        snapshot_sources(report, args.output)
        samples = prepare_cases(args, report)
        if args.prepare_only:
            report["status"] = "data_prepared"
        else:
            validate_runtime(report)
            report["status"] = "running"
            run(args, report, samples)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "Interrupted; flushed trials and available data/source snapshots are retained."
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    atomic_report(args.output, report)
    print(json.dumps({"status": report["status"], "progress": report["progress"],
                      "findings": report["findings"], "error": report.get("error"),
                      "report": str(args.output)}, indent=2), flush=True)
    return int(report["status"] not in {"passed", "data_prepared"})


if __name__ == "__main__":
    raise SystemExit(main())
