"""Six-question pure Random-K/native calibration at 288 and 64 input tokens.

Run with the original locked Linux interpreter. This runner deliberately has no
broad benchmark mode yet; review the calibration ZIP/log before expansion.
"""
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
from rq2_protocol import source_hashes, require
from rq2_random_protocol import (VERSION, PROFILE, BUDGETS, RANDOM_SEEDS, CONDITIONS,
    CASE_IDS, REPEAT_IDS, FRESH_CASE, PARITY_CONDITION, validate_bundle, summarize)
from rq2_native import assert_selection_equal
from run_rq1_input import load_model, save_image, interrupt_run
from run_rq2_input_merge import copy_evidence
from run_rq2_input_merge_v2 import row_repeat_check
from smoke_original_apet import MODEL_ID, VISION_ID
from summarize_rq1_pilot import write_csv

MEASUREMENT = ("scripts/rq2_random_protocol.py", "scripts/rq2_random_native.py",
    "scripts/run_rq2_random_calibration.py", "research/protocols/rq2_pure_random_calibration_v1.json",
    "research/protocols/rq2_input_merge_v2.json",
    "scripts/rq2_native.py", "scripts/rq2_frozen_protocol.py", "scripts/rq2_parity_diagnostic.py",
    "scripts/run_rq2_input_merge.py", "scripts/run_rq2_input_merge_v2.py")
AUXILIARY = ("research/tests/test_rq2_random_protocol.py", "research/tests/test_rq2_random_native.py",
    "research/tests/test_rq2_random_runner.py", "research/tests/test_rq2_native.py",
    "research/tests/test_rq2_frozen_native.py", "research/notebooks/kaggle_rq2_random_calibration.ipynb",
    "research/RQ2_PURE_RANDOM_CALIBRATION.md")


def measurement_hashes():
    return source_hashes(extra=MEASUREMENT)


def prepare_pixels(model, processor, image):
    """Preprocess only: unlike RQ1 encode_image, this never computes E."""
    import torch
    from llava.mm_utils import process_images
    pixels, _ = process_images([image.convert("RGB")], processor, model.config)
    require(isinstance(pixels, torch.Tensor) and tuple(pixels.shape) == (1, 3, 336, 336),
            "Expected one native LLaVA-1.5 336x336 input")
    pixels = pixels.to(device=model.get_input_embeddings().weight.device, dtype=torch.float16)
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    canvas = (pixels[0].float().cpu() * std + mean).clamp(0, 1)
    return pixels, (canvas.permute(1, 2, 0) * 255).round().byte().numpy()


def validate_bank(observations, item):
    """Source identity and paired selections stay exact; fresh merges may drift."""
    archive = item["reference"]
    full = observations["baseline576"]
    require(full["encoded"]["features"]["sha256"] == archive["projected_sha256"],
            "Full native projected features differ from archived reference")
    for name, observed in observations.items():
        require(observed["source_clip"]["sha256"] == archive["clip_sha256"]
                and observed["source_patches"] == full["source_patches"], "Source CLIP states differ: " + name)
    pair_controls = {}
    for budget in BUDGETS:
        merge, no_merge = (observations[f"native-{budget}-{suffix}"] for suffix in ("merge", "no-merge"))
        pair_controls[str(budget)] = assert_selection_equal(merge, no_merge)
        require(merge["input"]["errors"] == item["feature"]["diagnostics"]["input"]["errors"]
                and merge["input"]["basis_indices"] == item["feature"]["diagnostics"]["input"]["basis_indices"],
                "Archived E/FPS changed at input budget " + str(budget))
    for seed in RANDOM_SEEDS:
        large = observations[f"pure-random-288-s{seed}"]
        small = observations[f"pure-random-64-s{seed}"]
        require(set(small["encoded"]["selected_indices_sorted"]) <=
                set(large["encoded"]["selected_indices_sorted"]), "Random samples are not nested")
        for row in (large, small, observations[PARITY_CONDITION]):
            require(row["uncompressed_projected"] == full["encoded"]["features"]
                    and row["controls"]["no_fps_E_basis_or_merge"], "Pure random altered original vectors")
    require(observations[PARITY_CONDITION]["encoded"] == full["encoded"],
            "Pure random at576 does not reproduce the native full tuple")
    return {"status": "passed", "archived_source_exact": True,
        "same_image_frozen_reuse_exact": True, "paired_selection": pair_controls,
        "pure_random_nested": True, "pure_random_no_FPS_E_basis_merge": True,
        "pure576_features_and_mask_exact": True}


def save_bank(args, report, key, snapshots):
    """Keep full native evidence once; pure masks reuse full576 feature bytes."""
    import torch
    native, masks = {}, {}
    for name, snapshot in snapshots.items():
        if name.startswith("pure-random-"):
            masks[name] = snapshot["encoded"][1].detach().cpu()
        else:
            native[name] = {k: tuple(t.detach().cpu() for t in value) if k == "encoded"
                           else value.detach().cpu() if k in ("pre_merge", "post_merge") else value
                           for k, value in snapshot.items()}
    relative = "features/" + key + ".pt"
    torch.save({"schema": VERSION, "native_snapshots": native, "pure_masks": masks,
        "pure_feature_reference": "baseline576.encoded[0]" if masks else None,
        "note": "Pure projected tensor bytes equal baseline576; each condition's original mask is stored."},
        args.output / relative)
    report["feature_sha256"][relative] = sha256(args.output / relative)
    return {"file": relative, "sha256": report["feature_sha256"][relative]}


def compare_fresh(first_obs, first_snapshot, observed, snapshot, first_row, row):
    from rq2_parity_diagnostic import compare_snapshots, score_comparison
    keys = ("source_clip", "source_patches", "pre_merge", "basis_slot_to_source", "input", "input_merge")
    require(all(first_obs.get(k) == observed.get(k) for k in keys)
            and first_obs["encoded"]["mask"] == observed["encoded"]["mask"],
            "Fresh native source, E, selection, basis or merge graph changed")
    return {"status": "recorded", "structural_identity_exact": True,
        "numeric_differences": compare_snapshots(first_snapshot, snapshot),
        "reference_score_difference": score_comparison(first_row["reference_score"], row["reference_score"]),
        "generated_ids_equal": first_row["generation_tokens"]["generated_token_ids"] == row["generation_tokens"]["generated_token_ids"],
        "note": "Fresh native numerical/answer drift is recorded, never replaced by a favorable realization."}


def run_panel(args, report, manifests, records, loaded):
    import torch
    from PIL import Image
    from rq1_native import max_score_delta
    from rq2_random_native import capture_condition, generate_condition
    tokenizer, model, processor = loaded
    report.update(expected_case_ids=list(CASE_IDS), questions=[], question_sha256={}, feature_sha256={},
        progress={"completed_questions": 0, "expected_questions": len(records)}, image_bank={})
    for name in ("questions", "features", "images"):
        (args.output / name).mkdir(exist_ok=True)
    bank, rows = {}, []
    last_use = {r["case"]["image_sha256"]: i for i, r in enumerate(records)}
    with (args.output / "predictions.jsonl").open("x", encoding="utf-8") as stream:
        for position, item in enumerate(records):
            case, archived = item["case"], item["reference"]
            cid, key, seed = case["case_id"], case["image_sha256"], archived["seed"]
            args.manifest = args.inputs / cid.split(":")[0] / "manifest.json"
            image = save_image(args, manifests[cid.split(":")[0]], case)
            q = {"case_id": cid, "case": item["rq2"], "status": "running", "arms": {}, "controls": {}}
            path = args.output / "questions" / (cid.replace(":", "-") + ".json")
            write_json(path, q)
            report["active_case"] = cid
            try:
                if key not in bank:
                    pixels, canvas = prepare_pixels(model, processor, image)
                    Image.fromarray(canvas).save(args.output / "images" / (key + "-canvas.png"))
                    observations, snapshots = {}, {}
                    for condition in (*CONDITIONS, PARITY_CONDITION):
                        print(cid + ": capture " + condition, flush=True)
                        observations[condition], snapshots[condition] = capture_condition(
                            model, pixels, seed, condition, key)
                        q["capture_prefix"] = observations
                        write_json(path, q)
                    feature = save_bank(args, report, key, snapshots)
                    q["feature_bank"] = feature
                    write_json(path, q)
                    checks = validate_bank(observations, item)
                    bank[key] = {"seed": seed, "pixels": pixels, "observations": observations,
                                 "snapshots": snapshots, "checks": checks, "feature": feature}
                    report["image_bank"][key] = {"seed": seed, "feature": feature,
                        "encodings": observations, "controls": checks}
                    q.pop("capture_prefix", None)
                cache = bank[key]
                require(cache["seed"] == seed, "Same-image questions use different seeds")
                # Check archived identity for every question, including cached image pairs.
                checks = validate_bank(cache["observations"], item)
                q["controls"].update(checks)
                q["feature_bank"] = cache["feature"]
                pixels, snapshots = cache["pixels"], cache["snapshots"]
                for condition in CONDITIONS:
                    print(cid + ": answer " + condition, flush=True)
                    row = generate_condition(tokenizer, model, processor, image, case, pixels,
                        condition, snapshots[condition], seed, archived["measurement_target"])
                    require(row["reference_score"]["prompt"] == archived["prompt"]
                            and row["reference_score"]["target_token_ids"] == archived["target_token_ids"],
                            "Archived fixed question/target changed")
                    q["arms"][condition] = row
                    write_json(path, q)
                full = q["arms"]["baseline576"]
                delta = max_score_delta(full["reference_score"], archived["controls"]["reference"])
                require(delta == 0 and full["scoring"]["correct"], "Archived full reference drift; no retargeting")
                q["controls"]["baseline_reference_max_delta"] = delta
                parity = generate_condition(tokenizer, model, processor, image, case, pixels,
                    PARITY_CONDITION, snapshots[PARITY_CONDITION], seed, archived["measurement_target"])
                require(parity["encoding"]["encoded"] == full["encoding"]["encoded"]
                        and parity["generation_tokens"]["generated_token_ids"] == full["generation_tokens"]["generated_token_ids"]
                        and max_score_delta(parity["reference_score"], full["reference_score"]) == 0,
                        "Pure576 feature, generated-ID or reference-score parity failed")
                q["controls"]["pure576_parity"] = {"status": "passed", "row": parity,
                    "features_mask_and_generated_ids_exact": True, "reference_score_max_delta": 0}
                if cid in REPEAT_IDS:
                    q["controls"]["fixed_replays"] = {}
                    for condition in CONDITIONS:
                        again = generate_condition(tokenizer, model, processor, image, case, pixels,
                            condition, snapshots[condition], seed, archived["measurement_target"])
                        q["controls"]["fixed_replays"][condition] = row_repeat_check(q["arms"][condition], again)
                        write_json(path, q)
                if cid == FRESH_CASE:
                    q["controls"]["fresh_native"] = []
                    for budget in BUDGETS:
                        condition = f"native-{budget}-merge"
                        for repeat in range(2):
                            obs, fresh = capture_condition(model, pixels, seed, condition, key)
                            feature = save_bank(args, report, key + "-" + condition + f"-fresh{repeat + 1}",
                                                {condition: fresh})
                            entry = {"condition": condition, "repeat": repeat + 1,
                                     "encoding": obs, "feature": feature}
                            q["controls"]["fresh_native"].append(entry)
                            write_json(path, q)
                            row = generate_condition(tokenizer, model, processor, image, case, pixels,
                                condition, fresh, seed, archived["measurement_target"])
                            entry["row"] = row
                            write_json(path, q)
                            entry["comparison"] = compare_fresh(cache["observations"][condition], snapshots[condition],
                                obs, fresh, q["arms"][condition], row)
                            write_json(path, q)
                            del fresh
                q["status"] = "passed"
                write_json(path, q)
                report["question_sha256"][str(path.relative_to(args.output))] = sha256(path)
                for row in q["arms"].values():
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    rows.append(row)
                stream.flush()
                report["questions"].append({"case_id": cid, "status": "passed"})
                report["progress"]["completed_questions"] = len(report["questions"])
                report["summaries"] = summarize(rows)
                write_json(args.output / "report.json", report)
                print("Completed " + cid + ": " + " / ".join(k + "=" + repr(v["response"]) for k, v in q["arms"].items()), flush=True)
                if last_use[key] == position:
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
    rows = []
    for path in sorted((output / "questions").glob("*.json")):
        question = json.loads(path.read_text())
        for name, row in question.get("arms", {}).items():
            rows.append({"case_id": question["case_id"], "status": question["status"],
                "condition": name, "question": question["case"]["question"], "response": row["response"],
                "correct": row["scoring"]["correct"], "consensus_score": row["scoring"].get("consensus_score"),
                "reference_mean_logprob": row["reference_score"]["mean_logprob"]})
    if rows:
        write_csv(output / "answers.csv", rows)
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>RQ2 pure random calibration</title>'
        '<style>body{font:16px system-ui;max-width:1100px;margin:2em auto}pre{white-space:pre-wrap}</style>'
        '<h1>Pure random / native ApET: 288 and 64 input tokens</h1><p>Status: ' + html.escape(report["status"]) +
        '</p><p>Six selected diagnostic questions. Three random replicates per budget. Decoder pruning off. '
        'Controls test execution, not an accuracy threshold. No population or speed claim.</p>'
        '<p><a href="report.json">Report</a> · <a href="predictions.jsonl">Raw predictions</a></p><pre>' +
        html.escape(json.dumps(report.get("summaries", {}), indent=2)) + '</pre>', encoding="utf-8")
    write_json(output / "artifact-sha256.json", {str(p.relative_to(output)): sha256(p)
               for p in sorted(output.rglob("*")) if p.is_file()})
    with zipfile.ZipFile(output.with_suffix(".zip"), "x", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    print("Status:", report["status"], "Download:", output.with_suffix(".zip"), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration", action="store_true", required=True)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--vision-model", default=VISION_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-original-hf-cache"))
    parser.add_argument("--download-budget-gib", type=float, default=0)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-gpus", type=int, default=2)
    parser.add_argument("--setup-log", type=Path)
    parser.add_argument("--runtime-report", type=Path)
    args = parser.parse_args(argv)
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.output.exists() or args.output.with_suffix(".zip").exists():
        parser.error("Preserve existing output; choose a new path")
    if args.inputs == args.output or args.inputs in args.output.parents:
        parser.error("Output must be outside accepted inputs")
    if args.expected_gpus < 1 or not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        parser.error("Invalid GPU count/download budget")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    signal.signal(signal.SIGTERM, interrupt_run)
    args.output.mkdir(parents=True)
    started = time.perf_counter()
    report = {"version": VERSION, "mode": "calibration", "status": "running", "calibration_passed": False,
        "scientific_benchmark_passed": False,
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    try:
        report.update(provenance("Pure random and native input ApET at288/64; execution calibration only"))
        report["status"] = "running"
        validate_runtime(report)
        verify_native_reference(report)
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count() == args.expected_gpus,
                "Enable the expected CUDA GPUs")
        profile, index, manifests, records = validate_bundle(args.inputs)
        report.update(profile=profile, profile_sha256=sha256(PROFILE),
            measurement_sources_sha256=measurement_hashes(), conditions=CONDITIONS)
        copy_evidence(args, report)
        for name in AUXILIARY:
            dest = args.output / "evidence/sources" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPO / name, dest)
        report["evidence_sha256"] = {str(p.relative_to(args.output / "evidence")): sha256(p)
            for p in (args.output / "evidence").rglob("*") if p.is_file()}
        write_json(args.output / "report.json", report)
        print("Loading original LLaVA once: six questions, eleven primary conditions each.", flush=True)
        loaded = load_model(args, report)
        run_panel(args, report, manifests, records, loaded)
        require(report["expected_case_ids"] == list(CASE_IDS)
                and [q["case_id"] for q in report["questions"]] == list(CASE_IDS)
                and report["progress"] == {"completed_questions": len(CASE_IDS), "expected_questions": len(CASE_IDS)},
                "Incomplete calibration")
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
