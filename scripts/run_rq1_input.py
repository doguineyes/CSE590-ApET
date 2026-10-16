"""Run the bounded RQ1 input-stage pilot, or retained TextVQA generation baseline.

Launch with environments/original/.venv/bin/python -I on Kaggle, not the notebook
kernel. All measurement, scoring and caching below is project-owned instrumentation.
"""

import argparse
from datetime import datetime, timezone
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
from rq1_protocol import (VERSION, consensus_score, load_manifest, pilot_cases, sha256,
                          spearman, textvqa_evaluator, textvqa_summary, token_order, write_json)
from smoke_original_apet import (CONFIGURATIONS, MODEL_ID, MODEL_REVISION, VISION_ID,
    VISION_REVISION, checkpoint_plan, download, generate_native_answer, load_native_model)


def interrupt_run(signum, frame):
    raise KeyboardInterrupt("Notebook requested termination")


def load_model(args, report):
    plans = [checkpoint_plan(args.model, MODEL_REVISION, args.cache_dir, native=True),
             checkpoint_plan(args.vision_model, VISION_REVISION, args.cache_dir)]
    report["checkpoints"] = dict(zip(["llava", "vision"], plans))
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    missing = sum(plan["missing_bytes"] for plan in plans)
    free = shutil.disk_usage(args.cache_dir).free
    report["storage"] = {"missing_checkpoint_bytes": missing, "free_bytes": free,
                         "download_budget_gib": args.download_budget_gib}
    if missing and (missing > args.download_budget_gib * 1024**3 or missing + 1024**3 > free):
        raise ValueError("Missing checkpoints exceed download budget/free space; use --download-budget-gib 20 for a fresh session")
    return load_native_model(*[download(plan, args.cache_dir) for plan in plans], report)


def copy_evidence(args, report, data):
    destination = args.output / "evidence"
    destination.mkdir()
    shutil.copy2(args.manifest, destination / "manifest.json")
    for rel in ["environments/original/uv.lock", "environments/original/pyproject.toml",
                "research/provenance/original_llava_reference.json",
                "scripts/run_rq1_input.py", "scripts/rq1_native.py", "scripts/rq1_protocol.py",
                "scripts/analyze_rq1_input.py", "scripts/prepare_rq1_pilot.py",
                "scripts/smoke_original_apet.py", "scripts/original_diagnostics.py",
                "scripts/check_original_env.py", "scripts/gqa_scoring.py", "scripts/mmstar_scoring.py",
                "llava/eval/m4c_evaluator.py", *report["implementation"]["native_source_sha256"]]:
        target = destination / "sources" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / rel, target)
    report["evidence_sha256"] = {str(path.relative_to(destination)): sha256(path)
                                 for path in destination.rglob("*") if path.is_file()}


def save_image(args, data, case):
    from PIL import Image
    meta = data["images"][case["image_id"]]
    source = args.manifest.parent / meta["file"]
    target = args.output / "images" / (meta["sha256"] + ".jpg")
    target.parent.mkdir(exist_ok=True)
    if not target.exists():
        shutil.copy2(source, target)
    with Image.open(source) as raw:
        image = raw.convert("RGB")
        image.load()
    if list(image.size) != meta["size"]:
        raise ValueError("Decoded image size differs from manifest")
    return image


def case_seed(case, seed):
    if case["compression_seed"] is not None:
        return int(case["compression_seed"])
    import hashlib
    return (seed + int(hashlib.sha256(case["image_sha256"].encode()).hexdigest()[:8], 16)) % (2**31)


def generation_score(dataset, case, response):
    if dataset == "gqa":
        from gqa_scoring import score_answer
        return score_answer(response, case["canonical_target"])
    from mmstar_scoring import score_answer
    return score_answer(response, case["question"], case["canonical_target"])


def run_interventions(args, report, data, loaded):
    from PIL import Image
    import torch
    from rq1_native import encode_image, measure_question, replaced

    tokenizer, model, processor = loaded
    cases = pilot_cases(data, args.questions)
    if len(cases) != args.questions:
        raise ValueError("Requested more cases than the registered pilot")
    report["case_ids"] = [case["case_id"] for case in cases]
    report["protocol"] = {
        "version": VERSION, "stage": "input", "decoder_layers": [],
        "E": "Native L2 residual in 1024-D pre-projector CLIP patch space; basis=10, epsilon=1e-5, seed from archived image/sample",
        "E_pass": "Native input-only 288-token branch receives exact cached full-path CLIP features; its merged output is discarded",
        "I": "S(full) - S(single projected visual embedding set to zero); signed nats per answer token",
        "S": "Mean teacher-forced canonical reference token log probability, causal shift, no EOS",
        "reference_context": "Uncompressed 576 visual tokens; no decoder pruning; use_cache=False for scoring",
        "perturbation": "Singleton feature zeroing after projector; sequence length/order/position IDs/mask unchanged",
        "target": "Manifest canonical target fixed before inference; one leading separator space; prompt token prefix checked",
        "selection": "Outcome-enriched reviewed mechanism subset; not population sampling or proof of visual grounding",
        "sampling": "Uniform seeded permutation of all 576 indices, nested prefix; identical across same-image questions",
        "tokens_measured_per_question": args.tokens, "seed": args.seed,
        "controls": "Unchanged reference twice, clone-only no-op, native multimodal parity, final reference",
        "control_tolerance_per_answer_token": args.control_tolerance,
        "noise": "No deterministic numeric-policy change. Values near observed control drift need caution.",
        "timing": "Instrumented single-question scoring; not an inference speed benchmark",
    }
    (args.output / "questions").mkdir()
    (args.output / "features").mkdir()
    encoded = None
    current_image = None
    summaries = []
    for case in cases:
        image = save_image(args, data, case)
        seed = case_seed(case, args.seed)
        # Questions for the same image are adjacent in the frozen GQA order.
        if current_image != case["image_sha256"]:
            del encoded
            encoded = encode_image(model, processor, image, seed)
            current_image = case["image_sha256"]
            Image.fromarray(encoded["canvas"]).save(args.output / "images" / f"{current_image}-canvas.png")
            # Small frozen representations for later audit/paired interventions;
            # no model parameters or large vision attention tensors are retained.
            torch.save({"clip_with_cls": encoded["clip"].detach().cpu().contiguous(),
                        "projected_patches": encoded["projected"].detach().cpu().contiguous()},
                       args.output / "features" / f"{current_image}.pt")
            write_json(args.output / "features" / f"{current_image}.json", {
                "image_id": case["image_id"], "image_sha256": current_image, "seed": seed,
                "clip_sha256": encoded["clip_sha256"], "projected_sha256": encoded["projected_sha256"],
                "tensor_file": f"{current_image}.pt",
                "tensor_file_sha256": sha256(args.output / "features" / f"{current_image}.pt"),
                "diagnostics": encoded["diagnostics"], "processor": processor.to_dict(),
            })
        elif encoded["seed"] != seed:
            raise ValueError("Same-image questions have different compression seeds")
        free_generation = {}
        # Normal native generation through the same cached full image embeddings.
        cached = lambda *a, **kw: (encoded["projected"].clone(), encoded["mask"].clone())
        with replaced(model, "encode_images", cached):
            generate_native_answer(tokenizer, model, processor, image, case["generation_question"],
                "baseline", seed=seed, max_new_tokens=16, require_nonempty=False, record=free_generation)
        free_generation["scoring"] = generation_score(data["dataset"], case, free_generation["response"])
        path = args.output / "questions" / f"{case['question_id']}.json"
        order = token_order(case["image_sha256"], args.seed)[:args.tokens]

        def progress(record):
            record["free_generation"] = free_generation
            record["image_sha256"] = case["image_sha256"]
            count = len(record["tokens"])
            if count % 16 == 0 or count == args.tokens or record["status"] != "running":
                write_json(path, record)
            if count % 64 == 0 or count == args.tokens:
                print(f"{case['case_id']}: {count}/{args.tokens} interventions; status={record['status']}", flush=True)

        record = measure_question(tokenizer, model, case, encoded, order, args.control_tolerance, progress)
        rho = spearman([row["reconstruction_error"] for row in record["tokens"]],
                       [row["intervention_score"] for row in record["tokens"]])
        summaries.append({"case_id": case["case_id"], "image_id": case["image_id"],
                          "status": record["status"], "measured_tokens": len(record["tokens"]),
                          "spearman": rho, "seconds": record["seconds"],
                          "baseline_scoring": free_generation["scoring"],
                          "reference_noise": record["controls"]["post_token_logprob_delta"]})
        report["questions"] = summaries
        report["progress"] = {"completed_questions": len(summaries), "expected_questions": len(cases)}
        write_json(args.output / "report.json", report)
        print(f"Completed {case['case_id']}: rho={rho}, {record['seconds']:.1f}s", flush=True)
        if record["status"] != "passed":
            raise RuntimeError("Reference drift exceeded tolerance; saved measurements flagged, stopping pilot")
    report["peak_allocated_gib"] = {str(i): torch.cuda.max_memory_allocated(i) / 1024**3
                                    for i in range(torch.cuda.device_count())}
    from analyze_rq1_input import analyze
    analyze(args.output)


def run_textvqa(args, report, data, loaded):
    tokenizer, model, processor = loaded
    evaluator = textvqa_evaluator()
    report["protocol"] = {
        "version": "rq1-textvqa-generation-v1", "case_scope": "All 100 retained, no correctness filtering",
        "input": "Raw question + single word/phrase instruction; no OCR transcripts or reference answers",
        "generation": {"max_new_tokens": 16, "do_sample": False, "use_cache": True, "batch_size": 1},
        "configurations": {name: CONFIGURATIONS[name] for name in args.configurations},
        "seed": args.seed, "seed_rule": "seed + hash(image_sha256), modulo 2**31; reset per configuration",
        "scorer": "Native llava/eval/m4c_evaluator.py TextVQAAccuracyEvaluator; all ten answers, leave-one-annotator-out consensus",
        "preservation": "Full-credit preservation uses paired baseline full-credit cases; paired mean-score ratio reported separately",
        "order": "Manifest case order; baseline first, remaining configurations in supplied order",
        "note": "Generation screening only, no TextVQA token interventions or further exclusions",
    }
    rows = []
    with (args.output / "predictions.jsonl").open("x") as stream:
        for case in data["cases"]:
            image = save_image(args, data, case)
            for name in args.configurations:
                row = {"case_id": case["case_id"], "question_id": case["question_id"],
                       "image_id": case["image_id"], "image_sha256": case["image_sha256"],
                       "question": case["question"], "gold_answers": case["gold_answers"],
                       "canonical_target": case["canonical_target"], "configuration": name}
                try:
                    generate_native_answer(tokenizer, model, processor, image, case["generation_question"],
                        name, seed=case_seed(case, args.seed), max_new_tokens=16, require_nonempty=False, record=row)
                    row.update(consensus_score(evaluator, row["response"], case["gold_answers"]))
                except BaseException:
                    row["status"] = "failed"
                    stream.write(json.dumps(row, allow_nan=False) + "\n")
                    stream.flush()
                    raise
                stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                rows.append(row)
                report["summary"] = textvqa_summary(rows)
                report["progress"] = {"completed_predictions": len(rows),
                                      "expected_predictions": len(data["cases"]) * len(args.configurations)}
                write_json(args.output / "report.json", report)
                print(f"{len(rows)}/{report['progress']['expected_predictions']} {case['case_id']} {name}: "
                      f"{row['response']!r}, score={row['consensus_score']:.2f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["interventions", "textvqa"], default="interventions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory; adjacent ZIP produced even on failure")
    parser.add_argument("--questions", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--control-tolerance", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=["baseline", "input-only"])
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--vision-model", default=VISION_ID)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-original-hf-cache"))
    parser.add_argument("--download-budget-gib", type=float, default=0)
    args = parser.parse_args()
    if not 1 <= args.questions <= 12 or not 1 <= args.tokens <= 576:
        parser.error("Use 1..12 questions and 1..576 tokens; default is calibration 2 x 32")
    if not math.isfinite(args.control_tolerance) or args.control_tolerance <= 0:
        parser.error("Control tolerance must be positive and finite")
    if not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        parser.error("Download budget must be nonnegative and finite")
    if args.configurations[0] != "baseline" or len(set(args.configurations)) != len(args.configurations):
        parser.error("Use unique configurations, baseline first")
    args.output = args.output.resolve()
    args.manifest = args.manifest.resolve()
    archive = args.output.with_suffix(".zip")
    if args.output.exists() or archive.exists():
        parser.error("Output already exists; choose a new name to preserve earlier results")
    os.environ["MPLBACKEND"] = "Agg"
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    signal.signal(signal.SIGTERM, interrupt_run)
    args.output.mkdir(parents=True)
    report = provenance("Project RQ1 measurement on original ApET/LLaVA; native sources unchanged")
    report.update({"version": VERSION, "status": "running", "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}})
    started = time.perf_counter()
    try:
        validate_runtime(report)
        verify_native_reference(report)
        data = load_manifest(args.manifest)
        if (args.mode == "textvqa") != (data["dataset"] == "textvqa"):
            raise ValueError("TextVQA interventions deferred; choose its generation mode or a GQA/MMStar intervention manifest")
        report["dataset"] = {"name": data["dataset"], "manifest_sha256": sha256(args.manifest),
                             "retained_questions": len(data["cases"])}
        copy_evidence(args, report, data)
        loaded = load_model(args, report)
        if args.mode == "interventions":
            run_interventions(args, report, data, loaded)
        else:
            run_textvqa(args, report, data, loaded)
        report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "Interrupted; completed predictions/questions and token checkpoints retained"
    except Exception as error:
        report["status"] = "failed"
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        report["wall_seconds"] = time.perf_counter() - started
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / "report.json", report)
        with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as z:
            for path in sorted(args.output.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(args.output))
        print(f"Status: {report['status']}. Results: {archive}", flush=True)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
