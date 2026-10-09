"""Run one pretrained HF LLaVA image; preflight checks storage without weights.

Use .venv/bin/python. This exercises the baseline, not legacy LLaVA or ApET.
"""

import argparse
import fnmatch
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3
MODEL_ID = "llava-hf/llava-1.5-7b-hf"
MODEL_REVISION = "b234b804b114d9e37bb655e11cbbb5f5e971b7a9"
PATTERNS = ["*.safetensors", "*.json", "tokenizer.model", "*.jinja"]


def storage_plan(missing_bytes, free_bytes, budget_gib):
    # Notebook output quota can be smaller than filesystem free space.
    # The caller supplies the quota budget; retain 1 GiB for other output.
    required = missing_bytes + GIB if missing_bytes else 0
    return {
        "missing_download_gib": round(missing_bytes / GIB, 3),
        "required_with_headroom_gib": round(required / GIB, 3),
        "filesystem_free_gib": round(free_bytes / GIB, 3),
        "download_budget_gib": budget_gib,
        "fits": required <= free_bytes and required <= budget_gib * GIB,
    }


def checkpoint(args, report):
    local = Path(args.model).expanduser()
    if local.is_dir():
        config = json.loads((local / "config.json").read_text())
        if config.get("model_type") != "llava":
            raise ValueError("Expected an HF LLaVA checkpoint (model_type=llava), not legacy weights")
        weights = list(local.glob("*.safetensors"))
        if not weights:
            raise ValueError("No safetensors weights found in the checkpoint directory")
        index = local / "model.safetensors.index.json"
        if index.is_file():
            for name in set(json.loads(index.read_text())["weight_map"].values()):
                if not (local / name).is_file():
                    raise ValueError(f"Missing checkpoint shard: {name}")
        report["checkpoint"] = {
            "source": "local", "path": str(local.resolve()),
            "weights_gib": round(sum(path.stat().st_size for path in weights) / GIB, 3),
            "config_sha256": hashlib.sha256((local / "config.json").read_bytes()).hexdigest(),
            "note": "Local weight provenance must be recorded with the attached input.",
        }
        return str(local.resolve())
    if local.is_absolute() or args.model.startswith("."):
        raise ValueError(f"Checkpoint directory does not exist: {local}")
    if args.model != MODEL_ID:
        raise ValueError(f"This first test supports {MODEL_ID} or a local HF LLaVA directory")

    # Avoid an additional Xet chunk cache alongside the actual checkpoint.
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    from huggingface_hub import HfApi, snapshot_download, try_to_load_from_cache

    info = HfApi().model_info(args.model, revision=args.revision, files_metadata=True)
    selected = [file for file in info.siblings
                if "/" not in file.rfilename
                and any(fnmatch.fnmatch(file.rfilename, pattern) for pattern in PATTERNS)]
    if not selected or any(file.size is None for file in selected):
        raise ValueError("Model file sizes unavailable; refusing an unbudgeted download")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    for file in selected:
        cached = try_to_load_from_cache(args.model, file.rfilename, revision=info.sha,
                                        cache_dir=str(args.cache_dir))
        if not isinstance(cached, str) or not Path(cached).is_file():
            missing += file.size
    report["checkpoint"] = {
        "source": "huggingface", "model_id": args.model, "revision": info.sha,
        "files_gib": round(sum(file.size for file in selected) / GIB, 3),
        "cache_dir": str(args.cache_dir.resolve()),
    }
    report["storage"] = storage_plan(
        missing, shutil.disk_usage(args.cache_dir).free, args.download_budget_gib,
    )
    if args.preflight_only:
        return None
    if not report["storage"]["fits"]:
        raise ValueError("Checkpoint exceeds the download budget/free space. Attach HF-format "
                         "weights under /kaggle/input, or provide a verified larger cache budget.")
    return snapshot_download(
        args.model, revision=info.sha, cache_dir=str(args.cache_dir),
        allow_patterns=[file.rfilename for file in selected],
        local_files_only=(missing == 0),
    )


def infer(model_path, args, report):
    from PIL import Image, ImageDraw
    from llava_runtime import generate_answer, load_llava

    model, processor = load_llava(model_path, report)
    image = Image.new("RGB", (336, 336), "white")
    ImageDraw.Draw(image).rectangle((70, 70, 266, 266), fill="red")
    image_path = args.output.with_suffix(".png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    question = "What color is the square? Answer briefly."
    report["inference"] = {
        **generate_answer(model, processor, image, question, args.max_new_tokens),
        "image": str(image_path),
        "note": "A nonempty response passes the execution smoke test; inspect its content. "
                "This single run is not an accuracy or speed benchmark.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID, help="HF ID or local checkpoint directory")
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=REPO / ".runtime/huggingface")
    parser.add_argument("--download-budget-gib", type=float, default=0,
                        help="Remaining quota at the cache location; default permits no download")
    parser.add_argument("--preflight-only", action="store_true", help="No model download or inference")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, default=REPO / "research/results/llava-smoke.json")
    args = parser.parse_args()
    if not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        parser.error("--download-budget-gib must be a finite nonnegative number")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    report = {"status": "failed", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "executable": sys.executable, "scope": "HF LLaVA baseline; ApET disabled"}
    try:
        if Path(sys.prefix).resolve() != (REPO / ".venv").resolve():
            raise RuntimeError("Launch with the project's .venv/bin/python")
        report["packages"] = {name: importlib.metadata.version(name)
                              for name in ["torch", "transformers", "accelerate", "huggingface-hub"]}
        report["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
        report["lock_sha256"] = hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest()
        model_path = checkpoint(args, report)
        if args.preflight_only:
            report["status"] = "preflight_complete"
        else:
            infer(model_path, args, report)
            report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    serialized = json.dumps(report, indent=2)
    print(serialized, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialized + "\n")
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
