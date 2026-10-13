"""Run original ApET LLaVA on one image, with observed compression counts.

This is an execution smoke test, not a dataset baseline or accuracy benchmark.
Launch with environments/original/.venv/bin/python on Linux.
"""

import argparse
from contextlib import contextmanager
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
from check_original_env import REPO, provenance, validate_runtime, write_report


GIB = 1024 ** 3
MODEL_ID = "liuhaotian/llava-v1.5-7b"
MODEL_REVISION = "4481d270cc22fd5c4d1bb5df129622006ccd9234"
VISION_ID = "openai/clip-vit-large-patch14-336"
VISION_REVISION = "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"
CONFIGURATIONS = {
    "baseline": {"input_tokens": 576, "layers": [], "decoder_tokens": []},
    "input-only": {"input_tokens": 288, "layers": [], "decoder_tokens": []},
    "decoder-only": {"input_tokens": 576, "layers": [16], "decoder_tokens": [96]},
    "combined": {"input_tokens": 288, "layers": [16], "decoder_tokens": [96]},
}


def validate_native_config(config):
    if (config.get("architectures") != ["LlavaLlamaForCausalLM"]
            or "text_config" in config or "mm_vision_tower" not in config):
        raise ValueError("Use original-format LLaVA-1.5 weights, such as "
                         "liuhaotian/llava-v1.5-7b; llava-hf weights are a different format")


def select_files(siblings):
    """Choose one PyTorch weight format, excluding TF/Flax and extra projectors."""
    names = {file.rfilename for file in siblings if "/" not in file.rfilename}
    safe = {name for name in names if name.endswith(".safetensors")}
    weights = safe or {name for name in names
                       if name.startswith("pytorch_model") and name.endswith(".bin")}
    if not weights:
        raise ValueError("No supported PyTorch checkpoint weights found")
    metadata = {name for name in names if name.endswith(".json") or name == "tokenizer.model"}
    return [file for file in siblings if file.rfilename in weights | metadata]


def checkpoint_plan(model, revision, cache_dir, native=False):
    local = Path(model).expanduser()
    if local.is_dir():
        config = json.loads((local / "config.json").read_text())
        if native:
            validate_native_config(config)
        # Validate index references before attempting to load large local weights.
        for index in local.glob("*.index.json"):
            for name in set(json.loads(index.read_text())["weight_map"].values()):
                if not (local / name).is_file():
                    raise ValueError(f"Missing checkpoint shard: {local / name}")
        weights = list(local.glob("*.safetensors")) or list(local.glob("pytorch_model*.bin"))
        if not weights:
            raise ValueError(f"No PyTorch weights in {local}")
        return {"source": "local", "path": str(local.resolve()), "missing_bytes": 0,
                "config_sha256": hashlib.sha256((local / "config.json").read_bytes()).hexdigest(),
                "weights_bytes": sum(path.stat().st_size for path in weights),
                "note": "Record the provenance of locally attached weights separately."}
    if local.is_absolute() or model.startswith("."):
        raise ValueError(f"Checkpoint directory does not exist: {local}")
    if native and model != MODEL_ID:
        raise ValueError(f"This first smoke supports {MODEL_ID} or a local original-format checkpoint")

    from huggingface_hub import HfApi, try_to_load_from_cache
    info = HfApi().model_info(model, revision=revision, files_metadata=True)
    files = select_files(info.siblings)
    if any(file.size is None for file in files):
        raise ValueError("Checkpoint file sizes unavailable")
    missing = 0
    for file in files:
        cached = try_to_load_from_cache(model, file.rfilename, revision=info.sha,
                                        cache_dir=str(cache_dir))
        if not isinstance(cached, str) or not Path(cached).is_file():
            missing += file.size
    return {"source": "huggingface", "model_id": model, "revision": info.sha,
            "files": [file.rfilename for file in files],
            "weights_bytes": sum(file.size for file in files), "missing_bytes": missing}


def download(plan, cache_dir):
    if plan["source"] == "local":
        return plan["path"]
    from huggingface_hub import snapshot_download
    return snapshot_download(
        plan["model_id"], revision=plan["revision"], cache_dir=str(cache_dir),
        allow_patterns=plan["files"], local_files_only=(plan["missing_bytes"] == 0),
    )


def configure(model, name, basis_tokens=10):
    config = CONFIGURATIONS[name]
    model.visual_token_num = config["input_tokens"]
    model.model.basis_token_num = basis_tokens
    model.model.layer_list = list(config["layers"])
    model.model.image_token_list = [config["input_tokens"], *config["decoder_tokens"]]
    model.model.visual_id = 0
    return {**config, "basis_tokens": basis_tokens}


@contextmanager
def observe_compression(model, trace):
    """Observe original function outputs; no compression math is replaced."""
    encode = model.encode_images
    prune = model.model.layer_prune

    def observed_encode(*args, **kwargs):
        features, mask = encode(*args, **kwargs)
        trace["input"] = {"before": mask.shape[1], "after": int(mask[0].sum().item())}
        return features, mask

    def observed_prune(*args, **kwargs):
        features = kwargs["features"]
        before = features.shape[1]
        output = prune(*args, **kwargs)
        stage = kwargs["cur_num"]
        visual = min(model.model.image_token_list[stage], model.model.image_tokens[0])
        trace.setdefault("decoder", []).append({
            "layer": kwargs["rank_layer"], "sequence_before": before,
            "sequence_after": output[2].shape[1], "visual_before": visual,
            "visual_after": visual - (before - output[2].shape[1]),
        })
        return output

    model.encode_images = observed_encode
    model.model.layer_prune = observed_prune
    try:
        yield
    finally:
        model.encode_images = encode
        model.model.layer_prune = prune


def validate_trace(name, trace):
    config = CONFIGURATIONS[name]
    if trace.get("input") != {"before": 576, "after": config["input_tokens"]}:
        raise RuntimeError(f"Unexpected input compression for {name}: {trace}")
    decoder = trace.get("decoder", [])
    expected = list(zip(config["layers"], config["decoder_tokens"]))
    if [(row["layer"], row["visual_after"]) for row in decoder] != expected:
        raise RuntimeError(f"Requested decoder compression did not occur for {name}: {trace}")
    if any(row["sequence_before"] - row["sequence_after"] !=
           row["visual_before"] - row["visual_after"] for row in decoder):
        raise RuntimeError("Decoder sequence reduction does not match visual-token reduction")


def infer(model_path, vision_path, args, report):
    import torch
    from PIL import Image, ImageDraw
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
    from llava.conversation import conv_templates
    from llava.mm_utils import process_images, tokenizer_image_token
    from llava.model.builder import load_pretrained_model
    from llava.model.language_model.llava_llama_x import LlavaConfig

    if not torch.cuda.is_available():
        raise RuntimeError("Pretrained smoke requires a GPU; enable T4 x2")
    validate_native_config(json.loads((Path(model_path) / "config.json").read_text()))
    config = LlavaConfig.from_pretrained(model_path)
    # Only the local vision checkpoint path changes; native model and math remain intact.
    config.mm_vision_tower = str(vision_path)
    max_memory = {index: int(torch.cuda.mem_get_info(index)[0] - 2 * GIB)
                  for index in range(torch.cuda.device_count())}
    if any(value <= 0 for value in max_memory.values()):
        raise RuntimeError("Insufficient free GPU memory; stop other model processes")
    max_memory["cpu"] = 0
    started = time.perf_counter()
    tokenizer, model, processor, context_length = load_pretrained_model(
        model_path, None, "llava-v1.5-7b", llm_pruning=True,
        device_map="auto", use_flash_attn=False, config=config,
        max_memory=max_memory, visual_token_num=576, selected_indices=[],
        attn_implementation="sdpa",
    )
    model.eval()
    if type(model).__name__ != "LlavaLlamaForCausalLM_X":
        raise RuntimeError("Original decoder-compression model was not loaded")
    cpu_parameters = [name for name, value in model.named_parameters() if value.device.type != "cuda"]
    if cpu_parameters:
        raise RuntimeError(f"Model has CPU/disk parameters: {cpu_parameters[:5]}")
    report["model"] = {
        "class": type(model).__name__, "module": type(model).__module__,
        "context_length": context_length, "attention": "sdpa", "precision": "float16",
        "device_map": {name: str(device) for name, device in model.hf_device_map.items()},
        "load_seconds": round(time.perf_counter() - started, 3),
    }
    if args.image:
        with Image.open(args.image) as source:
            image = source.convert("RGB")
    else:
        image = Image.new("RGB", (336, 336), "white")
        ImageDraw.Draw(image).rectangle((70, 70, 266, 266), fill="red")
    image_path = args.output.with_suffix(".png")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(image_path)
    conversation = conv_templates["vicuna_v1"].copy()
    conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + args.question)
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
    device = model.get_input_embeddings().weight.device
    input_ids = input_ids.to(device)
    images, _ = process_images([image], processor, model.config)
    if not isinstance(images, torch.Tensor) or images.shape[0] != 1:
        raise RuntimeError("Smoke supports one LLaVA-1.5 image tensor")
    images = images.to(device=device, dtype=torch.float16)
    report["protocol"] = {
        "question": args.question, "prompt": prompt, "image": str(image_path),
        "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "seed": args.seed, "do_sample": False, "use_cache": True, "batch_size": 1,
        "min_new_tokens": args.min_new_tokens, "max_new_tokens": args.max_new_tokens,
    }
    report["runs"] = []
    for name in args.configurations:
        settings = configure(model, name)
        trace = {}
        record = {"configuration": name, "settings": settings, "status": "running", "trace": trace}
        report["runs"].append(record)
        # Preserve partial evidence if a later configuration fails or is interrupted.
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        for index in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(index)
            torch.cuda.synchronize(index)
        started = time.perf_counter()
        with observe_compression(model, trace), torch.inference_mode():
            output = model.generate(
                input_ids, images=images, image_sizes=[image.size], do_sample=False,
                min_new_tokens=args.min_new_tokens, max_new_tokens=args.max_new_tokens,
                use_cache=True, return_dict_in_generate=True, output_scores=True,
            )
        for index in range(torch.cuda.device_count()):
            torch.cuda.synchronize(index)
        record["generation_seconds"] = round(time.perf_counter() - started, 3)
        record["response"] = tokenizer.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()
        record["generated_tokens"] = len(output.scores)
        record["peak_allocated_gib"] = {
            str(index): round(torch.cuda.max_memory_allocated(index) / GIB, 3)
            for index in range(torch.cuda.device_count())
        }
        if not record["response"] or not args.min_new_tokens <= len(output.scores) <= args.max_new_tokens:
            raise RuntimeError("Empty answer or unexpected generation length")
        validate_trace(name, trace)
        record["status"] = "passed"
        del output
        print(f"{name}: {record['response']} | compression: {trace}", flush=True)
    report["note"] = "Execution checks only. Inspect the answer; no accuracy or speed claim. " \
                     "At least four output tokens exercise cached decoding. Native algorithm files are unchanged."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--vision-model", default=VISION_ID)
    parser.add_argument("--vision-revision", default=VISION_REVISION)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-original-hf-cache"))
    parser.add_argument("--download-budget-gib", type=float, default=0)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--configurations", nargs="+", choices=CONFIGURATIONS, default=["baseline", "combined"])
    parser.add_argument("--seed", type=int, default=590)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--question", default="What color is the square? Answer briefly.")
    parser.add_argument("--min-new-tokens", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--output", type=Path, default=REPO / "research/results/original-apet-smoke.json")
    args = parser.parse_args()
    if not math.isfinite(args.download_budget_gib) or args.download_budget_gib < 0:
        parser.error("--download-budget-gib must be finite and nonnegative")
    if not 4 <= args.min_new_tokens <= args.max_new_tokens:
        parser.error("Require 4 <= min-new-tokens <= max-new-tokens for the cached-decode smoke")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = "0"
    report = provenance("Original LLaVA ApET one-image execution smoke; no extraction or HF adapter")
    try:
        validate_runtime(report)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        plans = [checkpoint_plan(args.model, args.revision, args.cache_dir, native=True),
                 checkpoint_plan(args.vision_model, args.vision_revision, args.cache_dir)]
        report["checkpoints"] = {"llava": plans[0], "vision": plans[1]}
        missing = sum(plan["missing_bytes"] for plan in plans)
        required = missing + GIB if missing else 0
        free = shutil.disk_usage(args.cache_dir).free
        report["storage"] = {"missing_download_gib": round(missing / GIB, 3),
                             "filesystem_free_gib": round(free / GIB, 3),
                             "download_budget_gib": args.download_budget_gib,
                             "fits": required <= free and required <= args.download_budget_gib * GIB}
        if args.preflight_only:
            report["status"] = "preflight_complete"
        else:
            if not report["storage"]["fits"]:
                raise ValueError("Missing checkpoints exceed the cache budget or free space; "
                                 "inspect preflight and provide --download-budget-gib")
            model_path, vision_path = [download(plan, args.cache_dir) for plan in plans]
            infer(model_path, vision_path, args, report)
            report["status"] = "passed"
    except KeyboardInterrupt:
        report["status"] = "interrupted"
        report["error"] = "Interrupted; completed runs and available traces are preserved."
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    write_report(report, args.output)
    return int(report["status"] not in {"passed", "preflight_complete"})


if __name__ == "__main__":
    raise SystemExit(main())
