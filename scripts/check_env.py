"""Check the isolated research runtime without downloading model weights.

The linear solve checks a primitive used by ApET, not ApET equivalence.
The tiny random LLaVA check exercises Transformers/PyTorch integration.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
EXPECTED = {
    "torch": "2.9.0+cu128",
    "torchvision": "0.24.0+cu128",
    "transformers": "4.48.2",
}


def linear_solve_smoke(torch, device):
    # A small, deterministic, well-conditioned FP32 reconstruction.
    features = torch.arange(64, device=device, dtype=torch.float32).reshape(8, 8) / 64
    basis = torch.eye(8, device=device, dtype=torch.float32)[:3]
    gram = basis @ basis.T + 1e-5 * torch.eye(3, device=device)
    coefficients = torch.linalg.solve(gram, basis @ features.T).T
    reconstructed = coefficients @ basis
    errors = torch.linalg.vector_norm(features - reconstructed, dim=-1)
    if not torch.isfinite(errors).all().item():
        raise RuntimeError(f"Non-finite reconstruction errors on {device}")
    torch.testing.assert_close(reconstructed[:, :3], features[:, :3], atol=2e-5, rtol=1e-5)
    return {"device": str(device), "max_error": errors.max().item(), "status": "passed"}


def tiny_llava_smoke(torch):
    from transformers import AutoProcessor, LlavaConfig, LlavaForConditionalGeneration

    # Check processor imports as well as the model, without network access.
    del AutoProcessor
    config = LlavaConfig(
        text_config={
            "model_type": "llama", "vocab_size": 128, "hidden_size": 64,
            "intermediate_size": 128, "num_hidden_layers": 2,
            "num_attention_heads": 4, "num_key_value_heads": 4,
            "max_position_embeddings": 128,
            "bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 0,
        },
        vision_config={
            "model_type": "clip_vision_model", "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 2,
            "num_attention_heads": 4, "image_size": 16,
            "patch_size": 4, "projection_dim": 32,
        },
        image_token_index=127,
        image_seq_length=16,
        vision_feature_layer=-2,
    )
    torch.manual_seed(590)
    model = LlavaForConditionalGeneration(config).eval()
    input_ids = torch.tensor([[1] + [127] * 16 + [3, 4]])
    pixels = torch.zeros(1, 3, 16, 16)
    with torch.inference_mode():
        outputs = model(input_ids=input_ids, pixel_values=pixels, use_cache=True)
        generated = model.generate(
            input_ids=input_ids, pixel_values=pixels, attention_mask=torch.ones_like(input_ids),
            max_new_tokens=2, do_sample=False, use_cache=True,
        )
    if outputs.logits.shape != (1, 19, 128) or not torch.isfinite(outputs.logits).all().item():
        raise RuntimeError("Tiny LLaVA forward produced invalid logits")
    if not 20 <= generated.shape[1] <= 21:
        raise RuntimeError("Tiny LLaVA generation produced an unexpected sequence length")
    return {"status": "passed", "logits_shape": list(outputs.logits.shape),
            "generated_tokens": generated.shape[1] - input_ids.shape[1]}


def command_output(command):
    try:
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=False)
        return (result.stdout or result.stderr).strip()
    except FileNotFoundError:
        return "unavailable"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true", help="Fail if CUDA is unavailable")
    parser.add_argument("--expected-gpus", type=int, help="Fail unless this many GPUs are visible")
    parser.add_argument("--output", type=Path, help="Also save the diagnostic JSON report")
    args = parser.parse_args()
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "executable": sys.executable,
        "platform": platform.platform(), "prefix": sys.prefix,
        "lock_sha256": hashlib.sha256((REPO / "uv.lock").read_bytes()).hexdigest()
        if (REPO / "uv.lock").is_file() else None,
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "nvidia_smi": command_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv",
        ]),
        "status": "failed",
    }
    try:
        if sys.version_info[:3] != (3, 11, 14):
            raise RuntimeError("Expected Python 3.11.14; launch with .venv/bin/python")
        if Path(sys.prefix).resolve() != (REPO / ".venv").resolve():
            raise RuntimeError("This is not the project's .venv; do not run in the notebook kernel")
        if platform.system() != "Linux" or platform.machine() != "x86_64":
            raise RuntimeError("This candidate environment targets Linux x86_64")

        report["packages"] = {
            name: importlib.metadata.version(name)
            for name in [*EXPECTED, "accelerate", "datasets", "numpy", "pillow",
                         "safetensors", "sentencepiece"]
        }
        for name, expected in EXPECTED.items():
            if report["packages"][name] != expected:
                raise RuntimeError(f"Expected {name}=={expected}, got {report['packages'][name]}")

        import torch
        import torchvision
        import transformers
        import accelerate
        import datasets
        import numpy
        import sentencepiece
        import safetensors
        from PIL import Image

        report["torch_location"] = torch.__file__
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["gpu_count"] = torch.cuda.device_count()
        if torch.version.cuda != "12.8":
            raise RuntimeError("Expected the CUDA 12.8 PyTorch build")
        if args.require_cuda and not report["cuda_available"]:
            raise RuntimeError("CUDA unavailable: enable a GPU accelerator and inspect nvidia-smi")
        if args.expected_gpus is not None and report["gpu_count"] != args.expected_gpus:
            raise RuntimeError(f"Expected {args.expected_gpus} GPUs, got {report['gpu_count']}")

        report["cpu_reconstruction"] = linear_solve_smoke(torch, "cpu")
        report["tiny_llava"] = tiny_llava_smoke(torch)
        report["gpus"] = []
        if report["cuda_available"]:
            for index in range(report["gpu_count"]):
                device = torch.device("cuda", index)
                with torch.inference_mode():
                    values = torch.ones(16, 16, device=device, dtype=torch.float16)
                    product = values @ values
                    torch.testing.assert_close(product, torch.full_like(product, 16))
                    reconstruction = linear_solve_smoke(torch, device)
                    torch.cuda.synchronize(device)
                report["gpus"].append({
                    "index": index, "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "fp16_matmul": "passed", "reconstruction": reconstruction,
                })
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"

    serialized = json.dumps(report, indent=2)
    print(serialized, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n")
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
