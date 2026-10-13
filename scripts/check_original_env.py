"""Check the original ApET environment without downloading model weights."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "environments/original"
sys.path.insert(0, str(REPO))
EXPECTED = {
    "torch": "2.1.2+cu121", "torchvision": "0.16.2+cu121",
    "transformers": "4.37.2", "accelerate": "0.21.0", "numpy": "1.26.4",
    "pillow": "10.4.0", "huggingface-hub": "0.33.4",
}
SOURCES = [
    "llava/model/llava_arch.py", "llava/model/modeling_llama_x.py",
    "llava/model/language_model/llava_llama_x.py", "llava/model/utils.py",
    "llava/model/builder.py", "llava/model/multimodal_encoder/clip_encoder.py",
    "llava/mm_utils.py", "llava/conversation.py",
    "scripts/check_original_env.py", "scripts/smoke_original_apet.py",
]


def command_output(command):
    try:
        result = subprocess.run(command, cwd=REPO, capture_output=True, text=True)
        return (result.stdout or result.stderr).strip()
    except FileNotFoundError:
        return "unavailable"


def provenance(scope):
    return {
        "status": "failed", "scope": scope,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "executable": sys.executable,
        "platform": platform.platform(), "prefix": sys.prefix,
        "commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_status": command_output(["git", "status", "--short"]),
        "lock_sha256": hashlib.sha256((PROJECT / "uv.lock").read_bytes()).hexdigest(),
        "source_sha256": {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest()
                          for name in SOURCES},
        "nvidia_smi": command_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv",
        ]),
    }


def write_report(report, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2)
    output.write_text(serialized + "\n")
    print(serialized, flush=True)


def validate_runtime(report):
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Original runtime targets Linux x86_64; execute on Kaggle/Cloud")
    if sys.version_info[:3] != (3, 10, 18):
        raise RuntimeError("Expected Python 3.10.18")
    if Path(sys.prefix).resolve() != (PROJECT / ".venv").resolve():
        raise RuntimeError("Launch with environments/original/.venv/bin/python")
    report["packages"] = {dist.metadata["Name"]: dist.version
                          for dist in importlib.metadata.distributions()}
    for name, expected in EXPECTED.items():
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise RuntimeError(f"Expected {name}=={expected}, got {actual}")


def tiny_decoder(torch):
    from transformers.generation.utils import GenerationMixin
    from llava.model.language_model.llava_llama_x import LlavaConfig, LlavaLlamaForCausalLM_X

    torch.manual_seed(590)
    config = LlavaConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0,
    )
    config._attn_implementation = "sdpa"
    model = LlavaLlamaForCausalLM_X(config, visual_token_num=576, selected_indices=[]).eval()
    model.model.layer_list = []
    model.model.image_token_list = [576]
    model.model.basis_token_num = 2
    model.model.image_token_posi = [-1]
    model.model.image_tokens = [0]
    ids = torch.tensor([[1, 3, 4]])
    with torch.inference_mode():
        output = GenerationMixin.generate(
            model, input_ids=ids, attention_mask=torch.ones_like(ids),
            do_sample=False, min_new_tokens=3, max_new_tokens=3, use_cache=True,
        )
    if output.shape != (1, 6):
        raise RuntimeError(f"Unexpected tiny original decoder output: {tuple(output.shape)}")
    return {"status": "passed", "model_class": type(model).__name__,
            "generated_tokens": 3, "note": "Random tiny text-only decoder; no pretrained weights."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--expected-gpus", type=int)
    parser.add_argument("--output", type=Path, default=REPO / "research/results/original-runtime-smoke.json")
    args = parser.parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    report = provenance("Original ApET package imports, tiny decoder, and CUDA execution checks")
    try:
        validate_runtime(report)
        import torch
        import torchvision
        import accelerate
        import sentencepiece
        from llava.model.builder import load_pretrained_model
        from llava.model.language_model.llava_llama_x import LlavaLlamaForCausalLM_X

        report["native_loader"] = load_pretrained_model.__module__
        report["native_model_class"] = LlavaLlamaForCausalLM_X.__name__
        report["torch_location"] = torch.__file__
        report["cuda_runtime"] = torch.version.cuda
        report["cuda_available"] = torch.cuda.is_available()
        report["gpu_count"] = torch.cuda.device_count()
        if torch.version.cuda != "12.1":
            raise RuntimeError("Expected the CUDA 12.1 PyTorch build")
        if args.require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; enable a Kaggle GPU accelerator")
        if args.expected_gpus is not None and torch.cuda.device_count() != args.expected_gpus:
            raise RuntimeError(f"Expected {args.expected_gpus} GPUs, got {torch.cuda.device_count()}")
        report["tiny_decoder"] = tiny_decoder(torch)
        report["gpus"] = []
        for index in range(torch.cuda.device_count()):
            device = torch.device("cuda", index)
            with torch.inference_mode():
                values = torch.ones(16, 16, device=device, dtype=torch.float16)
                torch.testing.assert_close(values @ values, torch.full_like(values, 16))
                basis = torch.eye(4, device=device, dtype=torch.float32)
                solved = torch.linalg.solve(basis + 1e-5 * basis, basis)
                if not torch.isfinite(solved).all().item():
                    raise RuntimeError(f"Non-finite reconstruction solve on {device}")
                torch.cuda.synchronize(device)
            report["gpus"].append({"index": index, "name": torch.cuda.get_device_name(index),
                                   "fp16_matmul": "passed", "fp32_solve": "passed"})
        report["status"] = "passed"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["traceback"] = traceback.format_exc()
    write_report(report, args.output)
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
