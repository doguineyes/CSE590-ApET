"""Validate the extraction, run the identity gate, then compare input ApET.

Continue the existing Kaggle session: weights and the 24-sample baseline must
already be present. Dependencies and legacy model files are left untouched.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("/kaggle/working/mmstar-baseline-24.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/apet-hf-cache"))
    parser.add_argument("--dataset-cache", type=Path, default=Path("/tmp/apet-mmstar-cache"))
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--keep-tokens", type=int, default=96)
    parser.add_argument("--basis-tokens", type=int, default=10)
    args = parser.parse_args()
    if not 1 <= args.basis_tokens < args.keep_tokens < 576:
        parser.error("Require 1 <= basis-tokens < keep-tokens < 576")
    if Path(sys.prefix).resolve() != (REPO / ".venv").resolve():
        parser.error("Launch with the project's .venv/bin/python")
    from compare_mmstar import read_predictions

    baseline = json.loads(args.baseline.read_text())
    if baseline["status"] != "passed":
        parser.error("The uncompressed baseline must have passed")
    read_predictions(baseline, args.baseline)
    if baseline["checkpoint"]["source"] != "huggingface":
        parser.error("This convenience runner expects the pinned Hugging Face baseline checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = args.output_dir / "mmstar-apet-identity-24.json"
    compressed_path = args.output_dir / f"mmstar-apet-input-{args.keep_tokens}-24.json"
    if args.baseline.resolve() in {identity_path.resolve(), compressed_path.resolve()}:
        parser.error("Output filenames must preserve the original baseline")
    print("Checking original-source parity and tiny HF cached generation.", flush=True)
    subprocess.run(
        [sys.executable, "research/tests/test_apet_extraction.py", "--require-cuda", "-v"],
        cwd=REPO, check=True,
    )
    common = [
        sys.executable, "scripts/eval_mmstar.py", "--apet",
        "--model", baseline["checkpoint"]["model_id"],
        "--revision", baseline["checkpoint"]["revision"],
        "--dataset-revision", baseline["dataset"]["revision"],
        "--cache-dir", str(args.cache_dir), "--dataset-cache", str(args.dataset_cache),
        "--per-category", str(baseline["dataset"]["per_category"]),
        "--seed", str(baseline["dataset"]["seed"]),
        "--max-new-tokens", str(baseline["protocol"]["max_new_tokens"]),
        "--basis-tokens", str(args.basis_tokens), "--compare-to", str(args.baseline),
    ]
    print("Identity gate: retain all 576 image tokens and require unchanged raw answers.", flush=True)
    subprocess.run(common + ["--keep-tokens", "576", "--output", str(identity_path)], cwd=REPO, check=True)
    identity = json.loads(identity_path.read_text())
    if identity["comparison"]["identity_check"] != "passed":
        raise RuntimeError("Identity gate did not pass; compression was not started")
    print(f"Input-stage ApET: retain {args.keep_tokens} image tokens.", flush=True)
    subprocess.run(common + ["--keep-tokens", str(args.keep_tokens), "--output", str(compressed_path)],
                   cwd=REPO, check=True)
    compressed = json.loads(compressed_path.read_text())
    final = {
        "status": "passed", "scope": "Extracted LLaVA input-stage ApET only; no decoder pruning",
        "baseline_report": str(args.baseline), "identity_report": str(identity_path),
        "compressed_report": str(compressed_path),
        "identity_check": identity["comparison"]["identity_check"],
        "legacy_feature_check": compressed["compression"]["legacy_check"],
        "comparison": compressed["comparison"],
    }
    output = args.output_dir / "apet-comparison.json"
    output.write_text(json.dumps(final, indent=2) + "\n")
    print(json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
