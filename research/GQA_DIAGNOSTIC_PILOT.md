# Same-image GQA diagnostic pilot

Run this **after MMStar finishes and its ZIP is downloaded**. Then pull the update
in Kaggle. Use the same live notebook to reuse the environment and weight cache,
or open the [GQA notebook](notebooks/kaggle_original_gqa_pilot.ipynb) with Internet
and GPU T4 ×2 enabled. Do not run two model processes on these GPUs at once.

## Scope and data

Default: **10 images × 4 questions × 4 configurations = 160 predictions**.
Image groups contain open and yes/no questions, with varied semantic types.
Selection uses question metadata and seed 590, not model predictions.
All questions about an image share one stable image-based FPS seed.

The preparation script downloads only two pinned Test-Dev balanced parquet files,
about 68 MB total. It creates a portable manifest and copies the selected raw image
bytes without reencoding. It runs on CPU with existing Pillow, PyArrow and HF Hub;
there are no dependency changes and no need to install `lmms-eval`.

- Source: [LMMs-Lab formatted GQA](https://huggingface.co/datasets/lmms-lab-encoder/GQA/tree/a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8).
- Revision: `a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8`.
- Source population: 12,578 labeled questions and 398 images in Test-Dev balanced.
- Original semantics: [GQA annotations](https://cs.stanford.edu/people/dorarad/gqa/download.html).
- Scoring: exact short-answer match after whitespace trimming, lowercasing, and
  trailing-period removal, following our upstream ApET GQA converter. Prose,
  synonyms, and conflicting answers are not guessed. Empty answers count wrong.
  This selected diagnostic subset is not an official aggregate GQA score.

A real CPU-prepared bundle is available locally at
`research/results/gqa-pilot-data-10x4.zip` (about 2.1 MB). Results are ignored by Git;
pushing code does not upload that bundle. The notebook prepares the same question
selection itself, so uploading the bundle is optional. To use an attached/extracted
bundle, set `PREPARED_MANIFEST` to its `manifest.json` path in Cell 1.

## Native implementation and observations

The native source reference and compatibility patch remain unchanged. This pilot
uses the same four stage settings and shared two-GPU model loader as MMStar.
Our [implementation scope](IMPLEMENTATION_SCOPE.md) distinguishes project code
from ApET. The added instrumentation is `scripts/original_diagnostics.py`.

It reads local values when the original encode/prune/merge functions return:
reconstruction errors, FPS basis indices, selected indices, basis replacement
slots, dropped-to-retained merge assignments, and merge counts. It does not
replace native functions, recalculate FPS, write native locals, or store GPU tensors.
The monitored original files must still match the passed-pilot hashes.

The notebook first tests logging on actual native functions with small synthetic
CPU tensors. Inference also compares answers/token counts/compression traces with
logging off/on for the first question under every configuration. A mismatch stops
the pilot for review; this is a limited observation check, not universal parity proof.

Question-group records compare selections, bases, merge assignments and error
vectors across questions about one image. Decoder differences are recorded,
including maximum absolute error differences and comparison tolerances. Do not
interpret them automatically as question conditioning: the image precedes the
question under causal attention. Numerical/kernel effects can differ with shape.

Indices refer to the visual stream entering a stage. Basis slots contain substituted
features; merged/decoder tokens do not necessarily map one-to-one to original image
patches. Raw error magnitudes across feature dimensions/layers need careful
normalization or rank analysis. This pilot does **not** measure intervention-based
answer contribution yet. Logging adds overhead; these are diagnostic timings.
Basis-slot/source pairs respect the actual native assignment order: input replacement
uses sorted boolean-mask positions; decoder replacement uses the selected-index order.

## Cell 1: update after MMStar, then choose paths

```python
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_URL = "https://github.com/doguineyes/CSE590-ApET.git"
REPO_REF = "apet-analysis"
repo = Path("/kaggle/working/CSE590-ApET")
if not repo.exists():
    subprocess.run(["git", "clone", REPO_URL, str(repo)], check=True)
subprocess.run(["git", "fetch", "origin"], cwd=repo, check=True)
subprocess.run(["git", "checkout", REPO_REF], cwd=repo, check=True)
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True)
subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
run_dir = Path(f"/kaggle/working/original-gqa-pilot-{stamp}")
run_dir.mkdir()
original_python = repo / "environments/original/.venv/bin/python"
data_dir = Path("/kaggle/working/gqa-pilot-data-v1-10x4-s590")
PREPARED_MANIFEST = None  # Optional: "/kaggle/input/your-subset/manifest.json"
data_manifest = Path(PREPARED_MANIFEST) if PREPARED_MANIFEST else data_dir / "manifest.json"
report_path = run_dir / "original-gqa-pilot.json"
checkpoint_args = []  # Optional original-format local model/vision paths, as in MMStar.
```

## Cell 2: reuse setup, check observers, and prepare the subset

```python
import shutil

subprocess.run(
    ["bash", "bootstrap/setup_original_linux.sh", "--require-cuda", "--expected-gpus", "2",
     "--output", str(run_dir / "original-runtime.json")],
    cwd=repo, check=True,
)
subprocess.run(
    [str(original_python), "-I", "research/tests/test_original_diagnostics_native.py"],
    cwd=repo, check=True,
)
if not data_manifest.is_file():
    if PREPARED_MANIFEST:
        raise FileNotFoundError(f"Attached manifest not found: {data_manifest}")
    subprocess.run(
        [str(original_python), "-I", "scripts/prepare_gqa_pilot.py", "--output-dir", str(data_dir),
         "--image-count", "10", "--questions-per-image", "4", "--seed", "590", "--no-zip"],
        cwd=repo, check=True,
    )
data = json.loads(data_manifest.read_text())
print("Images:", data["selection"]["image_count"], "Questions:", len(data["questions"]))
print("Question types:", data["selection"]["type_counts"])

reference = json.loads((repo / "research/provenance/original_llava_reference.json").read_text())
snapshot_files = [*reference["native_source_sha256"],
    "scripts/check_original_env.py", "scripts/smoke_original_apet.py", "scripts/eval_mmstar.py",
    "scripts/eval_original_mmstar.py", "scripts/eval_original_gqa.py", "scripts/prepare_gqa_pilot.py",
    "scripts/gqa_scoring.py", "scripts/original_diagnostics.py", "scripts/convert_gqa_for_eval.py",
    "environments/original/pyproject.toml", "environments/original/uv.lock",
    "environments/original/.python-version", "bootstrap/setup_original_linux.sh",
    "research/provenance/original_llava_reference.json", "research/IMPLEMENTATION_SCOPE.md",
]
for name in snapshot_files:
    destination = run_dir / "source" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / name, destination)
```

## Cell 3: run the diagnostic pilot

Loads the model once for all 160 predictions plus unscored warmups/parity controls.
The same live MMStar cache is reused. In a restarted session, the explicit 20-GiB
budget permits missing pinned weights to download again. No kernel package changes
or restart are needed. Every completed prediction is flushed to JSONL.

```python
subprocess.run(
    [str(original_python), "-I", "scripts/eval_original_gqa.py", "--data-manifest", str(data_manifest),
     "--configurations", "baseline", "input-only", "decoder-only", "combined",
     "--max-new-tokens", "16", "--download-budget-gib", "20", *checkpoint_args,
     "--output", str(report_path)],
    cwd=repo, check=True,
)
```

## Cell 4: inspect and download

Run this after a failure too. The bundle contains raw predictions/diagnostics,
summary and same-image comparisons, question metadata, selected images, and
environment/native/project source snapshots. It excludes weights and installed
packages. If a native observer test fails before inference, send its cell output
too because unittest output is not part of the inference JSON.

```python
import zipfile
from IPython.display import FileLink, display

if report_path.is_file():
    result = json.loads(report_path.read_text())
    print("Status:", result["status"], "Progress:", result.get("progress"))
    print("Observer parity:", result.get("observer_parity"))
    if result.get("error"):
        print("Error:", result["error"])
    for name, metrics in result.get("summaries", {}).items():
        print(name, "correct:", metrics["correct"], "/", metrics["evaluated"],
              "accuracy:", metrics["accuracy"],
              "APR:", result.get("comparisons", {}).get(name, {}).get("apr"))
bundle = run_dir.with_suffix(".zip")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            archive.write(path, str(path.relative_to(run_dir)))
print(f"Download before ending the session: {bundle}")
display(FileLink(str(bundle)))
```

Send this ZIP with the completed MMStar ZIP. We will check answer normalization,
logging parity, per-image question differences, and selected-token stability before
choosing the smaller intervention subset for RQ1.
