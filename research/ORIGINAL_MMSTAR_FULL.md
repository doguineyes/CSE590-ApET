# Original ApET: full MMStar baseline

The 24-image pilot passed. Its CPU rescoring is complete in
`research/results/original-mmstar-24-rescored-v2.zip`.
See the [pilot note](notes/2026-10-13-original-mmstar-pilot.md) and
[implementation scope](IMPLEMENTATION_SCOPE.md).

This run uses all 1,500 images across four configurations: **6,000 predictions
with one model load**. Native ApET algorithms and ML dependencies are unchanged.

| Configuration | Input tokens | After decoder layer 16 |
|---|---:|---:|
| Baseline | 576 | 576 |
| Input only | 288 | 288 |
| Decoder only | 576 | 96 |
| Combined | 288 | 96 |

Commit/push to `apet-analysis`. Download your current results before stopping the
GPU session. When ready, enable **Internet and GPU T4 ×2** and use the
[full-run notebook](notebooks/kaggle_original_mmstar_full.ipynb), or append the
following four cells to the existing notebook. Do not pull/edit code during a run.

Stopping the session while editing conserves GPU quota. A restarted session may
need environment installation and weight downloads again. Do not rely on `/tmp`
being persisted. The cells support both cached weights and a fresh session using
the same lock/checkpoint revisions. Attached original-format checkpoints can be
supplied through `checkpoint_args`; weight persistence is optional for this run.

## Cell 1: get the code and choose a unique result directory

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
run_dir = Path(f"/kaggle/working/original-mmstar-1500-{stamp}")
run_dir.mkdir()
full_report = run_dir / "original-mmstar-1500.json"
original_python = repo / "environments/original/.venv/bin/python"
checkpoint_args = []  # Optional: ["--model", "/kaggle/input/...", "--vision-model", "/kaggle/input/..."]
print(f"Results: {run_dir}")
```

## Cell 2: environment, storage preflight, and source snapshots

Preflight queries metadata without downloading weights or loading the pretrained
model. The explicit 20-GiB budget covers the approximately 14.2-GiB pinned
LLaVA/CLIP files plus the preflight margin. Cached files are reused. Weights stay
in the temporary cache, outside notebook Output; they are excluded from result ZIPs.

```python
import shutil

gpus = subprocess.run(
    ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
    capture_output=True, text=True, check=True,
).stdout.strip().splitlines()
if len(gpus) != 2 or any("T4" not in name for name in gpus):
    raise RuntimeError(f"Enable GPU T4 x2 before installing/running: {gpus}")
subprocess.run(
    ["bash", "bootstrap/setup_original_linux.sh", "--require-cuda", "--expected-gpus", "2",
     "--output", str(run_dir / "original-runtime.json")],
    cwd=repo, check=True,
)
subprocess.run(
    [str(original_python), "-I", "scripts/smoke_original_apet.py", "--preflight-only",
     "--download-budget-gib", "20", *checkpoint_args,
     "--output", str(run_dir / "original-preflight.json")],
    cwd=repo, check=True,
)
preflight = json.loads((run_dir / "original-preflight.json").read_text())
print(json.dumps(preflight["storage"], indent=2))
if not preflight["storage"]["fits"]:
    raise RuntimeError("Checkpoint storage/budget preflight failed; inspect the report before inference.")

reference = json.loads((repo / "research/provenance/original_llava_reference.json").read_text())
snapshot_files = [*reference["native_source_sha256"],
    "scripts/check_original_env.py", "scripts/smoke_original_apet.py",
    "scripts/eval_original_mmstar.py", "scripts/eval_mmstar.py",
    "scripts/mmstar_scoring.py", "scripts/rescore_original_mmstar.py",
    "bootstrap/setup_original_linux.sh", "environments/original/pyproject.toml",
    "environments/original/uv.lock", "environments/original/.python-version",
    "research/provenance/original_llava_reference.json", "research/IMPLEMENTATION_SCOPE.md",
]
for name in snapshot_files:
    destination = run_dir / "source" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / name, destination)
```

## Cell 3: full paired evaluation

The v2 scorer accepts an explicit choice plus matching labeled option text. Gold
is used only after parsing to calculate correctness. Ambiguous answers stay
unparsed/incorrect. Old strict correctness/APR and bare-letter compliance are
reported separately. This is our project scoring, not an official MMStar score.

```python
subprocess.run(
    [str(original_python), "-I", "scripts/eval_original_mmstar.py", "--all-samples",
     "--configurations", "baseline", "input-only", "decoder-only", "combined",
     "--seed", "590", "--max-new-tokens", "16", "--download-budget-gib", "20",
     *checkpoint_args, "--output", str(full_report)],
    cwd=repo, check=True,
)
```

The pilot suggests roughly one hour for inference/recording, excluding setup and
missing-weight downloads; actual runtime may differ. Completed rows are flushed
and summaries updated. After a failure/interruption, use Cell 4 to download
evidence. Partial runs are not resumed; rerun in a new result directory. Variable
output lengths and diagnostic overhead mean these timings are not a controlled
speedup benchmark.

## Cell 4: inspect and download

```python
import zipfile
from IPython.display import FileLink, display

if full_report.is_file():
    result = json.loads(full_report.read_text())
    print("Status:", result["status"], "Progress:", result.get("progress"))
    if result.get("error"):
        print("Error:", result["error"])
    for name, metrics in result.get("summaries", {}).items():
        apr = result.get("comparisons", {}).get(name, {}).get("apr")
        print(name, {key: metrics.get(key) for key in
                    ["evaluated", "correct", "accuracy", "unparsed", "strict_accuracy", "format_compliance_rate"]},
              "APR:", apr)
else:
    print("No inference report yet; bundling available setup/preflight evidence.")

bundle = run_dir.with_suffix(".zip")
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(run_dir.rglob("*")):
        if path.is_file():
            archive.write(path, str(path.relative_to(run_dir)))
print(f"Download before ending the session: {bundle}")
display(FileLink(str(bundle)))
```

The ZIP includes summaries, JSONL predictions, images, setup/preflight evidence,
and source/environment snapshots. It excludes installed packages and model caches.
If the link fails, use Kaggle's Output file browser. Export the notebook separately.

## Optional CPU rescoring

The original pilot is already rescored locally. For another saved ZIP or JSON
with sibling predictions/images, ordinary Python is sufficient—no ML imports:

```bash
python3 scripts/rescore_original_mmstar.py /path/to/saved-result.zip \
  --output research/results/another-rescored-v2.json
```

Kaggle's kernel Python can run this on CPU too. Choose a new output name.
Generation provenance, raw answers, and previous scores are preserved. New
records include scorer version/source hashes and input-artifact hashes.
