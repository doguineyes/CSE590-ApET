# Continue the passed original ApET notebook

The October 13 pilot has now passed and its saved predictions have been rescored
on CPU. See [the result note](notes/2026-10-13-original-mmstar-pilot.md).
The current runner uses the versioned v2 scorer; strict scores are retained
separately. For the next 1,500-image run, use [the full-run guide](ORIGINAL_MMSTAR_FULL.md).

Commit and push the update to `apet-analysis` first. Keep **Internet enabled** and
**GPU T4 ×2**. Append the following three cells to the existing smoke notebook.
Do not restart a working session: its installed environment and `/tmp` weight
cache can be reused. If you changed the checkpoint/cache paths in the smoke,
pass those same paths to the pilot with `--model`, `--vision-model`, and `--cache-dir`.

The cells are also supplied in
[a continuation notebook](notebooks/kaggle_original_mmstar_pilot.ipynb).
Copy its cells into your existing notebook to retain the current session's files.

## Cell 1: update and sync

This adds pinned PyArrow 19.0.1 for reading MMStar. The existing Torch,
Transformers, and other locked versions are unchanged. Setup reuses installed
packages; the notebook kernel stays unchanged. This cell assumes the checkout is
already on `apet-analysis`, as in the starter notebook.

```python
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

repo = Path("/kaggle/working/CSE590-ApET")
original_python = repo / "environments/original/.venv/bin/python"
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True)
subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True)
subprocess.run(
    ["bash", "bootstrap/setup_original_linux.sh", "--require-cuda",
     "--expected-gpus", "2", "--output", "/kaggle/working/original-runtime-mmstar.json"],
    cwd=repo, check=True,
)
```

## Cell 2: paired pilot, one model load

Four examples from each of six categories give **24 images and 96 scored
predictions**, plus four warmups and one baseline repeat. All settings use the
same pinned checkpoint, sample IDs, prompts, preprocessing, and per-sample seed.

| Configuration | Input visual tokens | After decoder layer 16 |
|---|---:|---:|
| `baseline` | 576 | 576 (no compression) |
| `input-only` | 288 | 288 (no compression) |
| `decoder-only` | 576 | 96 |
| `combined` | 288 | 96 |

The pilot downloads only the pinned MMStar parquet (about 42 MB). Large model
downloads are refused if the passed smoke's checkpoints are missing. It loads
the model once for all images and settings. Unlike the cached-decoding smoke,
this runner allows early EOS and does **not** force four output tokens.

```python
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
pilot_report = Path(f"/kaggle/working/original-mmstar-24-{stamp}.json")
subprocess.run(
    [str(original_python), "-I", "scripts/eval_original_mmstar.py",
     "--per-category", "4", "--configurations",
     "baseline", "input-only", "decoder-only", "combined",
     "--output", str(pilot_report)],
    cwd=repo, check=True,
)
```

Each completed answer is flushed to JSONL, and the summary is updated. If the
cell fails, continue to Cell 3 to download whatever evidence was saved.

## Cell 3: inspect and download

This bundles the JSON summary, JSONL predictions, original selected image bytes,
environment files, and available smoke reports. It excludes weights and `.venv`.
If the download link fails, find the ZIP in Kaggle's Output file browser.

```python
import zipfile
from IPython.display import FileLink, display

result = json.loads(pilot_report.read_text())
print(json.dumps({
    key: result.get(key) for key in
    ["status", "error", "progress", "summaries", "comparisons", "baseline_state_check"]
}, indent=2))

bundle = pilot_report.with_suffix(".zip")
image_dir = pilot_report.parent / (pilot_report.stem + "-images")
files = [pilot_report, pilot_report.with_suffix(".jsonl")]
if image_dir.is_dir():
    files.extend(sorted(path for path in image_dir.rglob("*") if path.is_file()))
files.extend(pilot_report.parent / name for name in [
    "original-runtime-mmstar.json", "original-runtime-smoke.json",
    "original-apet-preflight.json", "original-apet-smoke.json", "original-apet-smoke.png",
])
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in files:
        if path.is_file():
            archive.write(path, str(path.relative_to(pilot_report.parent)))
    for name in ["pyproject.toml", "uv.lock", ".python-version"]:
        path = repo / "environments/original" / name
        archive.write(path, f"environment/{name}")
print(f"Download: {bundle}")
display(FileLink(str(bundle)))
```

Download the ZIP before ending the session. Exporting the notebook preserves its
cells; keep the result ZIP separately for the raw predictions and images.

## What we check before scaling

- All 96 predictions completed, and each requested token reduction occurred.
- The uncompressed answer repeats correctly after switching configurations.
- Accuracy and parsing failures by configuration and category; manually inspect
  unparsed responses before changing the scoring rule.
- **Answer preservation rate (APR)**: baseline-correct paired examples still
  correct under compression / baseline-correct paired examples. A zero
  denominator is reported as null. Lost and gained sample IDs are saved.
- Raw questions, full Vicuna prompts, labels, image bytes/hashes, category, seed,
  generation settings, token counts, traces, timings, memory, package/source
  provenance, and checkpoint revisions are retained.

This is a diagnostic pilot, not an official benchmark score or an RQ1 answer.
Its traces contain token counts, not per-token reconstruction errors or answer
contribution interventions. It also has diagnostic overhead and variable output
lengths, so its timing is not a controlled speedup measurement.

After reviewing the pilot, `--per-category 250` selects all 1,500 MMStar examples
for all chosen configurations. Use a fresh output name and review the pilot
first. Then validate small GQA and TextVQA runs with their appropriate scoring
before full baselines. Instrument RQ1 after those baselines work.

## Reuse boundaries

Within one live notebook session, setup and cached weights are reusable. Separate
script invocations still reload model weights into GPU memory; this pilot saves
that cost by running all conditions in one process. Separate notebook sessions
normally have separate filesystems and processes. For now, use the same locked
setup in each; persist large weights/data as attached Kaggle inputs if downloads
become a bottleneck. Sharing a binary virtual environment adds portability and
lifecycle complications without helping this first pilot.
