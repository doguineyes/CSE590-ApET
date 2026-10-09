# Candidate runtime v1: fresh Kaggle smoke test

This is a new **Linux x86_64 environment**, separate from Kaggle's notebook
kernel. It does not install the legacy `llava/`, `qwen/`, or `video-llava/`
implementations. Those integrations still need separate compatibility work.

The initial candidate uses Python 3.11.14, uv 0.9.5, PyTorch 2.9.0+cu128,
torchvision 0.24.0+cu128, and Transformers 4.48.2. Other dependencies and their
transitive versions are recorded in the committed `uv.lock`.

## First run in a completely fresh notebook

After pushing these files to GitHub, create a notebook with Internet enabled.
Choose GPU T4 x2 to check both GPUs. A CPU-only run also works if you omit
`--require-cuda --expected-gpus 2` below, but it still installs the same CUDA
PyTorch wheels so the dependency environment stays consistent.

The first install downloads several GB of packages. Leave sufficient disk space
and allow it to finish. Installation is repeated in a fresh session; within the
same checkout, rerunning the setup reuses the environment and download cache.

Cell 1: clone the branch you pushed. Prefer its exact commit SHA for recorded runs.

```python
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/doguineyes/CSE590-ApET.git"
REPO_REF = "apet-analysis"  # Or the exact commit SHA containing the runtime files.
repo = Path("/kaggle/working/CSE590-ApET")

if not repo.exists():
    subprocess.run(["git", "clone", REPO_URL, str(repo)], check=True)

subprocess.run(["git", "checkout", REPO_REF], cwd=repo, check=True)
subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True)
```

Cell 2: install the isolated runtime and run its tests.

```python
report = Path("/kaggle/working/runtime-smoke.json")
subprocess.run(
    [
        "bash", "bootstrap/setup_linux.sh",
        "--require-cuda", "--expected-gpus", "2",
        "--output", str(report),
    ],
    cwd=repo,
    check=True,
)
```

Cell 3: inspect the saved report.

```python
import json

result = json.loads(report.read_text())
print(json.dumps(result, indent=2))
assert result["status"] == "passed"
```

You can also upload `research/notebooks/kaggle_runtime_smoke.ipynb` as a starter
notebook; it contains these steps. For a one-GPU cloud VM, use
`--require-cuda --expected-gpus 1`.

## What passing means

The checker validates:

- the exact Python, Torch, torchvision, and Transformers versions;
- the project's interpreter and isolation from the notebook kernel;
- imports of the additional runtime dependencies;
- an FP32 regularized reconstruction solve on CPU;
- a tiny random-weight LLaVA forward pass and cached generation on CPU;
- FP16 matrix multiplication and the reconstruction solve on every visible GPU.

The report includes the Git revision, lockfile SHA-256, driver information,
runtime versions, GPU names, and any failure. If setup fails before Python can
run, inspect the notebook cell output: there may be no JSON report yet.

This does **not** download a model checkpoint, test real LLaVA-1.5-7B inference,
run ApET, establish ApET equivalence, or evaluate MMStar. Those are the next
steps after this runtime passes on Kaggle. CUDA wheel/driver compatibility is
confirmed by actually running CUDA operations, not by comparing version labels.

## Next: one image through pretrained LLaVA

The runtime passed on Kaggle T4 x2 at commit `b0562a1`: Python 3.11.14,
Torch 2.9.0+cu128, Transformers 4.48.2, tiny LLaVA generation, and CUDA operations
on both GPUs. This is evidence for the runtime; pretrained model inference is
the next check.

With notebook Output at 7.3 / 19.5 GiB, roughly **12.2 GiB remain**. That can hold
small result files, but not the full pretrained checkpoint. The
[HF LLaVA-1.5-7B weights](https://huggingface.co/llava-hf/llava-1.5-7b-hf/tree/b234b804b114d9e37bb655e11cbbb5f5e971b7a9)
are approximately 13.2 GiB before any headroom. Disk storage and GPU VRAM are
separate budgets.

The Output meter describes saved output; it does not establish all available
temporary capacity. The user successfully cached this checkpoint under
`/tmp/apet-hf-cache` and ran pretrained LLaVA on the red square. It answered
`Red`, with 576 image tokens and reported peak allocated memory of 6.712 and
6.867 GiB on the two GPUs. The reported generation time was 1.49 seconds for
that single diagnostic, not a benchmark. The Output meter stayed around 7 GiB.

For a fresh session, check temporary capacity with `shutil.disk_usage("/tmp")`,
then use `--cache-dir /tmp/apet-hf-cache --download-budget-gib 16` in both the
preflight and inference commands. Proceed only when `storage.fits` is true.
Keep reports under `/kaggle/working`. Temporary weights may disappear when the
session ends; attaching a checkpoint as an Input remains a reusable alternative.

After pushing the next script, update the existing notebook checkout:

```python
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True)
subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True)
```

This assumes you are on the `apet-analysis` branch, as in the starter notebook.
If you checked out an exact SHA, fetch and check out the new SHA instead.
The dependency definition has not changed; no reinstall or kernel restart is
needed for this script update.

First inspect storage and any already attached checkpoints. This cell does not
download weights or run a model:

```python
import shutil

print("Filesystem free GiB:", round(shutil.disk_usage("/kaggle/working").free / 2**30, 2))
configs = list(Path("/kaggle/input").rglob("config.json"))
print("Attached checkpoint config files:")
for path in configs:
    print(path)
if not configs:
    print("No config.json found under /kaggle/input")

subprocess.run(
    [str(repo / ".venv/bin/python"), "scripts/smoke_llava.py",
     "--preflight-only", "--download-budget-gib", "12",
     "--output", "/kaggle/working/llava-preflight.json"],
    cwd=repo, check=True,
)
```

The script checks Hugging Face file metadata at a pinned model revision. A
`preflight_complete` report means inspection succeeded; inspect `storage.fits`
to see whether downloading is allowed. With a 12 GiB budget and no cached weights,
`fits: false` is expected. Filesystem free space can exceed the notebook's Output
quota, so the script checks both the reported filesystem space and your explicit
budget. The budget must describe the actual cache location; another directory
does not automatically provide more capacity.

Prefer attaching the **HF-format** checkpoint as a Kaggle Input and loading it
directly. The directory must contain `config.json` with `model_type: llava`,
the tokenizer/processor files, and all safetensors weight shards. Original
`liuhaotian/llava-v1.5-7b` files are not interchangeable with this HF conversion.
Do not copy the attached checkpoint into `/kaggle/working`.

Once attached, use the directory containing its `config.json`:

```python
MODEL_PATH = "/kaggle/input/REPLACE-WITH-YOUR-INPUT/checkpoint-directory"
subprocess.run(
    [str(repo / ".venv/bin/python"), "scripts/smoke_llava.py",
     "--model", MODEL_PATH,
     "--output", "/kaggle/working/llava-smoke.json"],
    cwd=repo, check=True,
)
print(Path("/kaggle/working/llava-smoke.json").read_text())
```

Alternatively, on a machine with sufficient space, omit `--model` and explicitly
set `--download-budget-gib` to the remaining capacity at `--cache-dir`. Default
budget is zero; an uncached remote checkpoint will not download accidentally.
The script downloads only safetensors and processor/configuration files at the
pinned revision, caches one copy, and reserves 1 GiB of storage headroom.

The test uses the installed Transformers LLaVA implementation, FP16 weights,
SDPA attention, and automatic GPU placement with 2 GiB VRAM reserved per GPU.
It creates a white image with a red square, asks its color, and saves the answer,
device map, generated token count, peak allocated GPU memory, Git commit, and
lockfile hash. Loading from a local input never downloads weights or falls back
to remote processor files. Record the input's source/revision separately: its
config hash does not fingerprint all weights.

`passed` means generation produced a nonempty answer; inspect whether it answers
"red". One image does not establish accuracy, speed, or ApET correctness.
After this passes, run a small uncompressed MMStar baseline, then introduce
the extracted ApET core and compare it with that baseline.

## Small MMStar baseline after pretrained inference passes

`scripts/eval_mmstar.py` evaluates 24 samples by default: four from each of
[MMStar's six categories](https://github.com/MMStar-Benchmark/MMStar), sampled
with seed 590. It pins the dataset revision and saves the exact row/sample IDs
for a later paired comparison. The ~42 MB `mmstar.parquet` file is downloaded
once and prepared in `/tmp/apet-mmstar-cache`; the duplicate TSV is not fetched.
Although only 24 samples are evaluated, the cache contains the full parquet.

Push the new runner and shared `scripts/llava_runtime.py` helper, then run this
cell in the existing notebook:

```python
import json
import subprocess
from pathlib import Path

repo = Path("/kaggle/working/CSE590-ApET")
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True)
report = Path("/kaggle/working/mmstar-baseline-24.json")
subprocess.run(
    [str(repo / ".venv/bin/python"), "scripts/eval_mmstar.py",
     "--cache-dir", "/tmp/apet-hf-cache",
     "--dataset-cache", "/tmp/apet-mmstar-cache",
     "--per-category", "4", "--seed", "590",
     "--output", str(report)],
    cwd=repo, check=True,
)
result = json.loads(report.read_text())
print(json.dumps(result["summary"], indent=2))
```

No dependency changes or kernel restart are needed. Existing model weights are
reused; the runner refuses to download missing weights. In a fresh session,
run the pretrained smoke setup again first, or supply `--model` with an attached
HF checkpoint directory.

The runner reuses the FP16/SDPA loading and generation helpers from the square
test. It performs one unscored warmup, then generates one answer per image.
Each raw answer, ground truth, parsed prediction, category, latency, and peak
allocated GPU memory is flushed to `mmstar-baseline-24.jsonl`. The JSON summary
records overall/category accuracy, unparsed answers, sample IDs, model/dataset
revision, package versions, Git commit, lockfile hash, and dataset file hash.
If evaluation fails partway through, completed prediction rows remain saved.
Reruns with the same output filename replace those files; use a new filename
when preserving a run.

Answers must be an unambiguous single option letter, optionally with a short
prefix such as `Answer: A`. Longer or ambiguous answers count as incorrect and
are listed as unparsed; raw responses remain available for inspection. This is
an explicitly documented diagnostic scoring rule, not the official MMStar
evaluation pipeline. `passed` means the execution completed, regardless of
accuracy. Neither the 24-sample accuracy nor its timing is a publication result.
Keep this baseline and its selected IDs before enabling ApET.

## Running later scripts

Always launch project scripts through the project Python:

```python
subprocess.run(
    [str(repo / ".venv/bin/python"), "scripts/check_env.py", "--require-cuda"],
    cwd=repo,
    check=True,
)
```

For later experiments, replace `scripts/check_env.py` with their script or module
command. Do not add `.venv/site-packages` to `sys.path`, install into Kaggle's
kernel, or expect `import transformers` in a normal notebook cell to use this
environment. No notebook-kernel restart is needed for these subprocesses.

The bootstrap installs uv at `.runtime/bin/uv`; it does not depend on PATH changes
carrying over between notebook cells. For uv commands after setup, use this
absolute executable path. Experiments may also run directly through
`.venv/bin/python` as above.

The installer explicitly overrides inherited `UV_INSTALL_DIR` settings. If an
older bootstrap reported installation into `/usr/local/bin` followed by a
missing `.runtime/bin/uv`, fetch the bootstrap fix and rerun setup in the same
session; a notebook restart is unnecessary.

## Updating the environment deliberately

Edit `pyproject.toml` and regenerate the lockfile with the pinned uv version on
the Linux runtime (or resolve metadata on the Mac without installing the stack).
The following Linux notebook cell uses the already installed project Python:

```python
subprocess.run(
    [str(repo / ".runtime/bin/uv"), "lock", "--python", str(repo / ".venv/bin/python")],
    cwd=repo,
    check=True,
)
```

Then rerun the bootstrap and validate in a fresh notebook before committing the
updated definition and lockfile. Normal setup uses `uv sync --locked`: it fails
instead of silently rewriting an inconsistent lockfile.

Do not commit `.venv`, `.runtime`, checkpoints, or the installed interpreter.
Keep the runtime report with your experiment outputs. This first candidate
does not enable FlashAttention or quantization; add them only when needed and
validate the resulting lockfile separately.
