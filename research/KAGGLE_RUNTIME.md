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
