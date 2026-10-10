# Candidate runtime v1: fresh Kaggle smoke test

The first validated session and its results are recorded in
[the October 9, 2026 research note](notes/2026-10-09-kaggle-apet.md).

For local code navigation on an Intel Mac, use the separate
[editor environment](../environments/editor/README.md).

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
The loader explicitly casts the parquet's binary `image` column to the Datasets
`Image` feature so rows decode into Pillow images. It validates all selected
images on CPU before loading the model. A raw `bytes` image passed directly to
generation would otherwise fail with `'bytes' object has no attribute 'convert'`.

After pulling this decoding fix, run the CPU regression tests in the project
environment, then rerun the evaluation cell below. This needs neither CUDA
operations nor extra downloads:

```python
subprocess.run(
    [str(repo / ".venv/bin/python"), "research/tests/test_mmstar_images.py", "-v"],
    cwd=repo, check=True,
)
```

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

## Extracted input-stage ApET and paired comparison

The first independent module is [apet_compression](../apet_compression/README.md).
It preserves the original LLaVA input-stage FPS, approximation errors, basis-slot
replacement, token merging, and spatial order. The HF adapter compresses the
vision features before projection and sends 96 retained visual tokens to the
language model. Decoder-layer pruning is still disabled. This is a first
input-stage experiment, not a reproduction of the paper's complete ApET setup.

Neither `pyproject.toml` nor `uv.lock` changes. After committing and pushing the
new files, continue the same notebook where the 24-sample baseline passed:

```python
import json
import subprocess
from pathlib import Path

repo = Path("/kaggle/working/CSE590-ApET")
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True)
subprocess.run(
    [str(repo / ".venv/bin/python"), "scripts/run_apet_comparison.py",
     "--baseline", "/kaggle/working/mmstar-baseline-24.json",
     "--cache-dir", "/tmp/apet-hf-cache",
     "--dataset-cache", "/tmp/apet-mmstar-cache",
     "--keep-tokens", "96", "--basis-tokens", "10",
     "--output-dir", "/kaggle/working"],
    cwd=repo, check=True,
)
print(Path("/kaggle/working/apet-comparison.json").read_text())
```

The convenience runner performs these steps sequentially and stops on failure:

1. Run original-source parity tests on CPU and each visible GPU, and tiny random
   HF LLaVA generation tests. These require CUDA in this runner and download no
   checkpoint. The original functions are executed through AST extraction;
   the legacy LLaVA/Qwen packages are never imported.
2. Run the same 24 samples with the adapter retaining all 576 visual tokens.
   Compare with the saved baseline and require every raw response to match.
   This identity gate catches unintended integration changes before compression.
3. Run those same samples with 96 retained tokens and 10 FPS basis vectors.
   The first unscored warmup compares actual vision features with the original
   input-stage implementation. Per-image FPS uses `590 + sample index`, so the
   warmup and sample iteration have the same deterministic selection.
4. Save paired accuracy changes, gained/lost correct sample IDs, raw unparsed
   answers, and diagnostic timing. Report/model/dataset provenance must match.

Output files are `mmstar-apet-identity-24.json`/`.jsonl`,
`mmstar-apet-input-96-24.json`/`.jsonl`, and `apet-comparison.json`.
The original baseline remains available. Weights and dataset cache are reused;
the evaluator refuses a missing-weight download. Preserve all these result files
before ending the session. Rerunning the convenience runner replaces its own
identity/compressed reports, so change `--output-dir` to retain another run.

If the original baseline files are missing, first rerun the uncompressed
`scripts/eval_mmstar.py` cell above. In a fresh session the runtime and temporary
model cache must also be prepared again. Keep the same checkpoint, dataset,
sample IDs, max token count, and answer parser for the comparison.

For individual runs, use `scripts/eval_mmstar.py --apet --keep-tokens 96
--basis-tokens 10 --compare-to <baseline.json> --output <new-report.json>` with
the same cache arguments. Omitting `--apet` runs the uncompressed baseline.
`scripts/compare_mmstar.py <baseline.json> <new-report.json>` can also inspect
two completed reports and their prediction files without loading any model.

The adapter targets Transformers 4.48.2, a single image, and batch size one.
It uses removable hooks on the loaded projector, then normal HF generation with
a shorter placeholder sequence. Installed model libraries and legacy model files
are not edited. FPS/merging and generation statistics add diagnostic overhead;
the 24-sample result establishes behavior and a paired comparison, not a paper
accuracy or speedup claim.

## Full MMStar workload and runtime comparison

After the 24-image identity and original-feature gates pass, run all 1,500 MMStar
images with the baseline and input-stage ApET. The pinned dataset has six
categories with 250 images each, so `--per-category 250` selects every row.
Keep the current runtime, checkpoint, prompt, strict parser, 16-token limit,
batch size one, and seed unchanged. Full coverage still uses our diagnostic
scoring; it is not the official MMStar evaluation protocol.

Push the evaluator's timing/progress update first, then run this cell in the
existing T4 ×2 notebook. Model and dataset downloads are reused from `/tmp`.
The two subprocesses run sequentially and stop on failure:

```python
import json
import subprocess
from pathlib import Path

repo = Path("/kaggle/working/CSE590-ApET")
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True)
output_dir = Path("/kaggle/working/mmstar-full")
output_dir.mkdir(exist_ok=True)
baseline = output_dir / "mmstar-baseline-1500.json"
common = [str(repo / ".venv/bin/python"), "scripts/eval_mmstar.py",
          "--cache-dir", "/tmp/apet-hf-cache",
          "--dataset-cache", "/tmp/apet-mmstar-cache",
          "--per-category", "250", "--seed", "590",
          "--max-new-tokens", "16", "--log-every", "50"]

for name, extra in [
    ("mmstar-baseline-1500.json", []),
    ("mmstar-apet-input-96-1500.json",
     ["--apet", "--keep-tokens", "96", "--basis-tokens", "10",
      "--compare-to", str(baseline)]),
]:
    output = output_dir / name
    subprocess.run(common + extra + ["--output", str(output)], cwd=repo, check=True)
    result = json.loads(output.read_text())
    print(name, "status:", result["status"])
    print(json.dumps({"summary": result["summary"], "timing": result["timing"]}, indent=2))
    if "comparison" in result:
        print(json.dumps(result["comparison"], indent=2))
```

There is no bootstrap, dependency reinstall, or kernel restart for this update.
Allow time to finish and download outputs before ending the session; the
24-image median is only a rough planning guide. Every 50 completed answers the
runner prints elapsed time and a rough remaining-time estimate. Image sizes,
question lengths, answer lengths, and later categories can change that estimate.
It also checkpoints an atomic JSON progress report; every prediction row is
flushed immediately. A normal Ctrl-C caught by the subprocess saves an
`interrupted` report. A forcibly killed process may leave `running` status and a
checkpoint lagging behind the JSONL file. Only `passed` reports are complete and
eligible for paired comparison. Rerunning replaces the outputs; there is no
automatic resume. Use a new output directory to preserve another run.

New summary fields include total/mean/median/p95/max generation latency,
generated-token counts, and the highest peak allocated memory on each GPU.
Timing fields separate preparation, model loading, unscored warmup, scored
evaluation wall time, and total script wall time. Generation latency includes
vision encoding, compression when enabled, LLM prefill and cached decode, with
all GPUs synchronized; it excludes CPU processing and report writes. Scored
wall time includes CPU processing and logging. These are single-request workload
measurements; they do not measure concurrent serving capacity. Allocated memory
does not include all driver/caching overhead and is not Kaggle output disk use.

The paired comparison also records generation-time totals, answer-token counts,
and whether hardware/device placement matched. An observed timing ratio can
reflect different answer lengths as well as compression. A later controlled
benchmark should fix output length and repeat measurements before claiming a
general speedup. Keep unparsed raw answers for review: `A: explanatory text`
remains incorrect under the same strict parser as the 24-image run. Do not
change that rule for only one side of a comparison.

If both full runs finish with time to spare, run 288-token input compression
against the **same full baseline** for another accuracy/runtime tradeoff point:

```python
subprocess.run(
    common + ["--apet", "--keep-tokens", "288", "--basis-tokens", "10",
              "--compare-to", str(baseline),
              "--output", str(output_dir / "mmstar-apet-input-288-1500.json")],
    cwd=repo, check=True,
)
```

Preserve all JSON and JSONL files, including raw predictions. This optional cell
packages the results together as a small download in Kaggle's Output pane:

```python
import zipfile

archive = Path("/kaggle/working/mmstar-full-results.zip")
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(output_dir.iterdir()):
        if path.suffix in {".json", ".jsonl"}:
            bundle.write(path, arcname=path.name)
print(archive)
```

## Fixed-output timing benchmark

After the full 288-token accuracy run, use `scripts/benchmark_apet.py` to measure
the same images at 576, 288, and 96 visual tokens with exactly 32 and 128
generated tokens. This answers how compression's timing benefit changes when
decoding occupies more of the request. There are no dependency changes.

Commit/push the benchmark update and pull it in the current notebook. Run the
tiny-model regression test first: it deliberately makes ordinary generation
stop after one EOS token, then requires four tokens with the benchmark's length
control. It downloads no model and checks the new generation behavior:

```python
import subprocess
from pathlib import Path

repo = Path("/kaggle/working/CSE590-ApET")
python = str(repo / ".venv/bin/python")
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True)
subprocess.run(
    [python, "research/tests/test_apet_extraction.py", "AdapterTests", "--require-cuda", "-v"],
    cwd=repo, check=True,
)
```

Run the full 288-token comparison using the cell in the preceding section if
it has not already completed. Preserve the full baseline JSON and JSONL in
`/kaggle/working/mmstar-full`; the 24-image baseline cannot be used for that run.
Then start the fixed-output benchmark:

```python
import json

output = Path("/kaggle/working/mmstar-full/apet-fixed-output.json")
output.parent.mkdir(exist_ok=True)
subprocess.run(
    [python, "scripts/benchmark_apet.py",
     "--per-category", "2", "--repeats", "3", "--seed", "590",
     "--visual-tokens", "576", "288", "96",
     "--output-tokens", "32", "128", "--basis-tokens", "10",
     "--output", str(output)],
    cwd=repo, check=True,
)
result = json.loads(output.read_text())
print(result["status"])
print(json.dumps(result["summary"], indent=2))
```

The defaults select 12 balanced images (two from each category), three repeats,
three visual budgets, and two output lengths: **216 measured requests**, plus
six unscored warmups. The model is loaded once and keeps the same device map
for every condition. Every image/length/repeat forms a paired block; blocks are
shuffled with the fixed seed, and the first visual budget rotates between blocks
to reduce ordering effects. The per-image FPS seed stays constant across
repetitions. One-time original-feature parity runs during compression warmup.

`min_new_tokens` equals `max_new_tokens`; early EOS is suppressed until the
requested length, and each request must actually produce that token count.
The original MMStar prompt is retained to control the prefill workload. Forced
continuations can be repetitive or meaningless, so the benchmark records **no
accuracy score**. Its length controls are never applied to `eval_mmstar.py`.
See the [pinned Transformers generation parameters](https://huggingface.co/docs/transformers/v4.48.2/en/main_classes/text_generation).

Each output-length group contains latency summaries, total generation time,
peak allocated GPU memory, effective token throughput, and a paired timing ratio
against the 576-token baseline. Throughput includes vision and prefill; it is
not an isolated decode rate. Generation timing synchronizes all GPUs and includes
compression/diagnostic overhead, but excludes CPU preprocessing, result decoding,
statistics, and file writes. This benchmark does not separately measure prefill,
time to first token, concurrent requests, or training.

Completed requests are flushed to `apet-fixed-output.jsonl`, and a JSON progress
checkpoint is saved every six requests. The remaining-time estimate is rough
because the order mixes 32- and 128-token outputs. Caught interruption saves an
`interrupted` report; forced termination can leave `running` status. Partial
summaries report how many trials have matching baseline measurements, and only
those pairs contribute to the timing ratio. Require `passed` and all 216
requests for the completed default benchmark. Reruns replace these two files;
choose another output filename to retain another run.

For a shorter pilot, use `--per-category 1 --repeats 1`: 36 measured requests.
Use the progress estimates and remaining GPU quota to decide whether to start
the default run before the session deadline. The ZIP cell above includes the
new benchmark JSON and JSONL automatically. Preserve them before ending the
session.

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
