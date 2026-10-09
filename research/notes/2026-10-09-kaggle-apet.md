# ApET research session — October 9, 2026

Today established a reproducible Kaggle execution environment, verified the
Mac → GitHub → Kaggle development workflow, extracted the LLaVA input-stage
ApET compression core, and completed full MMStar evaluations and a controlled
timing benchmark on two Tesla T4 GPUs.

## Development and environment decisions

- **Intel Mac / VSCode:** read and edit code, use Git, and run lightweight
  checks that do not require the ML stack.
- **GitHub:** transfer versioned code, environment definitions, and experiment
  instructions between machines.
- **Kaggle CPU/GPU:** execute ML tests and initial experiments using the project
  environment. Avoid building a second ML environment on the Mac.
- **Google Cloud later:** validate the same Linux environment on the target GPU
  before larger experiments. No Cloud environment was deployed today.

We stopped relying on manually downgrading Kaggle's notebook packages and
restarting its kernel. The repository now defines a separate Linux x86_64
environment with `pyproject.toml`, `.python-version`, and `uv.lock`.
`bootstrap/setup_linux.sh` installs pinned uv and Python, runs `uv sync --locked`,
and checks the resulting runtime.

Versions successfully used today:

| Component | Version |
|---|---|
| uv | 0.9.5 |
| Python | 3.11.14 |
| PyTorch | 2.9.0+cu128 |
| torchvision | 0.24.0+cu128 |
| Transformers | 4.48.2 |
| Accelerate | 1.15.0 |
| datasets | 4.8.5 |
| NumPy | 2.4.6 |
| Pillow | 12.3.0 |

These are the installed versions; the committed lockfile also fixes the
dependencies declared as ranges in `pyproject.toml`.

Always launch experiments through **`.venv/bin/python`**. Kaggle's normal
notebook kernel remains a different environment. Code-only updates use
`git pull --ff-only` and a new project-Python subprocess; they need no dependency
reinstall or notebook restart.

Future library upgrades should update the environment definition and lockfile,
then repeat validation in a fresh Kaggle session. The current HF adapter explicitly
requires Transformers 4.48.2, so upgrading Transformers also requires checking
and updating its integration. uv controls the Python environment; host drivers,
GPU hardware, and available disk still need runtime verification.

## Problems resolved and checks completed

1. Fixed uv installation: inherited Kaggle installer settings initially placed
   uv in `/usr/local/bin`, while the bootstrap expected `.runtime/bin/uv`.
   The bootstrap now explicitly controls the install destination and checks it.
2. Passed environment checks, tiny random LLaVA generation, FP16 CUDA operations,
   and reconstruction checks on both T4s.
3. Ran pretrained HF LLaVA on a red-square image; it answered **“Red.”**
4. Fixed MMStar image decoding: loading its parquet directly exposed binary
   image bytes. The evaluator explicitly declares the dataset's image feature
   and validates decoded images before loading the model.
5. Passed the balanced 24-image baseline and extracted-core comparison. Keeping
   all 576 visual tokens preserved every raw answer in the identity check.
6. Passed original-source numerical parity checks within the configured
   relative/absolute tolerances. Tiny-model tests checked cached generation,
   adapter-hook cleanup, and fixed-output length control.

The final local test suite had **24 passing tests and 8 skipped ML-dependent
tests**. ML validation was performed with the Kaggle project environment.

## What was extracted

[`apet_compression`](../../apet_compression/README.md) provides a core that
imports PyTorch without importing model code, plus a separate HF LLaVA adapter.
The core preserves the original input-stage FPS basis selection, regularized
reconstruction, approximation-error ranking, basis placement, token merging,
and spatial ordering. Experiments used **10 basis tokens**, epsilon **1e-5**,
merging enabled, and private per-image FPS seeds of `590 + sample index`.

The adapter uses removable hooks on the loaded model's projector and shortens
its image-placeholder sequence. Installed Transformers files and the original
legacy model directories were not modified. The checkout package is importable;
it has not yet become a separately installed or published distribution.

**Scope:** LLaVA input-stage compression, one image and batch size one. The
vision encoder and full-grid projection still run. Decoder-layer pruning, Qwen,
training, and batched generation remain future work. Original-feature parity
under today's runtime is narrower than reproducing the complete legacy model
or the paper's full method.

## Full MMStar results

Model: `llava-hf/llava-1.5-7b-hf`, FP16, SDPA, weights distributed across two T4s.
The GPUs share model layers through `device_map="auto"`; this is not two
independent model replicas processing two images concurrently.
Dataset: all **1,500 MMStar samples**, six categories with 250 samples each.
All three evaluations used the same questions and ground truths, greedy
generation, batch size one, and a maximum of 16 new tokens. Answers averaged
approximately two generated tokens.

| Visual tokens | Correct | Accuracy | Mean generation/image | Total generation |
|---|---:|---:|---:|---:|
| 576 — baseline | 509 / 1,500 | 33.93% | 0.593 s | 14.82 min |
| 288 — input ApET | 506 / 1,500 | 33.73% | 0.381 s | 9.52 min |
| 96 — input ApET | 492 / 1,500 | 32.80% | 0.246 s | 6.14 min |

Generation timings synchronize all GPUs and include vision encoding,
compression when enabled, LLM prefill, and cached decoding. They exclude CPU
preprocessing, loading, and result writes. The 288-token script's total wall
time was **609.597 seconds, approximately 10.2 minutes**.

- **288 tokens:** 1.56× observed generation-time ratio, or 35.7% less generation
  time, with a 0.20 percentage-point accuracy decrease. There were 43 gained
  correct answers and 46 lost correct answers relative to the baseline.
- **96 tokens:** 2.41× observed generation-time ratio, with a 1.13 percentage-point
  accuracy decrease. There were 49 gains and 66 losses.
- Peak allocated memory stayed around **7 GiB per GPU**. Compression changes
  sequence processing while model weights remain loaded; this implementation
  showed a clearer time benefit than a memory benefit.

Scores use our conservative answer parser: unparsed responses count as
incorrect. Unparsed counts were 12, 8, and 2 for 576, 288, and 96 tokens.
Responses such as `A: explanatory text` can therefore affect the comparison.
These are diagnostic scores, using input-stage ApET only. `passed` means a run
completed its execution and validation checks; it is not an accuracy threshold
or an official MMStar/paper reproduction claim.

## Fixed-output timing benchmark

[`scripts/benchmark_apet.py`](../../scripts/benchmark_apet.py) loaded the model
once and held its GPU placement constant. It used **12 balanced images × 3
repetitions × 3 visual budgets × 2 output lengths = 216 measured requests**,
plus six unscored warmups. Paired image/length/repeat blocks were shuffled with
a fixed seed; visual-budget order rotated within each block.

The runner set equal minimum and maximum new-token counts and verified every
request's actual length. These forced continuations were used for timing and
were not scored for accuracy.

| Fixed output | Baseline mean | 288-token mean / ratio | 96-token mean / ratio |
|---|---:|---:|---:|
| 32 tokens | 2.451 s | 2.180 s / 1.12× | 2.029 s / 1.21× |
| 128 tokens | 8.479 s | 8.005 s / 1.06× | 7.775 s / 1.09× |

All six conditions had **36 measured requests and 36 baseline pairs**. The
benchmark completed in **1,157.786 seconds, approximately 19.3 minutes**.
Recomputing its summaries from the JSONL reproduced the reported values within
negligible floating-point rounding differences.

Interpretation: compression helped the short-answer evaluation substantially;
its relative benefit decreased with longer outputs. This is consistent with
decoding occupying a larger share of total time. Separate prefill/decode
measurements are needed to confirm that explanation. These tests measure
sequential inference, not concurrent serving capacity or training performance.

## Reproduction and result archive

The 288-token and fixed-output runs recorded:

- Git commit: `d35f426d8096fe84ac602cadc6c8a96c212a1552`.
- Lockfile SHA-256: `6825f43a39a544e02205cad425b085c2d4242d766706ca7ebd6520cfd9c77e63`.
- Model revision: `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`.
- MMStar revision: `bc98d668301da7b14f648724866e57302778ab27`.
- Dataset parquet SHA-256: `29afd74b0134cfab083a8909b5358577ab18fd41c1e612031577cfb3635531c2`.

Kaggle used `/kaggle/working/CSE590-ApET` for the checkout, with environment
files under `.venv` and `.runtime`. The Output meter was around 7.3 GiB after
environment setup. The roughly 13.2 GiB checkpoint was cached separately under
`/tmp/apet-hf-cache`; dataset files used `/tmp/apet-mmstar-cache`. These temporary
caches may disappear when a session ends. GPU memory and output disk are
separate resources.

Preserve these full-run exports together, each with **both JSON and JSONL**:

- `mmstar-baseline-1500`
- `mmstar-apet-input-288-1500`
- `mmstar-apet-input-96-1500`
- `apet-fixed-output`

Also retain `runtime-smoke.json`, the pretrained smoke result, and the 24-image
baseline/identity/compression reports including `apet-comparison.json`.
The raw exports are external experiment artifacts; this note summarizes their
review and does not bundle copies. The earlier baseline and 96-token JSONL files
were reviewed; include their JSON summaries as well for the complete shared
archive. [`research/KAGGLE_RUNTIME.md`](../KAGGLE_RUNTIME.md) contains the setup,
execution, update, and ZIP-download cells.

## Next session

1. Archive today's exports and commit this note. Keep today's environment and
   baseline as the reference for subsequent code changes.
2. Inspect gained/lost answers and unparsed responses, especially perception
   failures. Apply any revised scoring rule to every compared run and label
   rescored results separately.
3. Repeat 288- and 96-token experiments with several FPS seeds to assess
   sensitivity. The small net difference at 288 tokens is promising evidence
   from one run, rather than established accuracy equivalence.
4. Measure prefill and decoding separately, then improve one compression choice
   at a time. Consider additional token budgets and core ablations.
5. Add decoder-layer pruning, other model adapters, or deliberate library
   upgrades as separate changes with renewed validation. Validate Google Cloud
   when larger runs justify it.

Today's completed milestone: a working environment and development loop, an
independent first compression core, and measured accuracy/runtime tradeoffs on
Kaggle T4 ×2. The session ended after these experiments; no further GPU run is
needed to complete today's record.
