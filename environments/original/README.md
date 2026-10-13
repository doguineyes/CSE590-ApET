# Original ApET LLaVA: first Kaggle smoke

This candidate environment runs the repository's original `llava/` implementation,
including input and decoder compression. It uses no extracted compression core or
modern HF LLaVA adapter. Kaggle execution still needs to be verified.

Python 3.10.18 matches the original README. Torch 2.1.2 + CUDA 12.1,
torchvision 0.16.2, Transformers 4.37.2, Accelerate 0.21.0, NumPy 1.26.4, and
Pillow 10.4.0 match the original dependency stack. The environment includes only
the dependencies needed for this inference test, not the original web demo,
training, or optional LoRA/quantization integrations. Protobuf 3.20.3 is a
compatibility choice for the older tokenizer; all transitive versions are locked.

## 1. Push and open a fresh Kaggle notebook

Commit the new environment, bootstrap, smoke scripts, tests, documentation, and
notebook; push to GitHub. Open
[`kaggle_original_apet_smoke.ipynb`](../../research/notebooks/kaggle_original_apet_smoke.ipynb)
in Kaggle. Enable **Internet** and **GPU T4 x2**.

A fresh session avoids retaining old model processes and installing both large
Torch environments into the same notebook output. Existing notebooks can also
pull the update and run this setup, if sufficient disk and GPU memory are free.

The notebook clones `apet-analysis`; change its `REPO_REF` if you pushed elsewhere.
If you already have the checkout, update it with `git pull --ff-only` before running
the new scripts. Record an exact commit for later formal experiments.

## 2. Install and check the original environment

The notebook launches this command through `subprocess.run` from the checkout:

```bash
bash bootstrap/setup_original_linux.sh \
  --require-cuda --expected-gpus 2 \
  --output /kaggle/working/original-runtime-smoke.json
```

The script installs pinned uv and Python in `.runtime/`, syncs this project's own
lockfile into `environments/original/.venv/`, and checks native imports, a tiny
random-weight original decoder with cached generation, and CUDA math on each GPU.
No pretrained checkpoint is downloaded at this step.

The notebook kernel's Python and packages are unaffected. Always launch subsequent
original tests with **`environments/original/.venv/bin/python`**. No kernel restart
or shell activation is needed. Repeating setup reuses the locked environment.

## 3. Preflight the pretrained checkpoints

```bash
environments/original/.venv/bin/python -I scripts/smoke_original_apet.py \
  --preflight-only \
  --download-budget-gib 20 \
  --output /kaggle/working/original-apet-preflight.json
```

This queries metadata and cache availability; it does not download model weights.
Inspect `storage.fits` before proceeding. Both the LLaVA weights and the separately
loaded CLIP vision model are included in the budget. Default cache storage is
`/tmp/apet-original-hf-cache`, separate from notebook output storage; verify that
the location has enough free space. The budget is an explicit limit, not a promise
about a Kaggle session's capacity. Temporary caches may disappear after a session.

The test pins these upstream revisions:

- [Original-format LLaVA-1.5-7B](https://huggingface.co/liuhaotian/llava-v1.5-7b):
  `4481d270cc22fd5c4d1bb5df129622006ccd9234`.
- [CLIP ViT-L/14-336](https://huggingface.co/openai/clip-vit-large-patch14-336):
  `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`.

The previously downloaded `llava-hf/llava-1.5-7b-hf` checkpoint is a different
format and cannot be reused directly by the original model loader. To use attached
original-format weights, supply `--model /kaggle/input/your-checkpoint-directory`.
An attached CLIP directory can be selected with `--vision-model` too. Local file
provenance must be recorded separately.

## 4. Run one image, baseline and combined ApET

```bash
environments/original/.venv/bin/python -I scripts/smoke_original_apet.py \
  --download-budget-gib 20 \
  --configurations baseline combined \
  --output /kaggle/working/original-apet-smoke.json
```

The script creates a red-square image and asks its color, loads the original model
once, and runs FP16 greedy generation with:

| Configuration | Input visual tokens | Decoder compression |
|---|---:|---|
| `baseline` | 576 | None |
| `combined` | 288 | Reduce to 96 after layer 16 |

Both configurations use the same original decoder class, checkpoint, image,
prompt, seed, and GPU placement. The smoke uses SDPA without FlashAttention2,
allows no CPU/disk model offload, and requires at least four output tokens to
exercise cached decoding. It records actual input/decoder token reductions by
observing the original methods. Compression math and original algorithm files
are unchanged. The only checkpoint configuration override points the vision
tower at the separately pinned local CLIP snapshot.

Inspect the answers: red is expected for this synthetic image. `passed` means
nonempty generation completed and requested token reductions occurred. It does
not establish dataset accuracy, paper reproduction, or a performance benchmark.
The output contains package versions, Git state, native source hashes, checkpoint
revisions, prompt/image provenance, settings, traces, timing, and GPU allocation.

Optional: exercise the two other stage combinations on the same image:

```bash
environments/original/.venv/bin/python -I scripts/smoke_original_apet.py \
  --configurations baseline input-only decoder-only combined \
  --output /kaggle/working/original-apet-four-stages.json
```

The default download budget is zero, so this optional command requires the pinned
checkpoints to be cached already. `input-only` retains 288 tokens without decoder
compression; `decoder-only` keeps 576 initially and reduces to 96 after layer 16.

## Results and next step

Download `original-runtime-smoke.json`, `original-apet-preflight.json`, and
`original-apet-smoke.json` (or the first failure report), plus the generated PNG.
Share them for review. Exceptions and partial completed runs are saved; the
notebook cell also displays the subprocess failure. If failure occurs during
installation, share the cell's install output because no Python report exists yet.

This milestone establishes environment and execution feasibility. Dataset runners,
MMStar/GQA/TextVQA baselines, formal scoring/APR, and RQ1 intervention records are
the following steps. The current small records can be expanded as needed.

Code-only updates use `git pull --ff-only` followed by a new project-Python
subprocess. Dependency edits require updating this `pyproject.toml` and its own
`uv.lock`, then rerunning setup; avoid root `uv sync` for original ApET.

The standalone check and inference scripts explicitly use Matplotlib's `Agg`
backend. Kaggle's kernel can otherwise pass `MPLBACKEND=module://matplotlib_inline.backend_inline`
to a subprocess whose environment does not include the notebook plotting backend.
For this code-only fix, pull the update and rerun the inference cell; no package
reinstall or kernel restart is needed. Checkpoints already downloaded in the same
session are reused.
