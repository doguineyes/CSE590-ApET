# Original ApET: full GQA balanced Test-Dev

For routine CSE590 experiments, use the [fixed subsets](GQA_SUBSETS.md).
Full Test-Dev is optional final confirmation. If a full run is already active,
interrupt its inference cell and download its partial ZIP before switching.

The [GQA diagnostic pilot review](notes/2026-10-13-original-gqa-pilot.md) passed
on 2026-10-13. The full version is ready to push and run; no native, dependency
or scoring fixes were needed. Do not start this alongside another model process. Use the
[full-run notebook](notebooks/kaggle_original_gqa_full.ipynb), or copy its cells
into the live Kaggle notebook after downloading the earlier results.

## Scope and fixed protocol

**12,578 labeled questions × four configurations = 50,312 scored predictions**
on all 398 Test-Dev images. This is full **balanced Test-Dev**, not all GQA training
questions or the hidden test set. It follows the development split recommended by
the [GQA evaluation page](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html).

The two pinned source parquet files are the same ~68 MB used by the pilot:
[LMMs-Lab GQA at a fixed revision](https://huggingface.co/datasets/lmms-lab-encoder/GQA/tree/a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8).
All source question IDs, metadata and raw image bytes are retained. Unlike the
pilot, preparation performs no type filtering, sampling or question-text
deduplication. Questions per image vary.

The native LLaVA model, existing device-mask compatibility patch, locked original
environment, checkpoint revisions, Vicuna prompt, greedy FP16 generation,
16-token maximum and image-based FPS seed remain unchanged. Each question/image
pair is run under these four configurations:

| Configuration | Input visual tokens | Decoder pruning |
|---|---:|---|
| baseline | 576 | None |
| input-only | 288 | None |
| decoder-only | 576 | Layer 16 → 96 tokens |
| combined | 288 | Layer 16 → 96 tokens |

These are our experimental settings, not every paper configuration. Our two new
scripts handle preparation and evaluation; they do not implement ApET's algorithm.
Native hashes must match `research/provenance/original_llava_reference.json`.

## Records and scoring

The runner loads one model across both T4s per invocation. It saves and flushes a
JSONL record after every prediction: question ID, image ID/hash, raw response,
normalized prediction, gold answer, question types, prompt, seed, settings,
token-count traces, output length, generation time, peak allocated memory and
session ID. The copied manifest retains full source annotations and semantics.
Native/project source snapshots, environment provenance and the lock hash are
included automatically in the run folder.

Scoring remains `gqa-short-answer-v1`: exact answer match after whitespace
trimming, lowercasing and trailing-period removal, as in the pilot. Empty answers
count wrong. Literal matching is recorded separately. Prose and synonyms are not
automatically converted into short answers. These are **project accuracy/APR
results**, not a claim to reproduce all official GQA metrics or the ApET paper.
Raw answers allow later rescoring without inference.

Summaries include question-weighted accuracy, APR on baseline-correct paired
questions, lost/gained question IDs, semantic/structural/detailed type results,
per-image results, empty responses, generation lengths/times and memory peaks.
Partial summaries use only completed predictions/pairs. The full runner does not
record reconstruction vectors or merge arrays; retain the diagnostic pilot ZIP
for those. Timing includes the existing token-count hooks, with varying output
lengths, and is not a controlled speed benchmark.

## Running and saving time

The pilot's scoring, native logging checks and generation behavior have been
reviewed. Set `RUN_FULL_TEST = True` in notebook Cell 1. Internet and **T4 ×2** are
required. The default flag prevents accidental full execution during notebook
import. The notebook prepares the full data itself; no dataset attachment is
necessary. Existing environment, source parquet cache and model weights are
reused within the live session. Dependencies and weights are not shared
automatically between independent notebook sessions.

Setup and data preparation are CPU operations. The four configurations produce
over eight times the predictions of full MMStar. The 40-question pilot took
91.323 scored seconds; direct extrapolation gives approximately **7–8 hours**
for 12,578 questions. Allow roughly eight hours, with actual runtime dependent
on the full question mix and reduced logging overhead. This is an estimate,
not a guaranteed completion time. Progress prints an updated remaining-time estimate every 100
questions. Model loading and warmups are separate from scored session wall time.
Scored JSONL is flushed per prediction; summary JSON updates every 100 questions
and on normal exit/interruption.

Run command used by Cell 3:

```python
subprocess.run([
    str(original_python), "-I", "scripts/eval_original_gqa_full.py",
    "--data-manifest", str(data_manifest),
    "--configurations", "baseline", "input-only", "decoder-only", "combined",
    "--max-new-tokens", "16", "--download-budget-gib", "20",
    "--report-every", "100", *checkpoint_args, *resume_args,
    "--output", str(report_path),
], cwd=repo, check=True)
```

Download the ZIP made by Cell 4 before stopping the session, even after a failure
or interruption. It includes source, manifest, images, report and predictions;
it excludes installed packages and model weights. The summary may lag the JSONL
after a forced process termination; resume reconstructs it from the saved lines.

## Resume instead of repeating completed predictions

In the same session, rerun Cell 3: it adds `--resume` when its existing report is
present. It validates the complete prediction prefix and skips those predictions.
One interrupted final JSONL write is saved to a `.partial-*` file before removing
it; corrupt complete lines, duplicates, foreign IDs and changed scores stop the
run for inspection. A file lock blocks concurrent writers to the same report.

For a later session, attach your downloaded ZIP, extract its contents into a
new directory under `/kaggle/working`, and set `RESUME_RUN_DIR` to that directory
in Cell 1. The directory must contain `original-gqa-full.json`, its JSONL, the
`original-gqa-full-data/` folder and `source/`. The notebook checks out the
original report's Git commit and uses its copied data manifest. Setup restores
the locked environment; missing pinned weights may download again. Uploading a
ZIP alone does not restore installed packages or weight caches.

Resume refuses changed data, inference/scoring code, native hashes, dependency
lock, checkpoint identity, configuration order or generation settings. For local
checkpoint paths it hashes the actual weight/metadata files, adding CPU/I/O time.
Each load has separate device/driver/package provenance and warmup/wall timing;
resuming is not a guarantee of identical kernels across different hardware.
Use a new run directory for an intentional protocol change.

Local verification covers actual full data decoding/coverage and CPU orchestration,
including interrupted recovery. Pretrained native inference remains a Kaggle check.
