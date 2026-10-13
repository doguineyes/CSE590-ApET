# Fixed GQA subsets for CSE590

Use these sets for the main GQA work. Full Test-Dev remains optional final
confirmation. First interrupt the full-run inference cell, wait for its child
process to stop, and run that notebook's ZIP cell. Download partial evidence
before ending the session. A completed prediction remains useful even if the
full evaluation is interrupted.

## The frozen study

| Set | Images | Questions per image | Questions | Use |
|---|---:|---:|---:|---|
| Development | 100 | 8 | 800 | Initial baseline, budget/merge comparisons and tuning |
| Evaluation | 100 | 8 | 800 | Reserved comparisons after freezing a method |
| Diagnostic | 20 | 4 | 80 | Detailed native observations; nested in development |

Development and evaluation have disjoint image IDs **and raw image byte hashes**.
Diagnostic questions are a subset of development questions. The 10 images in
the inspected 40-question pilot are excluded from evaluation; all 10 are in
development and the default diagnostic set. At the default seed, the 40 original
pilot questions are also retained in the diagnostic set.

Selection uses the same two pinned balanced Test-Dev parquet files as the pilot,
about 68 MB total. Seed: **590**. Eligibility requires eight distinct question
texts, query/verify questions and at least two semantic types per image. Image
assignment uses seeded SHA256 ranking; questions use the existing metadata-based
type-diversity selector. No model predictions, correctness, or error cases are
read during selection. All original source metadata and image bytes are kept.

This is an image-balanced, type-diverse project subset, **not** a population-
weighted sample of all GQA questions. Report its ID/counts/type composition with
scores. Reserved evaluation is our split inside labeled Test-Dev, not the official
hidden test set. Keep it out of repeated tuning. A later error-focused diagnostic
subset should be selected only from development results, with preserved-answer
controls and multiple questions about each image; the initial 80 questions are
metadata-selected, not chosen for compression failures.

The root `study.json` records assignments, exact question IDs, source question
hashes, image hashes, source revision, selector hash and study ID. Each set's
manifest records its role, study hash, images, questions, types and seeds.
Evaluation refuses edited or mixed study manifests. Keep the same study for
comparisons; a new selection needs a new directory/version and fresh references.

Default study ID:
`3153bd275dbefd346e132c18ac16d7cbe46e5534c4f9e84fed2786abe038f512`.

## Run development on Kaggle

Commit/push the new files, then use
[the subset notebook](notebooks/kaggle_original_gqa_subsets.ipynb), or copy its
four code cells into the existing notebook after its full-run process exits.
Enable Internet and T4 ×2. The live session can reuse its original environment,
source parquet cache and model weights. Independent sessions restore the locked
environment and may download missing pinned weights again.

In Cell 1 set:

```python
RUN_SUBSET_TEST = True
SUBSET = "development"
CONFIGURATIONS = ["baseline", "input-only", "decoder-only", "combined"]
REFERENCE_REPORT = None
```

Run Cells 1 and 2 first. Check **800 questions, 100 images and 3,200 new predictions**.
Cell 3 loads the native model once across both GPUs and runs all configurations.
Cell 4 summarizes and downloads the ZIP. The 40-question pilot suggests roughly
30 minutes for four configurations on 800 questions; actual time varies with
answer length, question mix and reduced logging. Watch the printed ETA.

No GPU inference was performed locally. CPU tests and real data preparation
passed; pretrained subset execution is the next Kaggle check. This is an initial
baseline for the new frozen set, not a rescore of the selected 40-question pilot.

The notebook runs these standalone commands with the locked original Python:

```python
subprocess.run([
    str(original_python), "-I", "scripts/prepare_gqa_subsets.py",
    "--output-dir", str(study_dir), "--seed", "590", "--no-zip",
], cwd=repo, check=True)

subprocess.run([
    str(original_python), "-I", "scripts/eval_original_gqa_subset.py",
    "--data-manifest", str(study_dir / "development/manifest.json"),
    "--configurations", "baseline", "input-only", "decoder-only", "combined",
    "--download-budget-gib", "20", "--output", str(report_path),
], cwd=repo, check=True)
```

The actual inference notebook cell uses `Popen` and sends SIGINT on interruption,
allowing partial results to be saved before returning. It does not launch two
model processes. Use a fresh output directory for each experiment; rerun the
inference cell to resume its existing report. Native code, dependency locks,
checkpoints, prompt, greedy FP16 settings, 16-token limit, image seeds and
`gqa-short-answer-v1` scoring are unchanged.

## Reuse the saved baseline

Keep the initial development ZIP. For another run on the same manifest, set a
new run directory in Cell 1 and choose, for example:

```python
CONFIGURATIONS = ["combined"]
REFERENCE_REPORT = "/kaggle/working/your-initial-development-run/original-gqa-subset.json"
```

The matching JSONL must remain beside that JSON. After restoring a downloaded
ZIP, point to the extracted JSON under `/kaggle/working`. Do not point to the
40-question pilot or full GQA report: they use different dataset manifests.

Command equivalent:

```python
subprocess.run([
    str(original_python), "-I", "scripts/eval_original_gqa_subset.py",
    "--data-manifest", str(study_dir / "development/manifest.json"),
    "--configurations", "combined", "--reference-report", str(initial_report),
    "--download-budget-gib", "20", "--output", str(new_report),
], cwd=repo, check=True)
```

This generates **800 new predictions**, computes accuracy/APR against 800 saved
baseline predictions, and does not load/generate baseline a second time. It
validates the exact subset, prompt/scoring/inference sources, checkpoint identity,
dependency lock, output limit, record completeness and raw prediction hash.
It copies reference records/source snapshots into the new ZIP. Their images and
question manifest are shared with the current copied dataset. Raw references
are never rescored in place or edited.

Saved reference timings are historical. Repeat a small reference control in the
same session when making controlled timing claims. Current configuration choices
are the four existing native stage settings; new compression algorithms/budgets
still require their own explicit variant implementation and provenance. Native
reference guards remain enforced. Reusing evidence does not authorize silently
changing the original source reference.

## Resume and evidence

JSONL flushes after every prediction. Summary/ETA updates every 50 questions and
on normal exit/interruption. Resume validates source/data/protocol identity,
rebuilds progress from the complete JSONL prefix, and continues within a partial
question. A torn final write is preserved before removal; duplicate, reordered
or corrupt complete records are refused.

The ZIP includes question metadata, selected images, `study.json`, native/project
source snapshots, lock, environment/session provenance, raw predictions, scoring
version and paired/type/image summaries. Package/model caches are excluded.
The report distinguishes current predictions from saved reference predictions.

For a later session, extract the subset ZIP to a new directory under
`/kaggle/working`, set `RESUME_RUN_DIR` to it, and run the notebook. It restores
the recorded Git commit and copied subset manifest. A cached baseline can be
restored from the run's own reference snapshot. Resume requires the same
configuration order; the notebook reads it from the saved report.

An interrupted full Test-Dev ZIP is preserved separately. This first version
does not import its partial predictions into a differently ordered subset run;
we establish a complete, consistent 800-question reference first.

## Diagnostics and reserved evaluation

The existing observed runner accepts the 80-question manifest:

```python
subprocess.run([
    str(original_python), "-I", "scripts/eval_original_gqa.py",
    "--data-manifest", str(study_dir / "diagnostic/manifest.json"),
    "--configurations", "baseline", "input-only", "decoder-only", "combined",
    "--download-budget-gib", "20", "--output", str(diagnostic_report),
], cwd=repo, check=True)
```

Use a new output directory; this observed pilot runner does not support resume
or cached baselines. Keep the study manifest and source snapshots with its
results, as in the existing pilot notebook. This provides native errors, FPS
bases, selected indices and merge records; answer-contribution interventions
remain a separate next implementation step.

After freezing the chosen method, set `SUBSET = "evaluation"` in the subset
notebook and use a new run. Its first run also needs its own baseline; development
baseline predictions cannot be paired with different evaluation questions.

Optional locally prepared bundle:
`research/results/gqa-fixed-subsets-v1-s590.zip`, about 37.4 MB (35.7 MiB),
ignored by Git. Uploading it is unnecessary: Kaggle prepares the identical
sets from the pinned sources. Both manifests and all real images were validated
locally; seven CPU tests check splitting, hashes, cached-reference immutability,
protocol mismatch rejection and interrupted-run recovery.
