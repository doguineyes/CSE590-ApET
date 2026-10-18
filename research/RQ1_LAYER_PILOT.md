# RQ1 matched input / decoder-layer-16 pilot

This measurement was implemented before the input-validity results were known.
The [validity review](notes/2026-10-17-rq1-validity-review.md) now records
intervention-dependent rankings and grounding concerns on five MMStar diagrams.
Proceed with this small calibration and describe I as stage-specific zeroing
sensitivity. Keep its existing panel unchanged; the five flagged diagrams are
not in this layer panel. Review calibration controls before any full sweep.

The [saved-score reanalysis](notes/2026-10-17-rq1-score-reanalysis.md) is now
complete. The **zero-only** implementation described below remains unchanged.
After its passed calibration, the next requested comparison is the separate
[zeroing / IH extension](RQ1_LAYER_IH_PILOT.md), whose small calibration has now
also passed, followed by the completed
[combined full run](notes/2026-10-17-rq1-layer-ih-full-review.md). Saved
per-answer-token log probabilities permit first/whole scoring summaries later.
This pilot remains exploratory; successful controls do not by themselves justify
a full sweep or an intervention-independent claim about layer effects.

**Status:** the six-question × 32-position calibration passed on T4 ×2, with
zero full-forward control drift and exact repetition of all 192 archived input
effects. See the [calibration review](notes/2026-10-17-rq1-layer16-calibration-review.md).
A bounded zero-only full sweep was ready; the requested joint zeroing/IH
comparison has since completed all 576 positions on that panel. Full-dataset or
whole-decoder conclusions are not established. The notebook launches the
isolated child with `-X utf8` to avoid an ASCII-locale summary-export error.

## First run

1. Push the new files to `apet-analysis`. Finish the currently running validation
   process and download its ZIP/log before updating that live checkout.
2. In a fresh notebook, import
   [kaggle_rq1_layer16_pilot.ipynb](notebooks/kaggle_rq1_layer16_pilot.ipynb).
   Enable Internet and T4 ×2; attach the **same** private `rq1-inputs-v1` dataset.
   No new input ZIP, previous results ZIP or separate TextVQA audit is needed.
3. For a new calibration, set `RUN_LAYER_CALIBRATION = True`. Keep `RUN_FULL_LAYER_SWEEP = False` and
   `INPUT_ZIP = None` for automatic attached-input detection. Run cells in order.
4. Download `rq1-layer16-calibration-v1-<timestamp>.zip` and its companion `.log`.
   Review numerical controls and the input-validity results before the full sweep.

An existing live session reuses the original environment and checkpoint cache.
Each measurement command loads the model once, processes all six questions, and
releases it at subprocess exit. The notebook kernel/environment stays unchanged.
Do not update the checkout or launch a second GPU model while a run is active.

## Frozen panel and positions

The [profile](protocols/rq1_layer16_v1.json) fixes the same six questions already
chosen for the validity checks, before those new results are known:

| Dataset | Questions | Archived input-compression roles |
|---|---|---|
| GQA | 202218829 (Window), 202218593 (Bottom) | Same image; both preserved-correct |
| MMStar | 895 (Bedroom), 136 (Soccer) | Loss, preserved-correct |
| TextVQA | 35962 (Cookery school), 35670 (Dominion) | Loss, preserved-correct |

These are six questions on five images, a curated mechanism panel. The calibration
uses **32 random positions per question at both stages**, independent of E/I.
They are the first 32 of the existing image-hash/seed-590 permutation, shared by
same-image questions. This differs deliberately from the validity check's
post-hoc high-I strata. FPS basis and padding positions remain included/labeled.
There are **384 patch/stage scores**, plus reference, parity and no-op controls.

## What runs

All reference forwards use the unchanged original LLaVA-1.5-7B, FP16/SDPA,
576 visual tokens, and **no actual input or decoder pruning**. Fixed gold targets,
token IDs, prompts, masks, positions and full teacher-forced mean-log-probability
scoring are the same at both stages. EOS is excluded. Wrong fresh baselines or
target misalignment stop the run; there is no retargeting or case substitution.

- **Input:** recompute the existing native input E and singleton post-projector
  zeroing I for the same 32 positions. Previously saved inputs/results are untouched.
- **Layer 16:** hook the output of block 16 (module index 15), before block 17.
  Execute the **original `LlamaModel.layer_prune`** on protected tensor copies
  and an isolated metadata receiver, using FPS 10, epsilon 1e-5 and an observation
  keep budget of 288. Trace its native E/basis/selection/merge values, then discard
  its compressed output. RNG state is restored. That observation never shortens
  or modifies the reference sequence.
- **Layer I:** during a fresh full forward, zero one complete visual hidden-state
  vector at that boundary and let the unchanged downstream model continue.
  No suffix caching or custom decoder implementation is introduced.

Each question requires the existing input repeat/no-op/native-multimodal and
target-prefix controls, then layer observation-only/reference/no-op score parity,
exact repeated boundary-state agreement and a final reference/boundary repeat.
The existing 0.001 per-reference-token log-probability tolerance is fixed.
TextVQA retains its full-teacher-forced/first-token cached-generation policy.
Failed controls stop and preserve partial evidence; no tolerance is auto-increased.

## Interpretation and records

`questions.csv` and the HTML summary compare per-question E–I Spearman rho at
both stages, stage-specific non-basis and common-non-basis subsets, and cross-stage
E/I rank agreement. `tokens.csv` retains every stage/position E, signed I,
perturbed score, zeroing L2 norm, FPS role and photo/padding classification.
Raw question JSONs keep scores, controls, free answers and token mappings.
The ZIP includes native diagnostics, small input and full boundary feature
snapshots, images, source/lock hashes, accepted manifests and frozen protocols.
The console log remains separate. Failures also export a ZIP on normal cleanup.

Input E is in pre-projector CLIP space; layer E is in decoder hidden space.
Compare rankings rather than raw E values. Zeroing norms and the representation
meaning differ, so raw I magnitudes are not calibrated across stages either.
By layer 16, earlier layers have already copied visual information into other
visual/text states. This intervention measures the local remaining contribution
of a state at that boundary, not the total effect of removing its original input
patch. In this causal image-before-question layout, visual states cannot attend
later question/answer tokens; deeper visual E is not automatically question-conditioned.
Same-image visual-feature/E agreement is recorded rather than assumed exact across
different full-forward shapes. Six curated questions cannot establish a population
task/layer effect; do not pool their patches as independent samples.

## Existing live notebook

After the current process has finished, downloaded its ZIP/log and you have pushed:

```python
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True, timeout=120)
subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True)
subprocess.run([str(original_python), "-I", "-c", "import torch; print(torch.__version__)"], check=True)
for test in ["research/tests/test_rq1_layer.py", "research/tests/test_rq1_layer_native.py",
             "research/tests/test_rq1_native.py"]:
    subprocess.run([str(original_python), "-I", test], cwd=repo, check=True)
```

Copy the **`run_layer` helper code cell** from the new notebook into this live
notebook, keeping `repo`, `original_python`, `inputs` and `CHECKPOINT_ARGS`.
Then run `layer_calibration_dir = run_layer()`; this helper launches the new
`scripts/run_rq1_layer.py`, not the older `run_validation` or `run_experiment`.
The equivalent shell command, from the repository root, is:

```bash
environments/original/.venv/bin/python -X utf8 -I scripts/run_rq1_layer.py \
  --inputs /kaggle/working/rq1-inputs-v1 \
  --output /kaggle/working/rq1-layer16-calibration-new \
  --download-budget-gib 20
```

Use the detected `inputs` path if Kaggle mounted the extracted dataset elsewhere.
Every run needs a new output name. Do not install a new environment for this run.

## Later full sweep

Only after reviewing the calibration and validity diagnostics, enable the separate
full-sweep cell. It uses the same six cases and expands the nested positions to
576 at each stage: **6,912 patch/stage scores**. The CLI requires
`--tokens 576 --calibration-report <passed-calibration>/report.json`, checking
the calibration's six-case completion, model/profile/source/lock compatibility.
It does not infer scientific approval from successful controls; interpretability
still depends on reviewing the validity checks. No full sweep starts by default.

The first calibration has now been reviewed and the same-panel zero-only full sweep
passed the execution gate. The newly requested IH extension needs its own
[calibration and review](RQ1_LAYER_IH_PILOT.md) first; the old report cannot unlock
the IH full sweep. For the preserved zero-only workflow, add `"-X", "utf8"` before `"-I"` in the
`run_layer` command, rerun only the helper definition, and call
`run_layer(tokens=576, calibration=layer_calibration_dir)`. The repository notebook
already includes this launch fix. Measurement source hashes are unchanged, so
the passed calibration remains valid; no calibration rerun is required. Allow
roughly 70–90 minutes based on the measured intervention timings. Review the
full results before expanding depth, cohorts or counterfactuals further.
