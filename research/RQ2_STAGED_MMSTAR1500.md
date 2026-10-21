# RQ2: staged native ApET versus pure random on MMStar

Run this after reviewing the GQA results. This is the third broad comparison,
using **all 1,500 MMStar questions**, with no baseline-correctness filter.
The calibrated adapter, native files, original environment and random streams
stay unchanged. These are generated-answer experiments, separate from RQ1's
teacher-forced intervention measurements.

| Condition | Input states | After block 16 | Replicates |
|---|---:|---:|---:|
| Uncompressed baseline | 576 | 576 | 1 |
| Native ApET | 96 | 32 | 1 |
| Pure random | 96 | 32 | 101 / 202 / 303 |
| Native ApET | 288 | 96 | 1 |
| Pure random | 288 | 96 | 101 / 202 / 303 |

Full execution is **13,500 primary predictions plus 32 controls**. Native calls
the original FPS/E/basis replacement/merging at both boundaries. Pure random
gathers unchanged states without FPS, E, basis replacement or merging. This is
a comparison of complete procedures; it cannot attribute their difference to E
alone. Actual stage counts, source replay and cache geometry are checked.

## Kaggle steps

1. Push the new files, then import
   [kaggle_rq2_staged_mmstar1500.ipynb](notebooks/kaggle_rq2_staged_mmstar1500.ipynb)
   into an editable draft. Enable **Internet and T4 ×2**.
2. In the first code cell set:

   ```python
   RUN_STAGED_MMSTAR = True
   MMSTAR_SHARDS = [0, 1, 2, 3, 4, 5]  # All 1,500 questions.
   ```

3. Run cells in order: code → locked original runtime and inputs → prerequisites
   → inference → download. **No input upload is needed.** The preparer downloads
   the existing pinned MMStar parquet, verifies its SHA-256, and checks every
   question, label, category, raw image, decoded RGB hash and original prompt
   against the previous full baseline's identities before loading weights.
4. Download `rq2-staged-mmstar1500-v1-<timestamp>.zip` and matching `.log`.

The original environment and checkpoint cache are reused within a live session.
Fresh notebooks restore the same locked environment. `INPUTS_DIR` optionally
points to an already prepared directory containing this experiment's manifest.
An existing directory is validated rather than silently overwritten; for an
incomplete preparation, choose a new directory. The default False run flag
prevents accidental inference when Kaggle imports/runs a notebook version.

Prerequisites combine actual tiny native Torch checks with new CPU protocol,
scoring and orchestration checks. Zero skips and current source/runtime proof
are required. The existing passed staged calibration is supplied through its
compact Git reference; no calibration ZIP upload or pretrained rerun is needed.

## Shorter sessions

The six frozen parts each contain **250 questions / 2,250 primary predictions**.
Membership uses image identity, category and index only; no outcomes are used.
Questions sharing identical image bytes remain together. Start with
`MMSTAR_SHARDS = [0]` to observe runtime, then use `[1, 2, 3, 4, 5]` in a fresh
output, or run each part separately. Every part includes a full576 no-op control.
Repeated/fresh/cache controls occur on the predeclared questions 136 and 617.

Model loads once per invocation across its requested parts. The notebook prints
progress and an observed ETA. Broader MMStar runtime is not measured yet; do not
assume it equals earlier four-configuration evaluation. Failed/partial exports
retain raw arms for diagnosis; this version does not resume inside a failed part.
Rerun that part and keep the other passed exports. Do not count duplicate runs
as extra independent observations.

Combine disjoint passed results locally without Torch:

```bash
environments/editor/.venv/bin/python scripts/summarize_rq2_staged_mmstar1500.py \
  research/results/your-part0.zip research/results/your-parts1to5.zip \
  --output research/results/rq2-staged-mmstar1500-combined-v1
```

The combiner stream-verifies every ZIP member's CRC/hash, checks the frozen
inputs and compatible code/lock/packages/checkpoints, rescores raw predictions,
checks question/encoding/control records, rejects overlaps and labels results
complete only with all 1,500 questions. Tensor bytes are hashed, never loaded.

## Scoring, evidence and interpretation

Keep the existing question text, cleaned image placeholder plus option-letter
instruction, Vicuna conversation, sample seed `590 + index`, FP16, greedy cached
generation and 16-token cap. Random membership remains image/replicate/stage
based, exactly as in the calibrated adapter. For repeated images, native input
captures stay per question to preserve the original per-index FPS seed policy;
random memberships use the same image-based plans. Per-question encodings have
separate paths to avoid overwriting repeated-image evidence.

The primary scorer is the existing **project `mmstar-explicit-choice-v2`**:
strict choices or an unambiguous choice followed by its matching option text.
Save strict accuracy, format compliance, unparsed/empty outputs and capped
answers separately. Gold is used for correctness after parsing. Unparsed answers
count as wrong. This is not an official MMStar leaderboard reproduction.

Report accuracy, APR, paired losses/gains, each random seed and its range, and
both category levels. APR's denominator is the fresh baseline's correct answers;
baseline-wrong questions remain in accuracy and gain counts. The old native
baseline contained 521 correct answers under this scorer. New baseline drift is
recorded without filtering questions or retrying selected outcomes.

The 1,500 questions reuse **1,430 distinct image-byte hashes**. Paired intervals
resample those image groups and compute a question-weighted ratio of sampled
score sums to sampled question counts. Averaging random seeds happens within a
question first. Seeds and same-image questions are not independent extra samples.
Intervals condition on the fixed seeds and dataset; no controlled speed claim.

Each ZIP includes source/runtime/setup/prerequisite snapshots, raw predictions,
all question-level encoding/selection/merge/cache records, original/model-canvas
images, CSV, category summaries, paired intervals and an HTML gallery. Full
tensors are limited to two diagnostic questions (18 tensor files in the full
run). Six identity controls, 18 fixed replays, four forced cache probes and four
fresh whole-native repeats stay outside primary denominators. Fresh variation
is recorded and never replaces the first native realization.

Implementation is prepared and CPU-checked. Full pretrained execution and its
scientific results remain pending Kaggle.
