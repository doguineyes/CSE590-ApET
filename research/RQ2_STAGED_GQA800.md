# RQ2: staged ApET and pure random on GQA development800

Prepare this after the [TextVQA100 comparison](RQ2_STAGED_TEXTVQA100.md).
Review that run before starting the next GPU job. The validated staged adapter,
native model files, original environment, schedules and random streams are reused
unchanged; preparation does not depend on TextVQA's accuracy results.

## Fixed experiment

Use the existing **100 development images ×8 questions =800 questions**.
Keep all questions, including baseline-wrong cases and the flagged repeatability
case2058428. That case stays flagged; this run does not reopen its numerical
debugging. Question IDs, original metadata/types, image bytes, image-dependent
seeds, prompts and `gqa-short-answer-v1` scoring are pinned to the original study.
The runner rejects reserved evaluation and altered/filtered manifests.

For every question run the same nine conditions as TextVQA:

| Condition | Input states | After block16 | Replicates |
|---|---:|---:|---:|
| Uncompressed baseline | 576 | 576 | 1 |
| Native ApET | 96 | 32 | 1 |
| Pure random | 96 | 32 | 101 /202 /303 |
| Native ApET | 288 | 96 | 1 |
| Pure random | 288 | 96 | 101 /202 /303 |

Total **7,200 primary predictions**, plus32 bounded controls for the full800.
A fresh baseline runs alongside every question; old baseline predictions are not
substituted. Model loads once per invocation. Conditions run sequentially.

Capture the first input realization only once per image/condition and share it
across all eight questions, with exact feature/RNG replay. Compute decoder states,
FPS/E and merging separately per question. Random memberships are image/seed-based
at both boundaries. Input replay includes native post-encoding RNG continuation.

## Kaggle

1. Commit/push the new files and import
   [kaggle_rq2_staged_gqa800.ipynb](notebooks/kaggle_rq2_staged_gqa800.ipynb)
   into an editable draft. Enable **Internet and T4 ×2**.
2. Set:

   ```python
   RUN_STAGED_GQA = True
   GQA_SHARDS = [0, 1, 2, 3]  # All800 questions.
   ```

3. Run cells in order: fetch code → original environment and fixed GQA preparation
   → prerequisites → inference → download ZIP/log. **No Dataset upload is needed.**
   Preparation fetches the same pinned source parquet files (about68MB) and
   reconstructs the original study. It reuses a valid existing study directory.
   `DATA_MANIFEST` can point directly to an existing development800 manifest.
4. Download `rq2-staged-gqa800-v1-<timestamp>.zip` and its matching `.log` before
   ending the session. The ZIP includes source/runtime/setup/proof and raw evidence.

The shared study preparer reconstructs development, diagnostic and reserved split
files; the new runner uses development exclusively. Reserved images/questions
receive **no model inference** and are not copied into this experiment's gallery.
The study assignment metadata is preserved for split validation.

The default False flag prevents accidental background inference on import.
Current native and GQA prerequisites must pass with zero skips before pretrained
loading. The compact reference to the passed staged calibration travels with Git;
there is no need to rerun that pretrained calibration or upload its ZIP.

### Shorter sessions

Four prespecified, disjoint parts each contain **25 complete images /200 questions
/1,800 primary predictions**. Their membership was fixed before new outcomes.
Choose `[0]` first to measure runtime, then `[1, 2, 3]`, or run each part in a
separate session. Changing parts does not change the questions or methods.
Each part includes at least one full576 identity control. Additional declared
diagnostic controls are concentrated in part0.

Every invocation uses a fresh output directory. It can run any unique ordered
list of parts0..3, with one model load across the requested parts. Progress prints
per question and gives a rough ETA. Actual broader-run duration is unmeasured.
This version does not resume partway through a failed image or part; preserve
partial evidence and rerun the affected part. Completed separate parts remain
useful. Running a part twice does not create extra independent data.

After downloading separate passed exports, combine them locally:

```bash
environments/editor/.venv/bin/python scripts/summarize_rq2_staged_gqa800.py \
  research/results/your-part0.zip \
  research/results/your-parts123.zip \
  --output research/results/rq2-staged-gqa800-combined-v1
```

The combiner checks full ZIP CRC/hash inventories, input/source/runtime identities,
native calibration contract, raw answer scores and disjoint coverage. It rejects
overlap or incompatible source/runtime/checkpoint identities. It stream-hashes
tensor bytes without loading them. Only coverage of all800 questions is labeled
`complete`; fewer complete parts are `partial-coverage`. Source ZIPs remain intact.

## Results and limits

Export accuracy, APR, paired rescues/losses, per-seed means/ranges and comparisons
at each schedule. Report structural/semantic/detailed task groups. The reference
denominator is the **new baseline's correct answers**. The original development
baseline was552/800; agreement is recorded by later analysis, not an inclusion
threshold or a reason to retry particular questions.

Uncertainty resamples **whole eight-question image groups**, averaging random
seeds within a question first. The100 images,800 questions and three seeds are
not interchangeable sample sizes. Intervals condition on the three seeds and the
chosen image-balanced development cohort; this is not full-GQA population inference.
Whole native versus pure random compares complete procedures and cannot isolate
E selection, basis replacement or merging by itself. No speed claim is made.

Six full576 identity controls,18 exact fixed replays, four forced cache probes
and four fresh whole-native runs are separate from the7,200 primary rows. Fresh
variation is recorded rather than replacing the first primary. Every generation
records cache geometry. Full input/decoder tensors are limited to two diagnostic
questions/images; other cases retain JSON E/basis/merge/membership/fingerprint
records. Input metadata/bytes are stored once per image. The HTML gallery groups
the eight questions under each model canvas.

On interruption, all completed raw arms are retained. Summaries admit only fully
completed matched eight-question image groups; seven finished questions from an
unfinished image remain raw evidence and do not count as an independent image.

No compressor, dependency or original model change is introduced. MMStar
confirmation and any later merge factorial remain later steps. Reserved GQA
evaluation stays outside repeated development comparisons.

## Mac checks

```bash
environments/editor/.venv/bin/python -m unittest \
  research.tests.test_rq2_staged_gqa_protocol \
  research.tests.test_rq2_staged_gqa_runner -v
```

These are CPU orchestration/statistics tests. Actual native fixtures run inside
the locked Kaggle environment; no Torch install is required on the Mac.
