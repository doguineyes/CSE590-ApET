# RQ2 TextVQA100 comparison implemented — 2026-10-20

**Priority updated:** defer this input-only runner as an optional ablation. The
[staged calibration](../RQ2_STAGED_CALIBRATION.md) now comes first, followed by a
broader matched-schedule comparison. The implementation and local checks below
remain valid; they do not mean the input-only100 run should be the next GPU job.

The six-question pure-random calibration established real64-token execution,
but cannot establish a native/random ranking on a broader pool. The next run
uses the existing100-question TextVQA screening cohort, retaining every question
regardless of the previous uncompressed or compressed answer.

The separate [guide and notebook](../RQ2_TEXTVQA100.md) run full576 once, then
native input288/64 with and without merging and pure random at three fixed seeds
for both budgets. There are1,100 primary predictions, with no decoder pruning.
Random permutations and capture/replay use the already calibrated helpers, so
shared-image masks agree with the earlier experiment. Native files, environment
lock, previous profiles and all original results remain unchanged.

## Measurement and cohort boundaries

- Primary: native ten-annotator consensus and full-credit preservation against
  the fresh full576 answers. All100 questions include baseline-wrong and partial
  answers; the original historical baseline has44 full-credit cases.
- Secondary: the94 questions outside the six previously studied TextVQA cases,
  and those six separately. Neither is an independent held-out dataset.
- Paired seed-specific gains/losses, mean/range and bootstrap score intervals use
  question/image units. A seed-averaged random comparison still has100 units.
- New generated answers are checked against the archived100-case screening
  baseline/native288 responses. Drift is recorded without retries or exclusions.
- No new I measurement or teacher-forced reference target is introduced. RQ2 is
  scored by generated answers, avoiding an unnecessary100-case target audit.

Within-run source, E/FPS, paired merge/no-merge selection, pure-random original
feature identity, nested membership and input-count/position traces stay exact.
Full576 generation parity is tested on six questions; all11 conditions are
replayed on the first screening question and Brolio. Four additional fresh
native Brolio captures record numeric/answer drift without replacing the primary.
The older diagnostic's expected288 display default is explicitly set to the
actual budget in these new fresh64/288 comparisons; old archives are not edited.

Large tensors are saved only for Brolio and Dominion and Brolio's fresh controls.
Every question still saves predictions, scores, source/output fingerprints,
selected positions, native E/FPS/basis/merge graphs, original image and actual
model canvas. The export includes a100-case gallery, paired metrics, historical
comparisons, source/lock identities, runtime/test evidence and partial-failure
records. This is not an optimized-overhead or latency benchmark.

## Validation and next execution

Thirteen new CPU tests passed locally (23 including the existing random CPU
regression tests), covering unfiltered cohort identity,
partial-credit/zero-denominator metrics, no seed pseudo-replication, paired
intervals, drift reporting, source gates, stale prerequisite rejection, archive
failure preservation and idle notebook import. All100 local image/question/seed/
reference identities were validated against the previous manifest and responses.
A synthetic 1,100-row export also verified the 100/94-case summaries, historical
49.7%/44-full-credit baseline arithmetic, CSV/gallery creation, ZIP CRC and every
artifact hash. This is export validation, not new model evidence.
Two actual tiny native generation tests are included for Kaggle, alongside the
previous26 prerequisites. Torch/model execution is not attempted on the Intel Mac.

Push code, attach the existing **rq1-ih-inputs-v1** Dataset, set
`RUN_TEXTVQA100 = True` in the new notebook and run its cells. No new upload or
six-case calibration rerun is required. A live-session cell in the guide reuses
the environment/checkpoint cache. Review the resulting ZIP/log before implementing
the GQA800 and MMStar broad comparisons from the revised plan.
