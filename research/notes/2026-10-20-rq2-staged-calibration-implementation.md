# Staged ApET / random calibration implemented — 2026-10-20

The user agreed to prioritize a faithful native staged comparison before broad
input-only64 evaluation. The [current RQ2 plan](2026-10-20-rq2-revised-evaluation-plan.md)
now records this order. The previously prepared input-only TextVQA100 runner is
retained as an optional, deferred component experiment; no results are discarded.

## Implemented calibration

The [new notebook and guide](../RQ2_STAGED_CALIBRATION.md) use full576 and native
96→32/288→96, compared with three pure-random seeds at each identical schedule.
The second reduction is after block16. Six selected questions produce54 primary
answers; the model loads once. Conditions run sequentially in the original locked
environment. There is no broad staged runner or merge factorial yet.

The panel is GQA cake/plate and stove, MMStar soccer136, and TextVQA Brolio,
Dominion and Cookery school. The new protocol deliberately uses the clean soccer
control instead of the input-only calibration's already-flagged chart1014. Old
frozen case sets and results remain unchanged. These selected questions support
execution checks and qualitative observations, not a population method ranking.

Native input and decoder paths call the original functions, including independently
computed E/FPS/basis/merging at each stage. Pure random uses original projected
states and a selection mask at input, then gathers surviving contextual states
at layer16. It executes no FPS/E/basis insertion/merge. Original forward, cache
updates and position handling remain unchanged. Counts are matched by boundary;
equal FLOPs or runtime are not assumed. This is not a full paper reproduction.

## Important implementation detail: RNG continuation

Replaying input features skips the original FPS draw. Without restoring the RNG
state **after** that draw, decoder FPS could use a different basis from a direct
native run even under the same seed. The new input snapshot records/replays both
features and post-input CPU/CUDA RNG states. Decoder clone replays similarly
preserve source/output/RNG identities. Random membership uses independent Python
RNG streams derived from image, replicate and boundary, without consuming Torch RNG.

First input realizations are shared across same-image questions. Each question
captures its own first decoder output, preventing a question's text states from
being replaced by another question's full sequence. Frozen repeats are exact
controls; four fresh whole native runs on Brolio are separate variation records,
never answer-based replacement attempts.

## Controls and export

- Six full576 no-op controls require the same generated IDs, processed scores and
  answers as the uncompressed model.
- Eighteen fixed replays cover all nine conditions on cake/plate and Brolio.
- Four forced four-token soccer probes verify first/seventeenth block prefill
  geometry and three cached decoding steps at both schedules, native/random101.
- Four fresh whole-native Brolio runs use no feature replay or replacement prune.
  CUDA variation is recorded; no compressed-answer accuracy threshold determines a pass.
- All54 primary predictions use the existing scorers. TextVQA preserves all ten
  answers and partial credit. Repeats/seeds do not inflate question counts.

Exports contain raw answers, native E/FPS/basis/merge records, random memberships,
source/output/RNG fingerprints, cache traces, images/model canvases, CSV/HTML and
source/lock/runtime/test evidence. Input tensor banks are saved once per image;
decoder tensors are limited to soccer and Brolio. Failure exports retain partial
evidence. Original archives and all eight pinned native source files are unchanged.

## Validation and handoff

Eleven new CPU tests passed locally (23 including relevant existing random and
native-gate CPU regression checks): cohort/schedule and random policies,
partial-credit/seed denominators, cache faults, same-image input reuse, source
gates, exact no-op IDs/scores, stale prerequisites, failed archive preservation
and idle notebook import. The accepted IH bundle also validated all six question/
image/seed identities across five images. A simulated six-question run verified
54 primary predictions,86 total generations including separate controls, five
shared image banks,16 declared decoder snapshot saves, CSV preservation of the
multiline MMStar question, and all ZIP hashes/CRC. This mocked execution validates
recording/export only; it is not native-model or research evidence.
Actual tiny18-block native generation tests are included
for Kaggle, covering direct versus observed native execution, RNG continuation,
random guards, no-op/cache behavior, decoder replay and nonvisual preservation.
Pretrained and native Torch validation remain pending; no Torch is installed on Mac.

Push code, attach the existing **rq1-ih-inputs-v1** Dataset, and use the new staged
notebook with `RUN_STAGED_CALIBRATION = True`. All prerequisites must pass with
zero skips before pretrained loading. Download the timestamped ZIP and matching
log. Review calibration before broader staged TextVQA/GQA/MMStar evaluation.
