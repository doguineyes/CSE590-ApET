# November 5: review and first mechanism test

## Assessment

The direction is useful and fits the remaining project: explain the task-dependent
native-versus-random gap before proposing an improvement. RQ1 established bounded,
operator-dependent sensitivity results, not a general irrelevance of E. RQ2's broad
staged comparison established task differences; its small selected component panels
do not yet explain why. TextVQA is the appropriate first mechanism cohort.

Checked the archived staged TextVQA100 ZIP: status passed, 100 questions, 900 primary
predictions, 965.15 seconds. It provides exact historical random masks, predictions,
E and routing records; full raw features were only saved for selected cases.
The existing TextVQA source records contain **no OCR boxes**.

## Corrections to the planning note

- The actual historical seeds are 101/202/303. Extend to 404,…,2020 using the same
  mask RNG namespace. Repeating old seeds is a reproducibility check, not new samples.
- Evidence coverage is not yet measured. Obtain verified OCR annotations and audit
  coordinate mapping; do not equate normalized pixel values near zero with padding.
- Random masks also alter non-answer content. Within-question association is useful
  evidence but not proof of mediation. Zero explicit-region coverage need not imply
  zero information, because both CLIP and decoder features are contextualized.
- Basis replacement puts different content at retained positions. Native coverage
  must distinguish slot, basis source and merged source; naive slot overlap is wrong.
- Low-E padding can still enter via basis selection/placement; “never kept” is a
  hypothesis, not a guarantee. Ridge regularization also means the ordinary orthogonal
  projection identity for squared error is not exactly the implemented calculation.
- For future Experiment 2, excluding basis indices from top-K changes native selection
  in cases where native top-K would include them. Do not silently call that arm
  unchanged native. Check slot/content identity and keep an exact native reference.
- Image-state question invariance requires an identical causal prefix, image, stochastic
  choices and preprocessing; changed prompts before the image invalidate that check.

## Implemented now

`research/notebooks/kaggle_textvqa_seed_coverage.ipynb`:
three-question calibration (four seeds) followed by explicitly enabled full100
(20 seeds), unchanged original model/environment and existing staged adapter.
New scoped policy does not edit the archived policies or `llava/` implementation.

Saves raw normalized pixels, native pre-merge CLIP patch features, native 96-state
layer16 features, E/FPS/merge diagnostics, seed masks at both stages, generated answers,
scoring, historical drift, source/lock/runtime evidence and checksums. This covers
TextVQA preparation and Experiment 1 acquisition. GQA dumping, OCR mapping and the
selector×compensation grid are not implemented in this change. No cohort extension.

Review the calibration before the large run. Exact masks, budgets, input source
identity, frozen replay, identity576 and cache controls are execution requirements;
answer correctness is never a pass criterion. Old-seed prediction drift is retained
and reviewed rather than retried until a match.

## Local verification

Editor environment: 24 tests run, 23 passed; the actual Torch/native test skipped
because the local reading environment lacks Torch. Five new CPU tests cover all300
archived masks, new masks/counts/mapping, scoped restoration, invalid policies and
calibration gating. Notebook/Python syntax checked. Kaggle prerequisites require
all six new tests (including actual tiny native generation) and existing staged
prerequisites with zero skips. Pretrained calibration has not yet run.

See `research/TEXTVQA_SEED_COVERAGE.md` for the exact run sequence and limitations.
