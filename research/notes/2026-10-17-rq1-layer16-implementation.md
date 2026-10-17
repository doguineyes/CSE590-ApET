# RQ1 layer-16 implementation — 2026-10-17

The input-validity checks are running on Kaggle. Their outcomes are not required
to implement the next measurement. We fixed the same six-question panel before
seeing those results and added a matched input / decoder-block-16 runner. Review
the validity findings and the small layer calibration before expanding or making
a scientific claim about layer effects.

The calibration is six questions on five images, with 32 seeded random patch
positions per question at both stages (384 patch/stage scores). GQA is the Window /
Bottom same-image pair, both archived input-preserved; MMStar and TextVQA each
include a loss and preserved control. The order is independent of prior E/I and
nested in the full-grid permutation. This is a diagnostic panel, not a random
population estimate. No new dataset ZIP or environment is introduced.

Project hooks observe the native block-16 output (module index 15), then zero one
visual state during an otherwise unchanged full forward. Native `layer_prune`
executes on protected tensors/isolated metadata to obtain E/FPS/selection/merge
diagnostics; its compressed output is discarded and RNG restored. All actual
reference forwards retain 576 visual tokens and have decoder pruning disabled.
Native source files and the original uv lock remain unchanged.

The runner repeats the existing input reference/target controls and adds layer
observation-only/no-op parity, exact repeated boundary-state checks and a final
reference. Failed controls stop without retargeting or case replacement. A future
576-position sweep requires the matching passed six-case calibration report.
The notebook starts disabled and saves ZIP/log evidence using the existing
GitHub → Kaggle workflow. See [instructions](../RQ1_LAYER_PILOT.md).

Interpretation: E spaces and zeroing norms differ across stages. By layer 16,
other states may already carry information from the input patch, so local layer
I is not the total causal effect of that original patch. Visual decoder states
precede the question and cannot attend later tokens. Layer E is not automatically
question-conditioned. Compare within-question rankings with these limits stated.

CPU protocol/export checks and compilation are validated locally; real Torch
boundary tests run as Kaggle prerequisites. Pretrained calibration is pending.
Do not claim the layer runner has passed on two T4 GPUs before that run returns.
