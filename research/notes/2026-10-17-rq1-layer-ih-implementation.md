# RQ1 layer-16 IH comparison — implementation checkpoint

The user requested a small IH comparison before committing GPU time to the
zero-only full sweep. The previously passed six-question × 32-position zeroing
calibration and its raw ZIP/log remain unchanged. No conclusion about the new
IH results is assumed. The new calibration is the next run; the full sweep is
deferred until it is reviewed.

**Later checkpoint:** the new calibration has now passed. The
[saved-results review](2026-10-17-rq1-layer-ih-calibration-review.md) records zero
control drift, 384 exact zeroing repeats and 192 verified IH measurements.
The statements below describe the earlier implementation/validation boundary.

Implementation and complete run instructions are in
[RQ1_LAYER_IH_PILOT.md](../RQ1_LAYER_IH_PILOT.md). Use the same frozen panel,
positions, targets, input dataset, original uv environment and T4 ×2. Existing
zeroing code is reused byte-for-byte, with a new entry point/profile adding IH
only at the block-16 output boundary. No additional libraries are installed.

## Method/source distinction

Inspected the official repository at commit
`75909b2936a13d7214e83514f0bc0cb9cc91139e`, including its calculation scripts,
native pruning loop, retained-position policy and first-step logit export.
Four full source-file hashes and pinned URLs are recorded in the new protocol.
The source-only inspection checkout is in `/tmp`; no upstream package or model
fork is imported by the project.

The paper's section 3.2 uses a binary-mask equation. The released code physically
deletes the other visual states and preserves original retained position IDs.
Its example scripts use 2×2 groups. We implement that released operator using
its supported 1×1 setting so both methods compare individual positions. This,
our gold rendering, the same full teacher-forced forwards and the retained native
SDPA kernel are documented adaptations, not a complete IH reproduction.

Our original whole-answer zeroing score is preserved. Comparable outputs use
first-target-token probability units: `p(full)−p(zero-one)` versus
`p(retain-one)−p(remove-all-at-layer)`. No EOS, altered labels, retargeting or case
replacement is introduced. The first reference tokenizer piece may not be a
whole word. The all-absent-at-layer reference preserves text states that already
processed the image; it is not a question-only baseline.

Two distinctions remain confounded in the comparison: full context versus
retain-only context, and zeroing versus deletion. Differences cannot establish
which distinction causes them, nor identify intrinsic token importance. A bridge
experiment would be separate work if the results warrant it. The six curated
cases cannot establish an information-horizon curve or population layer effects.

## Native compatibility and safeguards

Temporary downstream pre-hooks shorten states before block 17, crop the unpadded
causal mask, and pass original retained position IDs through all later blocks.
Old native ApET slices RoPE tables by current sequence length, so a temporary
RoPE pre-hook extends only the lookup extent to the original length; the native
values/weights/position IDs stay unchanged. This prevents gapped position IDs
from indexing a shortened table. Hooks restore even on failure. Native source
files, layer/pruning configuration, model weights and dependency lock are untouched.

Require unchanged upstream boundary states, keep-all no-op parity, repeated
remove-all scores, a retained-case repeat, and final full-reference/boundary
checks, in addition to every existing zeroing control. The 0.001 tolerance is
unchanged; no automatic tolerance widening. The runner preserves partial ZIP/log
on handled failure and source evidence before loading weights.

The full gate requires the **new** passed IH calibration with matching six-case
completion, method/profile/source/native/lock/model evidence. The old zero-only
report is rejected. All full-run flags remain false by default.

## Validation boundary

Seven CPU protocol, probability-unit/sign, frozen-panel, full-gate, UTF-8 export,
failure-preservation and notebook-launch tests pass in the existing editor
environment. The existing nine layer protocol and six saved-review tests also
pass (22 CPU tests total). Python and notebook cells compile. Previously
calibrated zeroing measurement sources remain byte-identical; the old archived
calibration still satisfies its original source gate.
Native integration tests use an actual tiny original LLaVA and an independently
executed native suffix oracle to check deletion, text/position preservation,
RoPE lookup, no-op parity and hook cleanup. Those tests are **Kaggle prerequisites**;
local Mac Torch execution is unavailable. Successful GPU results are not yet
claimed. The notebook explicitly imports Torch before running native tests so
Kaggle cannot silently treat a missing runtime as success.

The small run adds 192 IH measurements to 384 existing zeroing measurements.
The optional same-panel full sweep has 10,368 patch/method scores. Runtime
estimates are provisional until the new calibration is reviewed. Stop expanding
at this bounded comparison before returning to RQ2.
