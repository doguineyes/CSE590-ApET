# RQ2 pure random calibration implementation — 2026-10-20

The first step of the revised RQ2 plan is ready for Kaggle. This is an
implementation checkpoint, not a pretrained result.

New project-owned files provide a separate frozen protocol, budget-aware native
capture/replay helper, six-question runner and
[notebook](../notebooks/kaggle_rq2_random_calibration.ipynb). See the
[run guide](../RQ2_PURE_RANDOM_CALIBRATION.md).

The primary calibration has six questions / five images and 66 predictions:
one full576 reference; native input288/input64, each with/without merging; and
pure random at both budgets with replicates101/202/303. Decoder pruning is off.
GQA contributes a same-image question pair, MMStar its existing soccer case,
and TextVQA Brolio, Dominion and Cookery school. The original accepted
`rq1-ih-inputs-v1.zip` is reused unchanged.

Pure random samples uniformly over all576 original positions without replacement,
with a per-image/replicate permutation shared across budgets. It runs only the
uncompressed native CLIP/projector branch, then applies a random selection mask.
It computes no FPS, reconstruction errors, basis replacement or merging. All576
states are projected before selection; this implementation checks answer behavior
and is not an optimized random-overhead benchmark.

Native64 still uses ten basis representations. Native generation, multimodal
insertion/compact positions, scorer and original environment are retained. The
new helper temporarily adds a named input-budget configuration and restores the
configuration table, model metadata and RNG even on exceptions. It does not
rewrite native compression. Old RQ2 helper files/profiles remain untouched.

Controls include exact full576 pure-random tuple/answer/score parity, archived
source and baseline identity, paired native selectors/basis assignment, nested
random masks, frozen clone replay, same-image reuse, fixed replay repeats and
separate fresh native numerical repeats on Brolio at each budget. A wrong
compressed answer does not fail calibration. No favorable retry is selected.

Exports preserve predictions, scores, images/canvases, controls, source/runtime
evidence and hashes, plus timestamped ZIP and console log. Native tensor banks
are stored once per image; all pure masks share the byte-identical full576 feature
tensor. Fresh native repeat tensors are preserved separately. Failed/interrupted
runs retain their available evidence and cannot claim calibration success.

Local verification:

- Ten CPU protocol/orchestration tests passed, covering nested/isolate-RNG
  sampling, matched seed summaries, source/selection rejection, same-image bank
  reuse, import/prerequisite guards, existing-output preservation and failed-run ZIP export.
- New Python modules and every notebook code cell compile. CLI help works.
- The existing35.7MB portable bundle passes accepted evidence/image/reference
  validation and yields exactly the intended six questions / five images.
- All eight pinned native model sources match the original reference; the
  original lock and older experiment protocols are unchanged.

Six additional native tests require the original Torch environment. They exercise
the actual native64 implementation, forbid FPS/solve during pure capture, check
restoration and tuple mutation rejection, and run tiny native generation/scoring
including full576 parity. The notebook runs these and CPU/source prerequisites
with zero allowed skips before loading pretrained weights.

Next: push, attach the same private Dataset in the new editable notebook, enable
Internet/T4×2, set `RUN_RANDOM_CALIBRATION = True`, and send the ZIP/log for review.
Only after this calibration should broader TextVQA100 evaluation be implemented.
