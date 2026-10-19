# RQ2 native parity diagnostic review — 2026-10-19

The one-question diagnostic completed all nine passes. It shows run-to-run
numerical variation in the **original native merging path**, including encodes
without the observation wrapper. The first pilot's requirement that separate
native executions have identical feature bytes is therefore unsuitable for this
case. This is not a completed RQ2 pilot and does not justify a silent tolerance.

## Preserved evidence

Copied the supplied files byte-for-byte into `research/results/`, retaining their
Downloads originals:

- `rq2-native-parity-diagnostic-v1-20261019T193300319406Z.zip`:
  60,645,329 bytes, SHA256
  `818798b70ff034cb88f32fb11759a80b236db54aee6af874742557135d5cb8b1`.
- Matching `.log`: 2,881 bytes, SHA256
  `6f098e406c310f79ced995d6c41582f16e0ba39cccd718076037b338a4fdb485`.

All 110 unique members pass CRC; all 109 artifact hashes, 103 evidence hashes,
measurement-source hashes, diagnostic JSON hash and tensor-file hash verify.
The clean runtime commit is `04c2946d42d0c8b1737287489ee18878477395c4`.
All eight native reference hashes and the original environment lock match.
The embedded setup log records **17 prerequisite tests passed**, with no skips.
Wall time is 150.81 seconds, including checkpoint downloads/loading; it is not a
performance benchmark.

The report explicitly has `diagnostic_only=True` and
`scientific_pilot_passed=False`. Its only question is `gqa:201636988`:
“Is the color of the cake different than the plate?”, fixed gold target `No`.

Separate derived files are in
[`results/rq2-native-parity-review-v1-2026-10-19/`](../results/rq2-native-parity-review-v1-2026-10-19/):
verification JSON, exported diagnostic JSON, stage comparisons CSV, reference
scores CSV and setup log. CPU review verifies hashes and exported JSON arithmetic;
it does not load the tensor pickle or recompute GPU-exported tensor deltas.
No local Torch installation or inference was used.

## What the repeated captures establish

All passes keep the same 288 mask coordinates. Observed native repeats and the
generation capture have exactly matching CLIP/source patches, pre-merge tensor,
all E/FPS/selected values, basis placement, merge argument indices and the entire
merge assignment/count/scaling record. Differences first appear after merging.

| Comparison | Stage | Max absolute difference on retained states | Relative L2 on retained states | Changed retained states |
|---|---|---:|---:|---:|
| Raw native 1 vs 2 | Projected | 0.0078125 | 0.0000846573 | 28 / 288 |
| Observed native 1 vs 2 | Before merging | 0 | 0 | 0 / 288 |
| Observed native 1 vs 2 | After merging | 0.0078125 | 0.000144039 | 30 / 288 |
| Observed native 1 vs 3 | After merging | 0.015625 | 0.000155447 | 30 / 288 |
| No-merge 1 vs 2 | Every captured stage | 0 | 0 | 0 / 288 |

The excluded 288 coordinates are exact in all recorded comparisons. Thus the
failure is not merely a hash difference in unused states. The states that reach
the language model vary. All three **unwrapped raw native** projected hashes also
differ: instrumentation is not required for variation to occur.

These observations strongly localize this input's variation to arithmetic within
native merging with unchanged assignments. GPU FP16 accumulation is a candidate,
but the diagnostic does not isolate one specific GPU operation. Instrumentation
may affect execution timing; this test cannot prove that observation has no
effect on a separate execution.

## Answer-score effects and limits

Across the seven native captures (raw, observed and generation), the fixed gold
`No` probability ranges from **0.49214246 to 0.49604755**, a range of **0.39051
percentage points**. The log-probability range is about 0.00790358. No-merge gives
**0.49995327** in both repeats, with identical feature bytes and scores.

The no-merge versus native probability difference is about 0.39057–0.78108
percentage points in these captures. Native variability is therefore comparable
to the smaller observed fixed-score difference. This calls for repeat uncertainty
when interpreting small score effects. It does not imply that all merging effects
are numerical noise: the intentional merge/no-merge projected-feature difference
has relative L2 about 0.294 on retained states, much larger than native jitter.

Only **one generated answer** is recorded in this diagnostic: `Yes`, IDs
`[3869, 2]`. Fixed-reference scores on encoding repeats are not repeated free
generation. There is no new evidence here of an answer flip, or of answer-level
stability across repeats. The earlier failed pilot's No / Yes / No strings remain
diagnostic evidence, not admitted RQ2 rescue/harm results.

This one image does not establish variability on every image, nor explain the
previously deferred GQA question `2058428`. It does not require rerunning RQ1:
this diagnostic has not identified an E/selection discrepancy or an error in the
saved full-context intervention measurements.

## Recommended next design

Continue RQ2 with a **versioned capture/replay protocol**, preserving v1 and its
failed records. Do not rerun the unchanged full v1 pilot: its same-image and
repeat gates also require exact independent merged-feature equality.

1. Capture the first native merge realization and first no-merge realization
   once per image, without inspecting answers or choosing a favorable repeat.
   Keep exact CLIP, E/FPS, selection, pre-merge, basis and mask agreement checks.
2. Reuse those fixed tuples for every same-image question, and for both generated
   answers and teacher-forced scoring through original multimodal preparation.
   Verify exact capture/replay identity, positions, call counts and budget.
3. Separate fixed-tuple replay repeats from independent fresh-native repeats.
   Record fresh feature deltas, score ranges and answer/ID changes on the known
   GQA case and the existing TextVQA controls. Repeats are uncertainty controls,
   not extra evaluated questions. Structural/source changes still require review.

This estimates merge/no-merge effects conditional on an explicitly recorded
native realization; it does not claim native arithmetic is reproducible.
Same-execution capture/replay identity and cross-execution repeatability are
different checks. Exact same-image control can be maintained without inventing
a broad numerical acceptance tolerance or modifying upstream compression math.

The proposed runner/protocol revision is **not implemented in this review**.
No measurement runner, native source, dependency, frozen cohort, reference answer
or previous result was changed. Review a small v2 calibration before its 30-case
panel; another broad numerical investigation is not needed first.

Subsequent update: the second v1 attempt reproduced the failure. The separate
[v2 correction](2026-10-19-rq2-v2-capture-replay.md) now implements this policy;
its seven-question Kaggle calibration remains the next required validation.
