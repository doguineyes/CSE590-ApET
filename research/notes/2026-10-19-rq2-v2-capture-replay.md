# RQ2 v2 capture/replay correction — 2026-10-19

The second v1 attempt reproduced the known failure. The diagnostic did not change
v1's separate-execution exact feature gate. It had collected the evidence needed
to design a different execution policy. Both failed v1 attempts remain preserved.

## Second failed attempt

Saved from Downloads, retaining the originals:

- `rq2-input-merge-pilot-v1-20261019T195332472383Z.zip`: 2,225,470 bytes,
  SHA256 `d316f171cf8d97a184590588c9f04c28cc78c153cb20f5b0901350b5b2cc5dc1`.
- Matching `.log`: 2,319 bytes,
  SHA256 `9e38bad7bb135e71f740cde2623e8ac36cdc3d42d425b0f856d5e66c0017099d`.

ZIP CRC and artifact hashes verify. Runtime commit is
`04c2946d42d0c8b1737287489ee18878477395c4`, wall time 15.3067 seconds.
The first question `gqa:201636988` fails `features_exact` again, with zero admitted
completed questions. Masks and generated token IDs match; projected feature
fingerprints differ. This is the same gate as the earlier failure, consistent
with the intervening [nine-pass diagnostic](2026-10-19-rq2-native-parity-diagnostic-review.md).

## Implemented correction

New v2 files preserve the v1 runner, protocol and artifacts. The native ApET
files, dependency lock, weights, 30 questions/24 images, prompts, generation
settings, target strings and scorers remain unchanged.

- Capture the first primary baseline/native/no-merge encoding once per image,
  before inspecting generated answers. No selecting, averaging or replacing a
  realization to obtain a preferred outcome.
- Require exact source/CLIP/E/FPS/selection/pre-merge/basis/mask agreement. Keep
  no-merge identity and explicit native ownership checks.
- Save actual first feature tuples before answer generation. Reuse exact clones
  for every same-image question and both generation and teacher-forced scoring.
  Validate pixels, devices, budget, one visual prefill, returned feature/mask
  bytes and immutability within that realization.
- Repeat fixed tuples for the known GQA case and four existing TextVQA controls;
  those repeats still require exact encoding, generated IDs and reference scores.
- On those controls, independently capture two more native realizations. Record
  feature deltas, score changes and generated-answer/ID changes. Numerical drift
  is recorded; source/E/selection/basis/merge-graph changes still fail. These
  repeats are not additional evaluated questions and cannot replace primary
  predictions.

This is an execution/measurement correction conditional on a recorded native
realization, not a deterministic rewrite of native merging. It does not establish
that observation is timing-neutral or that a separate native execution reproduces
the same floating-point bytes. No broad numerical acceptance tolerance is added.

## Run sequence and evidence

The new [v2 notebook](../notebooks/kaggle_rq2_input_merge_v2.ipynb) first runs seven
predeclared questions across all three datasets: the first GQA same-image pair,
MMStar313, TextVQA37533/35670/35962/36001. All fresh-repeat controls are included.
Both notebook flags default to False. The full30 requires the actual completed
calibration report, detailed hashed questions/features/evidence, source/profile,
native/runtime and checkpoint identity. The old diagnostic or v1 report cannot
satisfy the new gate. Full runs capture new first realizations, with calibration
kept as a separate execution, not an answer-matching constraint.

All first image realizations and fresh control tensors are archived. The full
ZIP embeds its calibration prerequisite evidence. Failures retain partial files
and a ZIP; the notebook separately streams the console and labels a fallback ZIP
if the runner cannot produce one. Preserve both result ZIP and `.log`.

[Run instructions](../RQ2_INPUT_MERGE_V2.md) cover a new notebook and a same-live-
session helper update without reinstalling the environment. Input remains the
existing `rq1-ih-inputs-v1` Dataset.

## Validation

34 CPU checks passed across old and new RQ2 modules. New tests exercise versioned
policy rejection, accepted identities, same-image bank reuse, full calibration
artifact/control requirements, exact fixed-replay policy, dispatch, failure ZIPs
and overwrite protection. Synthetic CPU fixtures are explicitly not model results.
Three Torch-dependent classes skip locally; the new five native tests exercise
actual native encoding and tiny decoder scoring/generation, mutation/count gates,
and independent drift versus structural changes on Kaggle.

All notebook code cells compile without saved outputs. Disabled execution flags
stop before setup. Eight monitored native hashes and the original runtime lock
match. No pretrained v2 result is claimed until Kaggle tests and calibration pass.
