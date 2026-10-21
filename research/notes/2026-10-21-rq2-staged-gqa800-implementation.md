# Staged GQA development800 preparation — 2026-10-21

Prepared the next planned experiment while TextVQA100 runs on Kaggle. This adds
no new GPU observations and does not depend on TextVQA's outcome. Review that run
before launching GQA. Use the [guide](../RQ2_STAGED_GQA800.md) and
[notebook](../notebooks/kaggle_rq2_staged_gqa800.ipynb).

## Frozen design

All800 original development questions on100 images, eight per image; same nine
conditions and schedules as staged TextVQA100: full576, native96→32/native288→96,
and pure random at each schedule with seeds101/202/303. Second-stage compression
is after block16. Full run has7,200 primary predictions plus32 separate controls.
The source question/image identities, original image seeds, question types,
single-word/phrase instruction and `gqa-short-answer-v1` scorer are unchanged.
Keep baseline-wrong questions and flag2058428 without relaunching its numerical
investigation. Every question gets a fresh baseline; no correctness inclusion gate.

Validated the exact existing study and development manifest locally:

- Study ID: `3153bd275dbefd346e132c18ac16d7cbe46e5534c4f9e84fed2786abe038f512`.
- `study.json` SHA-256:
  `81a49ac9a5f534aa43c1de51eb843284e53515d4d46773f933948a4afea3f962`.
- Development manifest SHA-256:
  `99860b9c8768a02a775d515e649234a6d2ea753448e25325ab759978d4c14a24`.
- All100 raw image byte hashes, decoded pixel identities and source metadata
  validate through the existing study loader. Development/evaluation assignments
  remain disjoint; the runner refuses evaluation or edited manifests.

The first input realization is captured once per image/condition and shared
across eight questions, with exact feature and post-input RNG continuation.
Decoder representations and native E/FPS/merge computation remain per question.
The model loads once for all requested image parts. Input metadata/image bytes
are stored once per image; detailed decoder JSON is retained for each question.
Full tensor banks are limited to two declared images/questions:18 files total.

## Execution and statistics

Four fixed disjoint parts each contain25 whole image groups /200 questions /
1,800 primary predictions. Default notebook requests all four. An optional first
part measures runtime before running the remaining three; source selection does
not change with outcomes. Each part has at least one full576 identity control.
Extra fixed replay/cache/fresh-native diagnostics are declared before inference.

The notebook reuses/restores the original locked environment and reconstructs
the same study from pinned GQA sources, about68MB. No Dataset upload is required.
The shared preparer also writes its reserved split files, but only development
receives inference or enters the gallery. Metadata for assignments remains in
evidence for leakage checks. Independent sessions restore their own runtime/cache.

Report accuracy, APR, paired rescues/losses, per-seed values, seed means/ranges
and semantic/structural/detailed groups. Paired95% intervals resample **images**,
after averaging seeds within each question. Eight same-image questions and three
random seeds do not multiply independent sample size. The chosen image-balanced
development subset does not establish full-GQA population accuracy.

The saved-ZIP combiner verifies CRC/artifact inventories, pinned inputs,
calibrated source hashes, source/runtime/checkpoint compatibility and raw answer
scores; overlapping parts are rejected. It stream-hashes tensor bytes without
deserializing them. Only all800 questions are labeled complete. Separate passed
parts can be pooled after downloading; original files remain unchanged.

Interrupted runs preserve all raw arms. Summaries admit only fully completed,
matched eight-question images. An unfinished image's seven passed questions
remain raw evidence. This version reruns an affected part rather than resuming
inside an image; completed separate parts are retained.

## Validation and boundaries

**13 new CPU tests passed**, including a synthetic full800 loop with906 image
captures,7,200 primary rows and32 controls;100 encoding files;18 tensor-save paths;
CRC/hash inventory checks; disjoint shard pooling; real archive-reader/scorer
round-trip checks using a synthetic fixture; overlap, incomplete-image, changed
data/source/runtime, baseline-wrong and notebook-idle guards. Synthetic fixtures
are not pretrained predictions or scientific evidence.

The11 existing staged CPU checks also pass: **24 local CPU tests total**. The Kaggle prerequisite suite contains
39 tests including actual tiny native fixtures, requires zero skips, and binds73
source/test/notebook files to the current runtime before pretrained loading.
The new notebook compiles with cleared outputs/default execution disabled.
No Torch installation, model loading or inference ran on the Mac. Original model
sources, lock, staged adapter, random policy and the running TextVQA execution
sources remain byte-for-byte unchanged.

Broad pretrained GQA results, runtime and export size remain pending. This is a
whole-procedure comparison, not an isolated E or merge ablation. MMStar confirmation
and any later merge factorial remain subsequent steps. No new compressor or
dependency version is introduced; reserved evaluation remains outside tuning.
