# Staged TextVQA100 implementation — 2026-10-21

Implemented the next step after the passed staged calibration: all100 fixed
TextVQA screening questions, baseline576 and native96→32/native288→96 versus
three matched pure-random seeds at each schedule. Decoder compression remains
after block16. There are900 primary predictions and32 separate control runs.

Use [the new guide](../RQ2_STAGED_TEXTVQA100.md) and
[notebook](../notebooks/kaggle_rq2_staged_textvqa100.ipynb). After pushing, attach
the same private `rq1-ih-inputs-v1` Dataset, enable Internet/T4×2 and set
`RUN_STAGED_TEXTVQA100 = True`. Run cells in order and download ZIP plus log.
No new input upload or pretrained six-question calibration rerun is needed.

## Scope and provenance

The existing native files, original lock, staged input/decoder adapter and random
policy are unchanged. New code handles the100-question loop, controls, compact
evidence and scoring/statistics. It reuses the calibrated capture/replay/RNG/cache
path. A compact calibration reference is distributed with Git, pinned to the
reviewed ZIP/report hashes and original45 measurement-source hashes. It is a
project-owned summary pointing to separately preserved raw evidence; it is not a
replacement GPU run. Current prerequisites additionally bind the broad runner,
protocol, tests, notebook and guide to the runtime before pretrained loading.

The full existing TextVQA manifest, questions,100 images, all ten annotator
references, seeds and historical baseline scores validate locally. All questions
remain included, irrespective of their historical or new baseline result. Fresh
baselines are primary; old input-only answers are not substituted for staged
conditions. Historical baseline drift is recorded without retargeting/retries.

Primary summaries keep TextVQA consensus credit, full-credit preservation (APR),
gains/losses, per-seed values, random seed means/ranges and paired score differences
with95% percentile intervals. Random seeds average within questions before
resampling. Resampling units are question/image, not prediction or seed. Report
all100 and the six previously studied cases separately from their94-case
complement. This complement is still development data, not reserved evaluation.

Six identity controls,18 fixed replays, four forced cache probes and four fresh
whole-native Brolio runs are excluded from summary denominators. Fresh variation
is recorded; separate native CUDA runs need not be byte-identical. Primary clone
replays retain exact source/RNG/output checks. No compressed-accuracy threshold
determines execution success.

Full input/decoder tensors are saved only for Brolio and Dominion: two input
banks plus16 decoder snapshots,18 files total. Input metadata is saved once per
image with hashed references in answer rows. Every primary row retains decoder
diagnostics and cache traces. Raw JSONL/CSV, image gallery, source/lock/setup/proof
and file hashes remain in the ZIP. Partial exports preserve unfinished raw arms
but summarize only complete matched questions; reruns use a new output path.

## Validation and limits

- **14 new CPU tests passed**, covering frozen cohort/stage/source drift,
  baseline-wrong inclusion, partial credit/undefined APR, seed denominators,
  94-case complement, historical baseline scope, complete and partial exports,
  source/runtime prerequisite binding and idle notebook imports.
- A synthetic full100 orchestration test exercises900 primary rows and32
  controls,100 encoding files and18 declared tensor-save paths, validates every
  artifact hash/CRC, and confirms the900 CSV records. Its mock states and answers
  are test fixtures only, not research observations or pretrained validation.
- **11 existing staged CPU regression tests passed**. The real40-test prerequisite
  suite (including actual tiny native fixtures) runs in Kaggle with zero skips.
- All100 accepted images and reference identities validated; notebook cells
  compile with empty outputs and default execution disabled. No local Torch
  installation or model execution was performed. The original eight native
  source pins and environment lock remain unchanged.

Full pretrained execution and broader-run duration/storage remain to be checked
on Kaggle. The six-question calibration is sufficient to reuse the unchanged
operator, but does not predict population method rankings. This whole-procedure
comparison cannot identify an isolated E, FPS or merge effect. GQA development
and MMStar expansion follow after reviewing TextVQA100. Reserved GQA evaluation
remains untouched; no compressor improvement is introduced here.
