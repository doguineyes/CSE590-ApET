# 2026-10-13: GQA diagnostic pilot prepared

Prepared while the full MMStar run was executing. No changes were made to native
ApET files, the running MMStar evaluator, shared model helpers, or locked dependencies.

## Data evidence

Pinned LMMs-Lab GQA revision: `a6e72d6e1b912da88af8b2f9eba05d5ea8ec2dd8`.
Two source parquets were downloaded and hash-verified: 2,090,335 bytes of question
metadata and 65,647,745 bytes of image data. They contain 12,578 Test-Dev balanced
questions and 398 images. Their hashes are recorded in `prepare_gqa_pilot.py` and
the prepared subset manifest.

Seed 590 selected 10 images from 384 eligible groups, four unique questions each.
Each group includes query and verify types and varied semantic types. The 40
selected questions comprise 13 attribute, 10 relation, 9 object, 4 global, and 4
category questions. Selection did not use any model predictions.

CPU-prepared bundle: `research/results/gqa-pilot-data-10x4.zip`, 2,151,621 bytes.
Manifest SHA256:
`62d6eab82b868837330c69a8ec67045d3f02f1edbaa3d5078f808b858034fd7b`.
Image bytes and decoded RGB hashes were validated. Raw source annotations,
semantic programs, question IDs, answers, and image IDs remain in the manifest.
The bundle is ignored by Git; the notebook can prepare the subset itself.

## Planned run and records

160 predictions: baseline, input-only, decoder-only, and combined, with one model
load. All questions about an image use the same SHA256-derived image seed. The
short-answer scorer follows the upstream ApET GQA converter's lowercase/trailing-
period normalization, plus whitespace trimming. It does not extract answers from
prose or guess synonyms. This is a selected diagnostic subset, not an official
aggregate benchmark score.

Project-owned return tracing reads the native reconstruction errors, FPS indices,
selected indices, basis assignment slots, dropped-to-retained merge assignments,
and counts. It does not substitute compression functions or retain GPU tensors.
Input boolean-mask replacement and decoder advanced-index replacement use different
assignment orders; logged basis-slot/source pairs respect those native orders.

The pilot requires logging-on/off answer, token-count, and compression-count checks
on the first question of every configuration. It records same-image changes in
bases, selections, merge assignments, and error vectors across questions. Logging
adds overhead; these timings are not speed benchmarks. No answer-contribution
interventions or reconstruction/contribution correlations have been run yet.

## Validation and next action

Ten CPU checks passed for grouping/seeds, scoring, tracer restoration, merge mapping,
manifest integrity, single-load orchestration, and partial-result preservation.
Two tests on actual native Torch functions are included for Kaggle; they were skipped
locally because Torch is absent. GPU inference and observation parity await Kaggle.

After MMStar finishes, download its evidence, commit/push this update, and run
the [GQA notebook/guide](../GQA_DIAGNOSTIC_PILOT.md) in the live session to reuse
the cache. Review those results before choosing an intervention subset for RQ1.
