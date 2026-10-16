# TextVQA reference audit v2 passed — 2026-10-16

All six frozen TextVQA references pass the revised scorer-validation policy.
The two-question Brolio/Dominion calibration is the correct next run; its result
is pending. No inference-code update is needed before that calibration.

## Archive and verification

ZIP and log were copied from the explicitly supplied Downloads paths to
`research/results/rq1-textvqa-target-audit-v1-20261016T182747377877Z.{zip,log}`.
Original names are preserved: the output label says **v1**, but the recorded
protocol is **rq1-target-audit-v2**. The label does not determine the protocol.

ZIP: 1,965,050 bytes / 43 entries; CRC passed. SHA-256:
`b3d15bf730fc7833d1f4d3d455bc4635abec5e0b5a32ad7306549da7e6cf8d6e`.
Log SHA-256:
`d7f4116b030edadeda95a66b846fcba75fae2cd0e118c8af5580636b1ad60d62`.
Runtime commit `7bd25dc3309953376ebf7bb0a7511c60c509f405`, clean worktree;
original lock unchanged. Runtime was **36.75 seconds**.

All recorded evidence hashes match, as do the six used image hashes. Actual
preprocessed canvases are 336 × 336. The runtime's audit-prerequisite validator
also accepts the extracted report against the local accepted manifest, frozen
selection, model/vision sources, seed, measurement sources and environment lock.
Scores and control deltas were independently checked from raw arrays.

## Numerical and reference checks

All six baseline answers receive full native ten-annotator consensus credit.
Fixed annotation-derived token sequences match generated prefixes exactly.
Reference repeats, clone-only no-ops, native multimodal parity and first-token
generation/teacher-forcing comparisons have **zero drift on every case**.
Targets contain two or three tokens; checks cover the complete target, not only
the first word.

| Case | Fixed target | Target tokens | Max cached-decoding/full-forward diagnostic gap |
|---|---|---:|---:|
| 37533 | Brolio | 2 | 0.00007637 |
| 35670 | Dominion | 2 | 0.00008458 |
| 35962 | Cookery school | 3 | 0.00549626 |
| 39323 | Sky mall | 3 | 0.00002114 |
| 36001 | Give up | 2 | 0.00008088 |
| 38025 | Laughter | 3 | 0.00000744 |

For the first three cases, cached image-feature hashes, full teacher-forced
scores and generated-token log probabilities exactly repeat the previous failed
audit. Cookery school's cached/full gap therefore repeats while all new
full-forward controls agree exactly. This supports separating that cross-path
gap from drift in the full-forward scorer; it does not identify its underlying
kernel/cache/processed-score cause or establish general numerical determinism.
The 0.001 full-forward tolerance was not increased, and no reference was changed.

## Current action

Let the already running **two questions × 32 patches** calibration finish and
download its ZIP/log. It uses the passed report through `--target-audit-report`.
Do not pull measurement-code changes between audit and calibration. Inspect its
fresh baseline competence, exact prefix alignment, full-forward controls, final
reference drift, effects and timing before enabling the six-case full sweep.
Rho sign or magnitude is not a calibration acceptance criterion. No full-text
intervention result or reconstruction/answer-contribution conclusion is claimed
from this audit alone.
