# TextVQA audit: separate cached decoding from scorer validation — 2026-10-16

The first six-case audit failed after three cases, before any token interventions.
Archived originals:
`research/results/rq1-textvqa-target-audit-v1-20261016T180759434962Z.{zip,log}`.
The ZIP has 34 entries, passed CRC, and all recorded evidence hashes match.
Runtime commit `f7a3a437be083414db0aab43fc2e2092bb74a0c8`, clean worktree.
Original native sources, dependency lock and selection remain unchanged.

## Observed failure

| Case | Generated answer | Fixed target IDs match | Reference repeat drift | Max cached-generation/full-teacher-forcing gap |
|---|---|---|---:|---:|
| 37533 | Brolio | yes | 0 | 0.00007637 |
| 35670 | Dominion | yes | 0 | 0.00008458 |
| 35962 | Cookery school | yes | 0 | 0.00549626 |

All three answers have full consensus credit. Cookery school uses target IDs
`[17278, 708, 3762]`. Full teacher-forcing log probabilities are
`[-0.35375750, -0.73848540, -0.21662362]`; cached-generation scores are
`[-0.35375750, -0.74398166, -0.21662295]`. The second target token caused the
0.001 gate failure; the first matches exactly and the third differs by about
0.00000067. The remaining three cases were not run.

This is not a casing mismatch, wrong answer or evidence of teacher-forcing
repeat drift. The archive establishes a difference between two execution paths.
FP16/kernel/cache-path differences are a plausible explanation, not a proven
root cause. Generation observations are processed generation scores; they should
not automatically be treated as numerically identical to raw full-forward logits.

## Correction to project instrumentation

The earlier guard conflated cross-path agreement with stability of the scorer
used for interventions. Audit **v2** keeps exact target-token prefix alignment
and the same **0.001** tolerance, while directly validating the full teacher-
forced scorer with reference repeat, clone-only no-op and native/cached-feature
multimodal parity for **all** target tokens. First-token prefill agreement with
generation is still required. Later incremental cached-generation differences
are saved as diagnostics, not treated as within-scorer drift.

The TextVQA intervention path uses the same distinction and still enforces
fresh full consensus credit, target IDs, full-forward repeat/no-op/native/final
checks. GQA/MMStar acceptance rules are unchanged. E, I, fixed reference targets,
patch selection, original model files and dependency versions are unchanged.
This is an explicit validation-policy revision, labeled
`full-teacher-forced-controls-v1`, not evidence that the old failed run passed.

Regression checks cover a 0.0055 cached-decode gap with stable full-forward
controls, rejection of 0.0055 same-forward drift, unchanged prefix failures,
and retention of diagnostic values. The Torch tiny-model test exercises the same
distinction on Kaggle. Pretrained scorer parity for these six cases remains
unvalidated until the new GPU audit returns. No tolerance was increased.

## Next execution

Push/pull once, run the prerequisite tests, then rerun **only the six-case audit**
with label `rq1-textvqa-target-audit-v2`. Send its ZIP/log for review before
calibration. Reuse the existing environment, weights cache and v1 input bundle;
no baseline rerun or new input upload is needed. The previous failed ZIP/log and
all earlier results are preserved.
