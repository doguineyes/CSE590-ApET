# TextVQA full input-stage RQ1 pilot — 2026-10-16

All six cases passed **576 interventions each: 3,456 complete patch effects**.
This completes the first full-grid input-stage pilot across the three proposal
datasets. The currently received full-grid cohorts comprise GQA 12, MMStar 3
and TextVQA 6 questions; the planned MMStar twelve-case extension is still pending.
No repeat of this TextVQA pilot is needed now.

## Archived evidence

Original ZIP/log were copied from Downloads to
`research/results/rq1-textvqa-input-pilot-v1-20261016T204945039325Z.{zip,log}`.
ZIP: 38,516,962 bytes / 77 entries, CRC passed. SHA-256:
`cf37506d9a3c087e60db409de57feb3691c813b2e3dd2cdc086de0e187ff99ea`.
Log SHA-256:
`bb58a1717f556be5c2c12a9fadda05e9d106ebca7812e39e2e3d33d1f7d850e0`.
Original files and Downloads copies are preserved.

Runtime commit `2f66364513a7e50b59a90d383d56faa815c28b35`, clean worktree,
metric `rq1-input-zero-v2`, TextVQA policy `full-teacher-forced-controls-v1`.
Original native source reference and lock match the accepted baseline. Every
recorded evidence hash and all six cached tensor-file hashes were verified.
Each question contains exactly indices 0–575 with finite, unique scores.
All 3,456 exported rows agree with raw records; native E values, I arithmetic
and correlations were independently checked by the role-aware CPU review.

All six fresh baseline answers have full consensus credit and exact target-token
prefix alignment. Full-forward repeats, clone-only no-ops, native/cached-feature
parity, first-token generation comparison and final-reference checks have **zero
drift**. The Cookery school later cached-decoding gap remains 0.00549626 nats,
recorded separately under the previously documented policy.

For Brolio and Dominion, image feature hashes, references and **all 64 repeated
calibration E/I values match exactly**. The full execution order begins with the
same 32 indices per case. This is cross-run evidence for these cases, not a claim
of universal numerical determinism.

The prerequisite is a newer passed six-case audit from the same commit:
`rq1-textvqa-target-audit-v1-20261016T204856423651Z`, report SHA-256
`a9a2bc307753c15c1c28a8c8b15c2dfaa477bf7af05c08adaec447fd23c47f55`.
Its complete report and six detailed target records are preserved and hashed
inside this pilot ZIP. Its standalone ZIP/log have not been supplied locally;
the audit log is an optional archival gap, not missing intervention data.

## Primary result

| Target | Archived input-only role | Rho, all 576 patches | Rho, 566 non-basis sources | Maximum positive I, nats/token |
|---|---|---:|---:|---:|
| Brolio | loss | −0.01341 | −0.01159 | 0.12858 |
| Dominion | preserved | −0.03539 | −0.04153 | 0.05063 |
| Cookery school | loss | +0.01010 | +0.01407 | 0.31589 |
| Sky mall | preserved | +0.04255 | +0.03991 | 0.03162 |
| Give up | loss | −0.03258 | −0.05883 | 0.13228 |
| Laughter | preserved | −0.03970 | −0.03166 | 0.00900 |

Median per-question rho is **−0.02299**, range **[−0.03970, +0.04255]**. Excluding
the ten FPS basis sources gives a median **−0.02162**. Basis residuals therefore
do not explain the near-zero ordering. The initial 32-patch correlations changed
from −0.19575 to −0.01341 for Brolio, and +0.04655 to −0.03539 for Dominion,
despite identical repeated measurements. This again illustrates sampling limits.

Across these selected questions, reconstruction error does not provide a useful
global monotonic ranking of positive singleton answer support under our proxy.
Both loss and preserved examples show near-zero correlations. **Low rho alone
does not distinguish or explain native compression failures.** Nor does it
establish statistical independence, rule out nonlinear/tail relationships, or
show that native compression is ineffective.

The three loss cases have larger maximum positive effects than the three controls
here, but their baseline reference scores differ too: mean log probabilities are
−0.139/−0.436/−0.619 for Brolio/Cookery/Give up versus −0.044/−0.043/−0.010 for
Dominion/Sky mall/Laughter. With three outcome-selected cases per group, different
questions and target lengths, this is descriptive context rather than evidence
of a predictive loss detector or an isolated causal group contrast.

## Maps and specific mechanism candidates

All six aligned 336-pixel figures were visually inspected. Positive sensitivity
peaks often lie near the printed answer evidence, while the overall E maps differ.
Useful examples from the native-role mapping (patch indices are zero-based):

- **Cookery school, patch 136:** near the wall lettering; I = +0.31589, its largest
  positive effect. E = 16.60149, around the 46th percentile among non-basis sources.
  Patches 136 and 137, the two largest effects, are outside the directly retained
  original-feature targets. Patch 140 has lower-quartile E (about 21st percentile)
  and I = +0.07838, also without direct original retention.
- **Brolio, patch 317:** near a wine label; I = +0.12858, its largest positive
  effect. E is around the 73rd non-basis percentile and this original-feature target
  **is retained**, despite the earlier compressed answer changing to Bottoli.
  Thus loss is not explained merely by omission of the most sensitive feature.
- **Give up, patch 82:** near upper poster lettering; E is around the 24th
  percentile and I = +0.11391, outside the directly retained original targets.

These are post-hoc investigation candidates, not verified causes. Native retained
features may be modified by merging, and sources without direct retention may
still contribute through merging. Zeroing one full-model representation changes
a different system from native joint selection/merging and sequence shortening.
Several low-E/positive-I examples lie in padding: contextualized patch features
cannot be interpreted as isolated local pixels, and artificial zeroing itself
needs interpretive caution. Do not label all such points task-critical lost text.

Auxiliary E versus absolute-I correlation among non-basis patches is −0.16950
for Cookery school and −0.30361 for Give up. This suggests an influence-magnitude
pattern worth inspection, but absolute I includes negative effects and is not a
replacement for the registered signed contribution metric.

## Runtime and next work

Wall time: **1,906.17 seconds / 31.77 minutes**. Per-question loops took about
292–370 seconds. Peak allocated GPU memory was **7.43 / 7.27 GiB**. These are
instrumented full-forward scoring measurements, not native compression latency.

Saved [review gallery](../results/rq1-textvqa-input-pilot-v1-20261016T204945039325Z-review/index.html),
question/patch CSVs, low-E/positive-I candidates and `calibration-comparison.json`
are separate from the original result files. Additional reference-score/target-
length context is in the adjacent `additional-descriptive-context.json`.

Next, inspect the planned twelve-case MMStar run when it returns, then consolidate
question-level results across datasets. Use selected maps and native merge
assignments to investigate actual compression losses; a global rank correlation
alone is insufficient. A small decoder-layer pilot remains required for the
layer component of RQ1 and needs separate instrumentation. No inference-code or
environment change was made during this result review.
