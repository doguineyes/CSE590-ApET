# RQ2 first merge/no-merge result — 2026-10-19

The corrected v2 full run passed all 30 questions / 24 images. This is the first
admitted RQ2 merge/no-merge panel. It isolates merging conditional on the same
native E selection, FPS basis and basis-slot placement; it does not complete the
proposal's selection-versus-merging study.

## Preserved and verified evidence

Copied the supplied ZIP and separate console log byte-for-byte into
`research/results/`, retaining their Downloads originals:

- `rq2-input-merge-v2-full-20261019T224044177722Z.zip`: 669,595,732 bytes,
  SHA256 `bf7128e8ff963328d2f8fd1e228f4d1f791f3f84dc6b79f41041492a3f65ad4c`.
- Matching `.log`: SHA256
  `e527a2cdddfb97921ad1799a22b5216a7db0fb68050f8c34e77138860c037fe1`.

All 385 unique ZIP members pass streamed CRC/hash verification; 384 artifact
hashes, 244 evidence hashes and 58 primary/fresh feature-file hashes verify.
Both measurement and auxiliary source copies match their recorded hashes.
The clean runtime commit is `20ca2dc30e94ac9bff459696c847bf38c47f08b8`.
Native reference sources and the original environment lock match.

The full ZIP contains the prerequisite seven-case calibration report, detailed
questions, tensors, sources and setup evidence. Its report SHA256 is
`a314ca1c4abbc9d356acfb2b4f90c51bf375201b342d59ccb1411921deba4970`.
The embedded calibration's recorded artifact hashes and calibration identity
verify against the full run. A separate calibration console `.log` was not
supplied; the evidence required to verify its gate is embedded, so no inference
rerun is needed for this review.

CPU review independently rescored all 90 primary predictions with the unchanged
dataset scorers, reproduced all summaries, checked same-image feature-bank
identity, source/selection/basis/replay flags, target-score controls and archived
baseline equality. It does not load tensor pickles or run a model locally.
Derived verification, answers CSV and fresh-repeat CSV are in
[`results/rq2-input-merge-v2-full-review-2026-10-19/`](../results/rq2-input-merge-v2-full-review-2026-10-19/).
Raw source archives remain unchanged.

## Primary outcome

Every selected question is fresh-baseline correct by the existing RQ1 diagnostic
cohort design. These counts therefore also describe preservation in this panel.

| Dataset | Full576 correct | Native input288 merge | Same selection/basis, no merge |
|---|---:|---:|---:|
| GQA | 12 / 12 | 8 / 12 | 9 / 12 |
| MMStar | 12 / 12 | 8 / 12 | 8 / 12 |
| TextVQA | 6 / 6 | 3 / 6 | 4 / 6 |
| Total, descriptive only | 30 / 30 | 19 / 30 | 21 / 30 |

No-merge rescues two of native merging's 11 losses and introduces no new
correctness losses in this panel. Nine baseline-correct questions remain wrong
without merging. The two rescues are `gqa:201636988` and `textvqa:37533`.

This is a small, outcome-enriched mechanism cohort with same-image question
dependence and previous MMStar flags. The total is not a benchmark accuracy
estimate, population benefit claim or significance result. Absence of a
merge-only correctness win in these 30 cases does not establish that merging
never helps.

## What the result does and does not explain

The user's interpretation is reasonable: omitting merging may avoid damaging
answer-relevant retained features. The controlled result is more specific than
an uncontrolled change in answers: CLIP/E/FPS/selection/pre-merge states and basis
placement are held fixed, so which positions were selected cannot alone explain
the rescue. The condition that changes is whether ordinary retained states are
modified by native merging before projection.

That is evidence for a merge-dependent failure in this fixed configuration. It
does not identify one token as the cause, demonstrate a visual letter-blurring
mechanism, or establish an intervention-independent importance score.

The nine unresolved losses also do not prove that E selection alone caused
them. No-merge still uses 288 states with native basis replacement, fewer visual
inputs and a changed context. A random-selection comparison is required to
estimate the value of E ranking itself.

## Text examples and the RQ1 link

| Question/target | Full | Native merge | No merge |
|---|---|---|---|
| Wine brand: Brolio | Brolio | Bottoli | Brolio |
| School: Cookery school | Cookery school | Culinary school | Culinary school |
| Poster instruction: Give up | Give up | Gone | Go up |

**Brolio is the clearest confirmation of the earlier RQ1-derived hypothesis.**
In this full run, source position 317 is an ordinary retained target, not a basis
replacement slot. Its recorded incoming dropped sources are 103, 109 and 331,
with merge count three. Native merging modifies this retained feature rather
than simply dropping its output coordinate. Disabling all ordinary merging
restores the exact printed brand without changing selected positions/basis.

The fixed reference mean log probability changes from -1.50463 with merging to
-0.26520 without it (full baseline -0.13933). Its first reference-token probability
is about 0.05874 versus 0.70997. These are reference-token/sequence diagnostics,
not whole-answer correctness probabilities. The magnitude and stable generated
strings distinguish this from a tiny score-only numerical discrepancy.

The global no-merge ablation changes other retained targets too. A predeclared
restore-only-317 test would be needed to attribute recovery specifically to that
target. Such a test would be a post-hoc oracle mechanism diagnostic, not a
deployable selection/protection rule.

Cookery remains incorrect and Give up changes to another incorrect phrase.
The latter shows that merging affects the error's form even when full-credit
correctness is unchanged. Similar-looking words must not be silently rescored
as correct. The GQA cake/plate rescue is a smaller score-margin case: native
gold `No` probability is about 0.49605 versus 0.49995 without merging.

Merging does not lower every fixed-reference score: for example, Cookery and a
preserved same-image Window question have higher gold scores with merging,
without a correctness change. These secondary scores should not be substituted
for the primary generated-answer results.

## Repeatability and pipeline checkpoint

All 15 fixed-tuple replay repeats (five controls times three arms) have identical
generated IDs and fixed-reference token scores. All ten independent fresh
native repeats preserve source/selection/basis/merge-graph identities. Their
projected features still vary, but none changes generated IDs or correctness.
In particular, native Brolio remains Bottoli and native cake/plate remains Yes.

All 21 calibration/full generated-ID comparisons for seven cases times three
arms also match despite independent first captures. These bounded controls
support the observed rescues; they do not prove universal determinism or stability
on every example/seed/hardware configuration.

Wall time is 190.52 seconds, including model loading and instrumentation. This is
execution evidence, not a controlled speed comparison. The v2 pipeline has now
worked on pretrained Kaggle hardware without rewriting native ApET or changing
the locked environment. Another infrastructure/parity rerun is unnecessary now.

## Next scientific step

Prefer the planned matched-budget crossed control over repeating this coarse
ablation on more images immediately: E-ranked versus random ordinary retained
positions, each with and without merging. Hold the ten FPS basis sources and
their output-slot assignments fixed; use three predeclared random-selection
seeds and reuse each selected set for its merge/no-merge pair. This addresses
whether E ranking helps and whether merging's effect depends on selection.

A tiny Brolio restore-only-317 diagnostic can complement that comparison if a
local mechanism illustration is useful. It should remain separate from practical
high-E protection and must not use gold-informed targets as a general method.
Random-destination and matched protected-target controls remain later planned
RQ2 components. This first merge/no-merge result does not complete all of RQ2.
