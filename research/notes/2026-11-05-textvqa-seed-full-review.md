# TextVQA100 twenty-seed review — November 5, 2026 (UTC archive November 6)

Archive: `textvqa-seed-full-v1-20261106T031102097140Z.zip`.
ZIP SHA256: `e0ec02ca94e6d6c01e76ffe6cd1521163a0da0eb5f9386468c59d2bb44671baf`.
ZIP and matching log copied from Downloads to research/results and checked byte-identical.

## Execution and integrity

Passed: 100 questions, 2,200 primary predictions, 100 feature files; 26.40 minutes.
All 463 listed artifact checksums verified. Every question has all22 arms and exact
96/32 random membership/mapping. Identity576, new-seed frozen replay and four-token
cache probe passed on the designated first question. These are bounded controls,
not independent repeatability checks for all100 questions.

## Results (consensus score ×100)

| Method | Score |
|---|---:|
| Uncompressed576 | 49.70 |
| Native96→32 | 41.50 |
| Pure random96→32, mean of20 seeds | 20.60 |
| Random seed range | 15.70–25.00 |

Native minus seed mean: +20.90 points. Paired image bootstrap95% interval
[13.92,28.32], 5,000 resamples, fixed RNG5901105; conditional on this development
cohort and these20 seeds. Random seed SD2.75 points. The old three-seed estimate
was20.63, so the previous native advantage was not explained by just three unlucky
random seeds. This is a whole-procedure comparison, not an isolated E-selector effect.

Full-credit outcomes under random masks: 7 questions correct for all20 seeds,
55 correct for none, 38 vary. Consensus score varies on46 questions. Seeds are repeated
measurements, not2,000 independent questions. The38 varying questions provide binary
within-question contrasts for the future coverage analysis; do not discard the others
from descriptive reporting or redesign the cohort after seeing these results.

Native gives full credit on37 questions; four of these never get full credit under
any of the20 random masks:35508 (can number40),35990 (Adidas),37278 (LG),39308
(Orcid.org). These are descriptive cases, not proof of lost evidence coverage.
Conversely12 native-not-full-credit questions get full credit under at least one
random mask. Searching20 masks is an oracle comparison, not a deployable selector.

Native loses10 uncompressed-full-credit answers and gains3; APR34/44=77.27%.
The advantage over random coexists with remaining compression losses.

## Historical drift

Baseline and all300 historical-seed predictions match archived generated token IDs.
One native output changed: textvqa:35257,2015→2010. Both receive zero consensus
credit, so the aggregate score is unchanged. Keep the first current realization and
record this variation; no corrective rerun or preferred-answer selection.

## What this does and does not establish

The20-seed run strengthens the TextVQA performance observation and supplies repeated
within-question outcomes for mechanism analysis. It does not yet establish that
answer-region coverage, detail, padding or feature norm explains the gap. OCR boxes
remain absent. Contextualized states and native basis placement/merging prevent
interpreting slot overlap as literal surviving information.

Next: verify OCR annotation availability and coordinate mapping; audit mappings;
compute feature norm/detail/geometric padding maps from the saved tensors on CPU;
then relate coverage to outcomes within questions, with image-level uncertainty.
No further GPU sweep is needed before this analysis. GQA preparation and Experiment2
remain deferred. Full TextVQA portion of Step0 feature acquisition is now complete.

Reproduce this review with:

```sh
python3 scripts/summarize_textvqa_seed_coverage.py \
  research/results/textvqa-seed-full-v1-20261106T031102097140Z.zip \
  research/results/textvqa-seed-full-v1-20261106T031102097140Z-review
```

Generated outputs: `summary.json` and `questions.csv`. Tensor files were not loaded
or deserialized during this integrity/descriptive review.
