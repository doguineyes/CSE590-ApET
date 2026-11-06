# Basis attribution from saved selector records — 2026-11-06

Source: selector-grid-textvqa-full-v1-20261106T165132846914Z.zip. All internal hashes verified. CPU implementation: scripts/analyze_selector_basis_cpu.py. Results: research/results/selector-basis-cpu-v1 (summary.json, per-question.csv, exclusive-loss-cases.csv). No GPU rerun or modification to running experiments.

## E-drop's second-stage scoring basis

On the stricter30 OCR-candidate subset,26 questions retain at least one candidate answer patch after stage1. Across these questions,94 answer-patch occurrences enter stage2;22 survive,72 are discarded.

- 26/94 (27.66%) belong to the10 scoring-basis tokens. Reference random membership is10/96=10.42%.
- All26 are absent from final E-drop selection. Their residual errors are near zero (maximum4.87e-5 across the30 cases).
- Under a synthetic random10-of96 basis on each saved pool, expected total answer hits=9.79; simulated95% range5–16 (20,000 draws). This is a descriptive reference, not a new model intervention.
- Image bootstrap95% interval for pooled basis membership:19.30–38.96% (10,000 draws, seed590). Fixed automatically selected OCR subset; not population inference.
- Basis exclusion accounts for26/72=36.11% of discarded answer-patch occurrences. The other46 are NOT basis tokens. Basis exclusion therefore does not explain all losses.
- Pooled answer-patch survival is22/94=23.40%. Mean survival across26 eligible questions is23.14%. A ratio of previously averaged coverages is a differently weighted statistic, not the unweighted mean of per-question survival.
- All32 unique candidates give26.17% basis membership, consistent with the strict30 result.

## Native direct-source coverage decomposition

Coverage is the fraction of OCR-overlapping patches represented by direct source lineage. It does not measure semantic information, and excludes additional sources carried by merging and attention.

| Native stage | Basis coverage | Ordinary coverage | Total | Basis share of mean total coverage |
|---|---:|---:|---:|---:|
| Stage1,96 slots | 9.64% | 31.40% | 41.04% | 23.50% |
| Final,32 slots | 11.06% | 8.94% | 20.00% | 55.32% |

The basis occupies10/96 stage1 slots and10/32 final slots. Its share of mean coverage exceeds its slot share at both stages. Components have zero direct-source overlap in these records; totals reconcile with the prior CPU audit exactly. Full32-candidate sensitivity gives55.34% final basis share.

This is evidence that basis insertion is substantial for direct OCR-region retention in native ApET, especially at the second cut. It does not establish that basis tokens cause55% of correct answers or that removing them would cause the corresponding failures. Native states differ from E-drop states due to basis insertion and merging; the native and E-drop scoring bases need not be the same.

## What the external review gets right and overstates

The basis blind-spot hypothesis is worth pursuing: low residuals of scoring-basis tokens make plain top-E selection systematically exclude them, even when they overlap candidate answer regions. The archive directly verifies this positional pattern.

However, "whatever goes wrong happens at the second cut", "blind spot confirmed", "ApET seems to work because FPS...", and "detail does the same job" are stronger causal/mechanistic claims than these records justify. FPS selects diversity relative to its feature-space distances; semantic distinctiveness is not guaranteed. Ridge residual is approximately, not mathematically necessarily exactly, zero. Nor can feature similarity to basis tokens be inferred merely from spatial proximity; feature tensors are not saved here.

A raw independent-failure expectation10*9/44 also assumes exchangeable independent difficulty across questions, which is not established. Shared failures cannot identify a mechanism.

## Next

Keep current GQA replication unchanged. Prioritize a bounded basis-preserving no-merge/native-drop and stage-localization comparison before claiming basis causes the answer benefit. Save layer16 states/norms in a separately versioned future run if testing feature outliers. No extra GPU work implemented in this CPU review.

An11-row case table records6 native-only and5 detail-only losses, including questions, references and actual responses. These cases have not been visually classified into mechanisms in this analysis.

Validation: synthetic slot-to-source replacement test, exact reconciliation of native totals for32/30 subsets, and patch-count conservation checks passed.
