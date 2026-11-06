# Selector-grid CPU review — 2026-11-06

## Sources and reproducibility

Source: `research/results/selector-grid-textvqa-full-v1-20261106T165132846914Z.zip` (all 3100 predictions passed). All internal artifact hashes verified before analysis. CPU script: `scripts/analyze_selector_grid_cpu.py`. Canonical derived outputs: `research/results/selector-grid-textvqa-cpu-v2/{summary.json,per-question.csv,mask-overlap.csv}`. An initial CPU output v1 had an OCR-status string mismatch and is explicitly marked INVALID; no GPU results changed.

Run with the editor Python, passing the ZIP, `research/results/textvqa-evidence-cpu-v1-reviewed`, and a fresh output directory. No model, Torch or GPU needed.

The 30-case OCR subset is 32 uniquely auto-matched candidates minus two previously audited exclusions. Only 20 matched candidates were individually visually audited; do NOT describe all 30 as certified answer boxes. Coverage is the fraction of patches intersecting candidate OCR boxes retained, not fraction of answer information preserved. No new image review or answer-based selection was performed here.

## Paired performance differences

100 questions on 100 images; 10,000 paired image bootstrap draws, seed590. Random arms average the same five tested seeds within each question. Intervals condition on these seeds; they do not include uncertainty over new seeds or multiple-comparison adjustment.

| Contrast | Score difference (percentage points) | 95% interval |
|---|---:|---:|
| Detail drop − random drop | +18.00 | +10.08 to +26.30 |
| Nonpad drop − random drop, matched5 | +4.96 | +2.00 to +8.18 |
| Detail merge − native | +2.00 | −5.10 to +9.10 |
| E drop − random drop | −0.10 | −5.82 to +5.86 |
| Detail drop − E drop | +18.10 | +8.40 to +28.10 |

Detail performs substantially better than random in this cohort. Padding exclusion has a smaller positive effect in the matched-seed comparison. These do not establish that detail explains the causal mechanism of ApET, nor that native and detail-merge are equivalent. No equivalence margin was specified.

## Answer-region coverage and padding

Coverage averaged per question on the strict30 subset. Random/nonpad are averages over matched5 seeds. Padding is mean fractional padding area in selected carrier patches across all100 images.

| Selector | Stage1 coverage % | Final coverage % | Stage1 padding % | Final padding % |
|---|---:|---:|---:|---:|
| Random drop | 14.87 | 4.21 | 27.66 | 27.83 |
| Nonpad drop | 24.11 | 8.26 | 0.00 | 0.00 |
| Detail drop | 51.73 | 29.05 | 0.70 | 0.82 |
| E drop | 33.71 | 7.52 | 7.42 | 3.18 |
| Native | 41.04 | 20.00 | 7.42 | 3.25 |

Native coverage traces direct basis-source replacement; native pixel statistics describe carrier slots. Neither native coverage nor the other methods' coverage includes information mixed across positions by the transformer. Native merging can carry additional source content not represented by direct-source coverage.

E stage1 retains more candidate answer patches than random, while much less of this coverage survives the second cut than under detail. This motivates stage localization, but does not prove stage2 causes the answer failures. Every method reduces96 to32, and redundant/contextual information and merging matter. Detail has a small nonzero padding fraction because border patches containing real image pixels can be selected; it is not equivalent to a strict nonpad eligibility filter.

The corrected detail score (excluding gradients touching padding) is higher for detail-selected patches by construction. Mean selected detail, stage1/final: random0.1802/0.1788, nonpad0.2545/0.2577, E0.2849/0.3035, detail0.5302/0.6799. These are normalized-pixel gradient units, not answer-relevance measurements.

## Mask overlap and basis interpretation

E-drop and native share all96 stage1 carrier slots. They do NOT share all token contents: native replaces designated slots with FPS basis source features and merges dropped features. At the final stage their carrier-slot overlap averages13.85/32. E-drop vs detail-drop averages28.30/96 initially and3.69/32 finally. Native carrier overlap must not be mistaken for direct source overlap.

Native full-credit losses vs full576:10; detail-merge losses:9. Only4 losses are shared (36584,37533,38806,38830). Native gains3; detail-merge gains4, sharing all3 native gains. Thus similar total scores conceal different failures. Even identical failures would not establish the same mechanism.

## Missing measurement and next decision

The current ZIP stores layer16 errors and fingerprints but no hidden-state tensors or per-token norms. We cannot calculate retained-vs-dropped layer16 norm distributions, or identify sink-like states, from this ZIP. Input feature tensors from the earlier seed study do not substitute for each new arm's layer16 states.

Continue the existing GQA drop-arm replication unchanged. CPU results motivate a small future stage-localization comparison, but not a claim that a stage failure is confirmed. Native-drop versus E-drop also changes subsequent states/masks; basis effects cannot be inferred as a simple additive decomposition. Additional GQA merge arms would be a separately versioned extension, not a change to the running protocol. Manual answer boxes can replace the provisional OCR proxy later.
