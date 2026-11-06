# TextVQA basis/localization calibration and full review — 2026-11-06

Both ZIP/log pairs archived under research/results; Downloads originals retained. Copy SHA-256, ZIP CRC and internal artifact checks passed.

Calibration205019609843Z:5 questions,80/80 predictions,statuspassed,166.41s.27 native prerequisites passed with zero skips;50 shared-stage controls passed,75 arm-state records deduplicated to25 files.121 internal hashes verified. All replay/cache/identity checks passed; reference outputs unchanged.

Full205534587657Z:100 questions,1600/1600 predictions,statuspassed,1005.90s (~16.8min).1000 common-stage input/state checks passed.500 unique layer16 tensor files.976 internal hashes verified. Only recorded historical drift is native question35257; prior reports already flagged this case. Fresh native aggregate score remains41.5. Native reference and E/detail baseline scores match the preceding cohort totals.

Full archive SHA-256:b2e542bf41243d61402c8ad1b0e652a18243d9d89eebb89a65ee040ea6aabba4
Full log SHA-256:8b797b0d077719ac84edb3fd1886dcd523b426e69a18eeda2f4d7f9606ce2f60

## Basis factorial, no merging

| Input stage | Decoder stage | Consensus score % |
|---|---|---:|
| E,no basis | E,no basis |20.0|
| E,no basis | Native basis replacement |26.7|
| Native basis replacement | E,no basis |24.4|
| Native basis replacement | Native basis replacement |40.1|

Reference full57649.7; native merged41.5. The total both-stage basis-preserving variant exceeds neither-basis by20.1pp (paired image bootstrap95%11.1–29.0). Holding E input and pre-pruning states fixed, switching to native basis-preserving decoder adds6.7pp (1.2–12.7). Holding native-drop input fixed adds15.7pp (7.7–23.9). Difference-in-differences interaction9.0pp (0.6–17.6), exploratory and not multiplicity-adjusted.

This supports the contribution of native basis-preserving selection as implemented, especially the decoder operator. It does not assign abstract causal importance to individual basis tokens. Native replacement changes contents/positions, and input interventions alter downstream representations and numerical regimes. Stable research E ordering versus native Torch topk remains a possible tie-policy distinction. Do not claim additive independent basis effects.

Native merge minus native-drop is+1.4pp (−2.8,+6.0), so neither equivalence nor zero merge effect is established. This differs from earlier E-only merge gains; merge effects depend on the rest of the pipeline.

## Stage localization

FPS→FPS40.5; E→FPS28.7; E→detail27.3; detail→E22.7; detail→detail38.1. E→random five-seed mean23.2.

With common E input states, FPS decoder improves+8.7pp (3.4–14.6) and detail decoder+7.3pp (0.4–14.4) over E decoder. Random decoder+3.2pp (−1.78,+8.38) is less conclusive. Replacing detail's second-stage rule with E drops15.4pp (−24.2,−6.7). These are conditional stage2 operator interventions on verified identical inputs; they do not establish the full mechanism of losing answer information.

FPS-only−native=−1.0pp (−7.0,+4.7): equivalence not established, and numerical caveat below is essential.

## Numerical findings: observed FPS fp16 overflow

At decoder stage, infinity in visited squared-distance computations occurs:
- Native ApET:100/100.
- Native-drop:100/100.
- Native-input→E:100/100.
- FPS→FPS:100/100.
- Detail→E:2/100.
- E→E, E→native, E→FPS:0/100.

No input FPS overflow observed; no NaN distances; all logged sequences reproduce actual FPS selections. Calibration already showed overflow in four corresponding decoder arms on all5 images. This is native incoming-dtype behavior; no precision changes were made. Saved output features remain finite and execution controls passed. Consequently this is not a corrupt run, but pure diversity-based interpretation of the affected FPS sequences is unsafe. Inf can create capped1e10 ties; actual FP32 counterfactual selections have not yet been measured. Sequence reproduction is a diagnostic consistency check, not proof of numerical fidelity to ideal distance calculations.

Next CPU check should replay FPS in FP32 on these saved96-token layer16 states with the same initial centroid, compare basis/selected indices and quantify tie/overflow consequences. No full inference rerun is needed for that index comparison. A subsequent targeted GPU precision intervention would be necessary to determine answer effects. Do not silently replace native FPS before preserving this reference.

## Reproducibility and scope

`scripts/summarize_basis_localization.py` produces `research/results/basis-localization-textvqa-full-review.json`. Paired100-image bootstrap10000 draws,seed590; each image contributes once; five random seeds averaged within image. Intervals are exploratory, fixed-seed conditional and not adjusted for multiple comparisons. OCR boxes are not used in these score analyses. Saved states support later norms and feature-space checks.
