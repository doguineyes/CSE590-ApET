# Understanding Reconstruction-Based Visual Token Compression: When and Why Does ApET Work?

**Midterm Progress Report — Draft, October 21, 2026**

Peng Chen · CSE590-01 · chen439@cs.stonybrook.edu

## Abstract

This project investigates when reconstruction-based visual token compression preserves a vision-language model's answers. Using native ApET with LLaVA-1.5-7B, we completed a diagnostic study of reconstruction error and intervention-specific answer sensitivity, followed by controlled component tests and staged comparisons against random pruning. On 30 selected questions, reconstruction error does not consistently align with two answer-response measures; the measures also disagree with each other. Across larger evaluations, native ApET substantially outperforms random pruning on a TextVQA screening cohort, has a smaller advantage on GQA development data, and shows no clear aggregate difference on MMStar. These findings distinguish geometric redundancy from measured answer support, while leaving the independent contributions of selection and merging partly unresolved.

## 1. Motivation and Research Questions

Visual tokens contribute substantially to inference cost in vision-language models. ApET [1] compresses them without training, using reconstruction error to select representations and similarity-based merging to aggregate discarded features. Our goal is to understand this procedure's behavior, rather than reproduce every result in the original paper or redesign the model.

The proposal asks:

1. **RQ1:** How does reconstruction-based visual redundancy relate to task-relevant answer contribution, and how does this relationship change across task types and model layers?
2. **RQ2:** How much of ApET's effectiveness comes from reconstruction-error-based token selection, and how much comes from similarity-based token merging?
3. **RQ3:** Can the resulting analysis support a lightweight improvement to ApET, such as adaptive compression strength or an alternative reconstruction basis?

This report closes the first experimental phase: bounded RQ1 diagnostics and RQ2 comparisons. RQ3 remains future work; RQ2's full component attribution is not yet complete.

## 2. Experimental Design

### Model and compression

All reported model experiments use native LLaVA-1.5-7B [3] and the public ApET implementation, in FP16 on Kaggle T4 ×2. The locked environment uses Python 3.10.18, PyTorch 2.1.2+cu121 and Transformers 4.37.2. Project-owned runners record inputs, configurations, source identities, raw answers and controls separately from the native compression implementation. Registered compatibility fixes support the runtime without replacing ApET's compression equations.

The uncompressed image produces 576 visual states. Native ApET uses ten farthest-point-sampled (FPS) basis features, calculates linear reconstruction residuals, selects high-error features, inserts basis representations into selected slots, and merges discarded features into cosine-nearest ordinary retained targets. Selection and merging therefore use different criteria; reconstruction coefficients do not directly specify merge weights.

### RQ1 measurements

The curated diagnostic panel contains 30 questions over 24 images: GQA12, MMStar12 and TextVQA6 [4–6]. GQA includes six same-image question pairs. We examine all 576 positions per question at the input boundary and six matched questions after decoder block 16. The input error $E_i$ is measured in native visual-feature space; input interventions act on the corresponding projected model states.

To compare two counterfactuals using the same score, let $p(y_1\mid H,q)$ denote the probability of the first tokenizer piece of a frozen reference answer. We measure:

$$
I_i^{\mathrm{zero}}=p(y_1\mid H,q)-p(y_1\mid Z_i(H),q),
$$

$$
I_i^{\mathrm{retain}}=p(y_1\mid R_i(H),q)-p(y_1\mid R_{\varnothing}(H),q).
$$

$Z_i$ zeros one visual vector while preserving sequence length. $R_i$ physically retains one visual state; $R_{\varnothing}$ removes all visual states at that boundary. Retention preserves nonvisual states and original position IDs. This is a source-informed 1×1 adaptation of Information Horizon [2], not its complete original benchmark. Positive scores indicate increased reference-piece support under the specified contrast. Neither score is intrinsic token importance or whole-answer accuracy.

### RQ2 comparisons and scoring

The broad evaluation includes TextVQA100 screening/development questions, GQA800 development questions from 100 images, and the full pinned MMStar1500 set. Each question receives nine primary predictions: full576, native ApET at two schedules, and pure random pruning with seeds 101/202/303 at both schedules. This totals 2,400 question evaluations and 21,600 predictions.

The schedules retain **96→32** or **288→96** states at the input and after block 16, respectively. Pure random uses the same boundaries and counts, with no FPS basis, error ranking, basis replacement or merging. Thus these are **complete-procedure comparisons**, not isolated tests of error ranking. Separate input-only tests hold selection fixed when manipulating merging.

GQA uses normalized short-answer accuracy, MMStar uses the project's explicit-choice scorer, and TextVQA uses ten-annotator partial-credit consensus scores. We also track answer preservation rate (APR): the fraction of baseline-correct answers still correct after compression; TextVQA APR uses full-credit answers. Baseline-wrong questions remain in aggregate evaluations. Paired 95% bootstrap intervals average random seeds within questions and resample image groups, rather than treating seeds or shared-image questions as independent examples.

## 3. RQ1: Reconstruction Error and Answer Response

**Table 1.** Median per-question Spearman correlation over all 576 positions. These summarize a selected diagnostic panel, not population estimates.

| Cohort | $E$–zeroing | $E$–retention | Zeroing–retention |
|---|---:|---:|---:|
| GQA12 | −0.087 | −0.068 | −0.016 |
| MMStar12 | +0.009 | −0.121 | −0.005 |
| TextVQA6 | −0.031 | −0.136 | −0.028 |

Reconstruction error does not consistently rank positive answer support under either intervention. Individual error–retention correlations range from −0.486 to +0.404, so the result is mixed rather than uniformly near zero. Filtering also matters: restricting analysis to non-basis positions fully inside the unpadded image changes the GQA median error–retention correlation to +0.087. These results do not imply that reconstruction error contains no useful information.

All six GQA pairs have identical input features and error rankings, but between-question retention-map correlations range from −0.443 to +0.655. The answer response depends on the question and target even when visual error does not. This is not pure isolation of question wording: the reference answer and baseline confidence change as well.

The six-case depth comparison gives no consistent improvement in error–sensitivity alignment after block 16. Input-to-layer retention-map correlations range from −0.038 to +0.198. Full sweeps also expose sparse strong states that 32-position calibration missed: for *Dominion*, the best retained first-piece probability is 20.2% at input and 98.1% at block 16. These are different contextual states, not full-answer recovery rates or evidence that a single raw patch independently contains the word.

Zeroing, half-strength replacement and neighbor replacement do not yield reliably interchangeable rankings in the targeted validity checks. The near-zero median agreement between zeroing and retention further supports treating them as different measurements. At depth, retained text states have already processed the image, so removing late visual states is not equivalent to a question-only model. Conversely, visual positions precede the question under causal attention: their native error rankings do not become question-conditioned merely by being deeper.

## 4. RQ2: Native ApET, Random Pruning and Merging

**Table 2.** Dataset-specific scores (%); random is the mean of three seeds. Differences and intervals are percentage points. Each dataset uses its own metric and sample role; scores are not pooled.

| Cohort | Schedule | Full576 | Native | Random | Native − random [95% interval] |
|---|---|---:|---:|---:|---:|
| TextVQA100 | 96→32 | 49.70 | 41.50 | 20.63 | +20.87 [13.93, 29.03] |
| TextVQA100 | 288→96 | 49.70 | 44.20 | 34.77 | +9.43 [3.20, 15.43] |
| GQA800 | 96→32 | 69.00 | 63.75 | 61.29 | +2.46 [0.25, 4.71] |
| GQA800 | 288→96 | 69.00 | 68.00 | 66.83 | +1.17 [−0.58, 2.83] |
| MMStar1500 | 96→32 | 34.73 | 33.40 | 33.53 | −0.13 [−1.35, 1.07] |
| MMStar1500 | 288→96 | 34.73 | 35.00 | 35.20 | −0.20 [−1.25, 0.85] |

Native ApET has its clearest advantage over random pruning on the evaluated TextVQA cohort, especially at the stronger compression schedule. It nevertheless loses score relative to full576. GQA shows a smaller advantage, with uncertainty spanning zero at 288→96. MMStar shows no clear aggregate difference; intervals containing zero do not establish equivalence. Its 34.73% baseline accuracy also limits conclusions about preservation of abilities the model rarely demonstrates.

Aggregate accuracy can hide substantial turnover. On MMStar, native 288→96 loses 36 baseline-correct answers and gains 40 previously wrong answers. Its small net increase therefore coexists with APR of 485/521 = 93.09%. At 96→32, APR falls to 442/521 = 84.84%.

The component diagnostics use a separate, failure-enriched panel of 30 baseline-correct questions at **input-only 288 states**. With the same error-selected positions and basis placement, native merging answers 19/30 correctly versus 21/30 without merging. Merging both helps and harms under other retained sets; this panel does not establish a general advantage for disabling it. High-error target protection barely changes the routing, limiting interpretation of its negative result.

A targeted TextVQA case provides stronger local evidence. For *Brolio*, the input-only merged condition produces *Bottoli*, while the same selection without merging produces *Brolio*. Restoring the pre-merge value of just the retained state at grid index 317 also restores *Brolio*; the unchanged sham and a count-matched control restoration remain wrong. This identifies a conditional merge-related failure with selection fixed. It does not establish a unique causal token, literal letter blurring, or a general protection rule.

## 5. Roadblocks, Limitations and Status

An early roadblock was dependency incompatibility between the older native implementation and Kaggle's changing default environment. A separately locked runtime and scripted bootstrap now allow repeatable notebook setup, while local work remains editing and CPU analysis. Frozen inputs, predictions, source snapshots and replay controls keep project instrumentation distinguishable from native ApET. Exact encoded replays were stable, although some fresh compressed runs changed answers; current intervals condition on the recorded native realizations and three random seeds, not all numerical variation.

Measurement interpretation was another substantive challenge. First-piece scores can disagree with whole-answer scores; zeroing and retention change different contexts and sequence geometries. Small samples can miss sparse strong responses. Question quality, answer normalization and weak visual grounding required review, especially for some MMStar diagrams. The diagnostic panel is selected, TextVQA100 is screening data, and GQA800 is development data. Two representation boundaries do not establish a general depth law, and benchmark timings with instrumentation are not controlled speedup measurements.

The completed phase supports two preliminary conclusions: reconstruction error is not an intervention-independent ranking of answer importance, and native ApET's advantage over pure random depends strongly on the evaluated task distribution and budget. These findings are compatible: a useful compression procedure need not reproduce singleton response rankings. However, the staged comparison changes selection, basis representation and merging together, so their separate contributions remain unresolved.

No additional GPU experiment is required for this midterm. After consolidating the report, a focused stage-wise merging ablation and selected exact-text failures can guide whether one lightweight RQ3 modification is justified. Adaptive budgets and alternative bases remain possibilities, not implemented improvements.

## References

1. Qiankun Ma et al. **ApET: Approximation-Error Guided Token Compression for Efficient VLMs.** CVPR, 2026. [Paper](https://arxiv.org/abs/2602.19870).
2. Yahong Wang et al. **When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs.** CVPR, 2026. [Paper](https://arxiv.org/abs/2512.07580).
3. Haotian Liu et al. **Improved Baselines with Visual Instruction Tuning.** CVPR, 2024. [Paper](https://arxiv.org/abs/2310.03744).
4. Drew A. Hudson and Christopher D. Manning. **GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering.** CVPR, 2019. [Paper](https://arxiv.org/abs/1902.09506).
5. Lin Chen et al. **Are We on the Right Way for Evaluating Large Vision-Language Models?** 2024. [Paper](https://arxiv.org/abs/2403.20330).
6. Amanpreet Singh et al. **Towards VQA Models That Can Read.** CVPR, 2019. [Paper](https://arxiv.org/abs/1904.08920).
