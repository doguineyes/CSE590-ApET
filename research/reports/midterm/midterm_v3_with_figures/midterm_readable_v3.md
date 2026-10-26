# Understanding Reconstruction-Based Visual Token Compression: When and Why Does ApET Work?

**Midterm Progress Report | October 2026**

**Peng Chen | CSE590-01 | chen439@cs.stonybrook.edu**

## 1. Abstract

This project asks when and why reconstruction-based visual token compression preserves a vision-language model's answers. We study ApET [1] with LLaVA-1.5-7B [3] on GQA, MMStar and TextVQA. We built a reproducible evaluation pipeline. On GQA, our runs keep the same share of uncompressed accuracy as the ApET paper at matched token budgets. On a 30-question diagnostic panel, ApET's reconstruction error does not consistently rank individual visual tokens by their effect on the answer: median per-question correlations range from −0.14 to +0.01. Yet at an average budget of 64 tokens, complete ApET beats random pruning by 20.9 points on TextVQA and 2.5 points on GQA. On MMStar, where the uncompressed model is near chance, there is no detectable difference. Almost all of ApET's advantage comes from losing fewer correct answers, not from gaining new ones. It is still unclear which part of ApET produces this advantage: error ranking, its basis tokens, or merging. The second half will first decide, after further analysis, how best to investigate this, and then whether any modification to ApET is justified.

## 2. Problem Statement and Goal Recap

LLaVA-1.5 turns each image into 576 visual tokens, which dominate the cost of inference. ApET [1] compresses them without training. It keeps the tokens that are hardest to reconstruct from a small set of basis tokens, and merges each discarded token into its most similar retained token. Our concern is that being hard to reconstruct is a geometric property. It need not match what the model needs to answer a particular question.

The proposal asks three questions:

- **RQ1:** How does reconstruction-based redundancy relate to a token's contribution to the answer, and does this change across task types and model layers?
- **RQ2:** How much of ApET's effectiveness comes from error-based token selection, and how much from similarity-based merging?
- **RQ3:** Can the analysis support a lightweight improvement to ApET, such as an adaptive compression budget or a different reconstruction basis?

The first-half commitments were a reproducible inference environment, prepared datasets and scoring, and instrumentation of errors, selections, merges and answer responses. All three are complete.

**Table 1.** Status at midterm.

| Question | Status | What we can claim now |
|---|---|---|
| RQ1 | Planned diagnostics largely complete | On 30 selected questions, error does not consistently rank tokens by answer sensitivity |
| RQ2 | Whole-procedure comparisons complete; component attribution open | ApET clearly beats random pruning on TextVQA, slightly on GQA, not on MMStar; the source of the gain is untested |
| RQ3 | Exploratory | A first feasibility check found no reliable image-level failure predictor; no modification chosen |

## 3. Methods and Implementation to Date

We run the public ApET implementation with LLaVA-1.5-7B in FP16 on Kaggle (2× T4), in a locked environment (Python 3.10, PyTorch 2.1.2 [7], Transformers 4.37.2 [8]). Our own runners and scoring code are kept separate from ApET's compression code. All predictions are archived.

![Two-stage ApET evaluation and RQ1 measurement points](figures/apet_study_pipeline_readable_v2.svg)

**Figure 1.** ApET compresses twice. It compresses once when visual tokens enter the language model, and once more after decoder block 16 of 32. RQ1 measurements are taken on separate uncompressed runs: reconstruction error is computed on the vision encoder's output, and answer responses are measured on the projected tokens that enter the language model.

**Terms used throughout.**

- **Schedule K1→K2:** keep K1 visual tokens at the input and K2 after block 16. Averaged over the 32 decoder layers, 96→32 corresponds to 64 tokens and 288→96 to 192 tokens. These are two of the budgets in ApET's Table 1.
- **Basis:** 10 tokens chosen by farthest-point sampling (FPS). They are always kept and used to reconstruct the rest.
- **Reconstruction error (E):** how poorly a token is approximated by a linear combination of the basis. ApET keeps the highest-E tokens.
- **Merging:** each discarded token is averaged into its most cosine-similar retained token.
- **Random pruning:** the same token counts at the same two points, chosen uniformly at random, with no basis, E or merging.
- **Answer preservation rate (APR):** among questions the uncompressed model answers correctly, the share still correct after compression. A *loss* is a correct answer that becomes wrong. A *gain* is a wrong answer that becomes correct.

**RQ1 measures.** For each of the 576 tokens, we measure how much it affects the probability of the first token of the reference answer, in two ways:

- **Zeroing:** the drop in that probability when only this token is set to zero.
- **Retention:** the probability when only this token is kept, minus the probability when all visual tokens are removed. This is a one-token adaptation of the Information Horizon analysis [2].

The two measures ask different counterfactual questions. Neither is ground-truth importance, and neither scores the full answer.

**RQ2 comparisons.** We compare uncompressed inference, complete ApET and random pruning (three seeds) at both schedules on larger cohorts. Paired 95% intervals use 2,000 bootstrap draws over images.

## 4. Preliminary Experiments and Results

**Cohorts.** RQ1 uses a 30-question diagnostic panel over 24 images: 12 GQA, 12 MMStar and 6 TextVQA questions [4–6]. All 30 were answered correctly without compression. They were chosen from our own input-only ApET runs at 288 tokens to include 11 questions that ApET got wrong (losses) and 19 it preserved. The panel is therefore enriched for failures and is not a representative sample. RQ2 uses 800 GQA development questions (100 images), all 1,500 MMStar questions, and 100 TextVQA development questions. TextVQA is scored with the standard ten-annotator consensus credit.

### 4.1 Reproduction check

At matched average budgets, our ApET runs keep 92.4% (64 tokens) and 98.6% (192 tokens) of uncompressed GQA accuracy. The paper reports 91.9% and 97.3% [1]. On TextVQA, we keep 83.5% and 88.9%, versus 91.1% and 93.5% in the paper. Our absolute baselines also differ from the paper's: 69.0 versus 61.9 on GQA, and 49.7 versus 58.2 on TextVQA. These differences reflect different question subsets and scorers, so we compare only relative retention. For TextVQA there is a further difference. Our prompt contains only the image and question. LLaVA's reference launcher instead uses an OCR-annotated question file (`llava_textvqa_val_v051_ocr.jsonl`), which we could not obtain locally. OCR text in the prompt would let the model read text without relying on visual tokens. This could plausibly raise both the baseline and the retention under compression, but we have not tested it. With only 100 questions, sampling error alone is about ±5 points.

### 4.2 Finding 1 (RQ1): reconstruction error does not rank answer sensitivity

![Per-question Spearman correlations](figures/rq1_correlation_strip.png)

**Figure 2.** Per-question Spearman correlations over all 576 positions. Each dot is one question; bars are medians.

Median correlations between E and either measure lie between −0.14 and +0.01 in every dataset (Figure 2). Individual questions range from −0.49 to +0.40. E is therefore not uninformative for every question, but the direction of its relationship to answer sensitivity is inconsistent. The two answer measures also disagree with each other (medians −0.03 to −0.01), so "token importance" depends on how it is measured. Excluding padding and basis positions moves the GQA median for E versus retention to +0.09, but does not change the overall picture. Adding a second measurement point after block 16, tested on six questions, did not consistently improve agreement.

![Same image, same E, different answer responses](figures/same_image_comparison.png)

**Figure 3.** Two GQA questions about the same image share an identical E map, yet produce different retention responses. For "Window", a handful of tokens on the right side sharply raise the answer probability. For "Bottom", the model already assigns 0.72 to the first piece "B" with no visual tokens, so no single token adds much. Neither pattern follows the E map. Note also the low E on the gray padding columns.

E is computed before the question is seen, so it cannot adapt to which part of the image a question needs. The two questions also differ in target answer and prior confidence, so this figure shows question dependence; it does not isolate the effect of wording.

### 4.3 Finding 2 (RQ2): complete ApET beats random pruning where fine detail matters

**Table 2.** Scores (%) by dataset. Random is the mean ± sample SD over three seeds. Intervals are paired 95% bootstrap intervals over images. "Lost" counts correct answers that became wrong (TextVQA: full-credit answers).

| Dataset (n) | Schedule (avg. tokens) | Uncompressed | ApET (share kept) | Random | ApET − random [95% CI] | Lost: ApET / random |
|---|---|---:|---:|---:|---:|---:|
| TextVQA (100) | 96→32 (64) | 49.70 | 41.50 (83.5%) | 20.63 ± 3.12 | +20.87 [13.93, 29.03] | 10 / 28.7 of 44 |
| TextVQA (100) | 288→96 (192) | 49.70 | 44.20 (88.9%) | 34.77 ± 4.80 | +9.43 [3.20, 15.43] | 7 / 16.0 of 44 |
| GQA (800) | 96→32 (64) | 69.00 | 63.75 (92.4%) | 61.29 ± 0.79 | +2.46 [0.25, 4.71] | 78 / 97.3 of 552 |
| GQA (800) | 288→96 (192) | 69.00 | 68.00 (98.6%) | 66.83 ± 0.62 | +1.17 [−0.58, 2.83] | 34 / 44.7 of 552 |
| MMStar (1500) | 96→32 (64) | 34.73 | 33.40 (96.2%) | 33.53 ± 0.53 | −0.13 [−1.35, 1.07] | 79 / 76.0 of 521 |
| MMStar (1500) | 288→96 (192) | 34.73 | 35.00 (100.8%) | 35.20 ± 0.29 | −0.20 [−1.25, 0.85] | 36 / 39.0 of 521 |

ApET's advantage is largest on TextVQA, where answers depend on small text regions. At 96→32, it preserves 77% of correct answers, compared with 35% for random pruning. The advantage comes from **fewer losses**. Gains are nearly identical for ApET and random in every row except MMStar 288→96 (for example, 3 versus 3.0 on TextVQA and 36 versus 35.7 on GQA at 96→32). Aggregate accuracy also hides many changed answers: even at GQA 288→96, the 1-point net drop combines 34 losses and 26 gains.

MMStar has little room to show compression effects for this model. The uncompressed baseline (34.7%) is close to chance, which is about 27% for its mostly four-option questions. The model predicts "A" on 70.6% of questions, although only 28.6% of the correct answers are A.

### 4.4 Open question: where does ApET's advantage come from?

Findings 1 and 2 are not contradictory. Finding 1 concerns how individual tokens are ranked. Finding 2 concerns the whole procedure. Random pruning removes three things at once: E-ranking, the FPS basis and merging. Two explanations remain open:

- **H1 (coarse filtering):** E reliably discards padding and flat background, which are easy to reconstruct, even though it ranks the remaining tokens poorly. Figure 3's low-E padding and the GQA median turning positive when padding is excluded are consistent with this.
- **H2 (coverage and merging):** the spatially diverse basis and the merged averages preserve a summary of the whole image that random pruning loses.

The existing diagnostic panel cannot separate these explanations. We ran the 30 panel questions with ApET's basis and merging fixed and randomized only the other retained tokens. Random retention preserved 23–27 of 30 answers, versus 19–21 for E-ranking. However, the panel was selected from ApET's own outcomes in this same setting, so ApET's 11 losses are failures by construction, and any other policy is expected to recover some of them. The panel therefore cannot rank E-ranking against random retention. The same applies to comparing ApET with and without merging. Within the random arms, which played no role in selection, disabling merging changed preserved answers from 74/90 to 77/90 across seeds, too small a difference to interpret.

One case shows that merging can cause a specific failure. With input-only compression, ApET changes the wine brand "Brolio" to "Bottoli". Keeping the same retained tokens, either disabling merging or undoing the merge for one retained token restores "Brolio". Control restorations do not. This case was chosen after the fact and shows only that such failures exist.

### 4.5 First RQ3 feasibility check

An adaptive budget would need to predict which images ApET will fail on. Using saved E values (no new GPU runs), a fixed three-feature model of image-level E statistics, evaluated with five image-grouped folds, gave out-of-fold AUROCs of 0.41–0.57. It showed no improvement over predicting the base rate on any dataset. Mean E alone was modestly associated with losses on MMStar (AUROC 0.61–0.64), but not on GQA. More fundamentally, at 96→32, 56 of the 100 GQA images had both lost and preserved questions under the same compression. An image-level signal cannot separate those cases. This agrees with Finding 1: failures depend on the question. The current evidence does not justify an image-level adaptive budget.

**Limitations.** The RQ1 panel is small and enriched for failures, so its correlations describe these 30 questions. Answer sensitivity is measured by first-token probability under two specific interventions. Depth coverage is limited to two points on six questions. The RQ2 cohorts are development subsets, and TextVQA has only 100 questions. Bootstrap intervals reflect question sampling and three random seeds. They do not capture run-to-run numerical variation, which changes a few compressed answers when runs are repeated from scratch. We make no speed claims because we have not run controlled timing.

## 5. Roadblocks, Risks and Mitigations

Dependency drift on Kaggle was the main early roadblock. The locked environment resolves it. The remaining issues are scientific. Different interventions rank tokens differently. Fresh compressed runs occasionally change answers for reasons we have not identified, although replays from saved features are exact. The reference TextVQA question file is unavailable locally. Likelihood and impact below are qualitative judgments.

**Table 3.** Risks.

| Risk | Likelihood / impact | Status and action | Contingency |
|---|---|---|---|
| Environment drift | Low / Medium | Green: locked environment works | Rebuild from the locked specification |
| Answer-sensitivity measures disagree | High / High | Yellow: two measures reported side by side | Restrict claims to each defined measure |
| Selection bias in diagnostic panels | High / High | Yellow: identified; panel used only for RQ1 and case studies | Run all component comparisons on unselected cohorts |
| Fresh-run variation | Medium / Medium | Yellow: replays exact; fresh changes logged | Repeat only cases a conclusion depends on |
| RQ2 attribution stays unclear; no RQ3 candidate | Medium / Medium | Yellow: candidate directions listed (Sec. 6) | Report a well-controlled negative result |

## 6. Updated Plan and Milestones

The main comparisons ran earlier than planned. The second half therefore begins with a decision rather than a fixed experiment: which open question to pursue next, and with what method. Making that choice well requires more analysis of the existing results. Directions currently under consideration include:

- **Separating H1 and H2 (RQ2).** One option is a basis-fixed random arm, which keeps ApET's basis and merging but randomizes the other retained tokens, so only E-ranking is removed. Another is a content-only random arm, which prunes randomly among non-padding positions. TextVQA would be the natural main cohort, because the ApET–random gap there is large enough to detect.
- **Question-dependent failures.** Finding 1 and Section 4.5 both suggest that failures depend on the question, not only the image. Analyzing which kinds of questions ApET loses may point to a different RQ3 direction than an image-level budget.
- **Merge-induced failures.** Estimating how often Brolio-type failures occur in unselected cohorts would show whether changes to merging are worth pursuing.

None of these is committed. Whichever direction is chosen will be run on unselected cohorts, with its conditions and analysis fixed in writing before any runs.

**Table 4.** Milestones.

| Target | Deliverable and completion criterion |
|---|---|
| Oct 30 | Submit report PDF and ~five-slide deck |
| Nov 6 | Choose the next-stage direction and method after further analysis of existing results; record the choice and rationale, or why existing evidence is insufficient |
| Nov 13 | Run the chosen comparison, if feasible: scores, APR, losses/gains, paired intervals, replay checks |
| Nov 20 | RQ3 go/no-go: justify one modification, or document why none is supported |
| Dec 15 (tentative) | Final presentation and report/code submission |

Whichever direction is chosen, the same evaluation standards apply: matched token counts at both schedules, three random seeds where applicable, paired intervals, and losses and gains reported alongside accuracy. If an RQ3 modification is chosen, it will be compared with fixed-budget ApET at the same average token count on held-out GQA questions not used during development. An informative negative result is an acceptable outcome. No accuracy improvement is promised.

## 7. References

1. Q. Ma et al. [**ApET: Approximation-Error Guided Token Compression for Efficient VLMs.**](https://arxiv.org/abs/2602.19870) CVPR, 2026.
2. Y. Wang et al. [**When Token Pruning is Worse than Random: Understanding Visual Token Information in VLLMs.**](https://arxiv.org/abs/2512.07580) CVPR, 2026.
3. H. Liu et al. [**Improved Baselines with Visual Instruction Tuning.**](https://arxiv.org/abs/2310.03744) CVPR, 2024.
4. D. A. Hudson and C. D. Manning. [**GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering.**](https://arxiv.org/abs/1902.09506) CVPR, 2019.
5. L. Chen et al. [**Are We on the Right Way for Evaluating Large Vision-Language Models?**](https://arxiv.org/abs/2403.20330) 2024.
6. A. Singh et al. [**Towards VQA Models That Can Read.**](https://arxiv.org/abs/1904.08920) CVPR, 2019.
7. A. Paszke et al. [**PyTorch: An Imperative Style, High-Performance Deep Learning Library.**](https://arxiv.org/abs/1912.01703) NeurIPS, 2019.
8. T. Wolf et al. [**Transformers: State-of-the-Art Natural Language Processing.**](https://aclanthology.org/2020.emnlp-demos.6/) EMNLP: System Demonstrations, 2020.
