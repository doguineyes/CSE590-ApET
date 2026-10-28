# ApET RQ1: Complete Experiment and Results Audit

**Project:** Understanding Reconstruction-Based Visual Token Compression: When and Why Does ApET Work?  
**Audit date:** October 28, 2026  
**Source:** [`doguineyes/CSE590-ApET`](https://github.com/doguineyes/CSE590-ApET), `apet-analysis` branch, reviewed at commit [`f65ec142`](https://github.com/doguineyes/CSE590-ApET/commit/f65ec1422a8421864530e8e3bfa38206b2b07083).  
**Evidence status:** This is a synthesis of tracked GitHub review notes, protocols, and compact exported tables. The large raw experiment ZIPs under `research/results/` are Git-ignored and were **not independently downloaded or rerun for this audit**. Their verification and numerical results are reported from the repository's documented audits. No new model inference was performed.

## The direct answer: Was RQ1 tested beyond 30 selected questions?

**Yes, we studied larger datasets, but not with the same full RQ1 token-sensitivity experiment.** The distinction matters:

1. **The primary RQ1 answer-sensitivity experiment** measured *all 576 token positions* for **30 carefully selected questions across 24 images**: 12 GQA questions (6 images), 12 MMStar (12 images), and 6 TextVQA (6 images). We ran singleton **zeroing** and singleton **retain-one** interventions at the LLM input boundary. This produced **17,280 token-level scores for each complete intervention type**. The RQ1 correlation findings reported in the midterm are based on these 30 questions, **not** a random or representative dataset-wide sample.
2. **A depth extension** measured all 576 positions for **six of those same questions** at the boundary after LLM block 16. This was a matched two-boundary study, **not** an additional six independent questions and not a sweep through every decoder layer.
3. **Additional RQ1 checks** investigated high-/middle-/low-reconstruction-error groups, six same-image GQA question pairs, alternative perturbations on six questions × 32 positions, first-token versus whole-answer scoring, and limited MMStar visual-grounding checks.
4. **Much larger later experiments** used **TextVQA100, GQA800, and MMStar1500: 2,400 questions** to compare uncompressed LLaVA, complete ApET, and pure random pruning. These saved full input reconstruction-error (`E`) arrays and decoder-stage `E` arrays. We later related **image-level `E` summaries** to whether compressed answers failed. **But we did not run all-576-position zeroing/retention on all 2,400 questions.** Those larger runs primarily support **RQ2 (full-compression outcomes)** and **exploratory RQ3 feasibility**, not a large-scale replication of RQ1's per-token `E`–sensitivity correlations.

In short, **30 questions is the correct denominator for the full RQ1 token-level finding; 2,400 questions is the larger answer-outcome dataset, with different measurements and conclusions.**

---

## 1. RQ1 question and what was operationally measured

The proposal's RQ1 asks:

> How does reconstruction-based redundancy relate to a token's contribution to the answer, across tasks and layers?

We studied that question in **LLaVA-1.5-7B** with native ApET error calculation. At the input stage, the vision encoder produces **576 contextual visual features**, with **10 farthest-point-sampled (FPS) basis features** used to estimate how well each feature can be represented as a linear combination of the basis. Define the reconstruction error (schematically) as:

\[
E_i = \|h_i-\widehat{h}_i(B)\|_2.
\]

`E` describes a **feature-space linear reconstruction residual**, not a measured effect on the final answer. The original implementation uses a regularized solve; this schematic formula does not specify its coefficients or numerical regularization.

We compared `E` against two **different operational answer-response measures**:

| Measure | Intervention | Recorded response | Interpretation and caveat |
|---|---|---|---|
| **Singleton zeroing**, `I_zero` | In the full 576-state context, set **one** visual feature to zero, preserving sequence length and positions | `p(first gold token | full) − p(first gold token | zero token i)` | How the fixed target probability responds to zeroing this one state in full context |
| **Singleton retain-one**, `I_IH` | Physically retain **only one** visual state, preserving its original position information; compare with a *remove-all-visual-states* condition | `p(first gold token | retain i) − p(first gold token | remove all)` | How much that state changes the fixed target probability relative to this particular null |

Here, *gold token* means the **first tokenizer piece of the frozen reference answer**. For `Dominion`, for example, the scored piece is `▁Domin`, not the complete word. The original zeroing analyses also archived whole-answer teacher-forced mean log-probability responses; some early figures and group tests use that **different score definition**. All matched RQ1 table entries below use **first-piece signed probability effects**.

At the input stage, `E` is measured in CLIP's pre-projector feature space; the zeroing/retention interventions are on corresponding **projected LLaVA visual inputs**. At the after-block-16 boundary, `E` and interventions act in the decoder representation. Neither intervention is the same as native ApET's joint selection, FPS basis handling, merging, and sequence shortening. Neither establishes intrinsic or ground-truth token importance.

Primary numerical summary: [Input retain-one full review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-18-rq1-input-ih-full-review.md) · [RQ1 consolidation](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-18-rq1-consolidation-and-scope.md) · [frozen input retain-one protocol](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/protocols/rq1_input_ih_v1.json).

## 2. How the 30 questions were selected

The panel was curated using prior **uncompressed 576-token vs. native input-only 288-token** generation outcomes, **before** the RQ1 `E`/intervention measurements were used for the panel. All selected questions were **correct in the uncompressed baseline** and were intentionally enriched for compression failures:

| Dataset | Candidate/review history | Final RQ1 panel | Historical input-only compression outcome in the final panel |
|---|---|---|---|
| GQA | 109 unique candidates reviewed across rounds; 49 in the primary reviewed pool | **12 questions / 6 images** (six question pairs) | **4 losses**, 8 preserved |
| MMStar | 48 unique candidates across rounds; 18 in the reviewed primary pool, plus separate qualitative cases | **12 questions / 12 images** | **4 losses**, 8 preserved |
| TextVQA | A frozen **100-question** screening pool; selected examples nested in it | **6 questions / 6 images** | **3 full-credit losses**, 3 preserved |
| **Total** | Larger preparation pools are **not** the intervention sample | **30 questions / 24 images** | **11 losses**, 19 preserved |

The candidate pools and the TextVQA100 screening are real work, but **having prepared or screened a question is not the same as having measured its 576 singleton intervention responses**. This is an exploratory, outcome-enriched mechanism panel, not an unbiased estimate for all GQA/MMStar/TextVQA questions. The review includes assistant-assisted, unblinded candidate selection; it is not an independently human-validated benchmark.

Sources: [dataset preparation](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-15-rq1-dataset-preparation-summary.md) · [selection funnel and panel provenance](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/README.md) · [panel cases CSV](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/panel_cases.csv).

## 3. Experiment inventory: what actually ran

**Do not add calibration/replay counts as if they were new independent questions.** The stages below frequently reuse the same six or thirty questions.

| Stage / date | Questions | Token-level extent | Purpose | Status |
|---|---:|---|---|---|
| **Input singleton zeroing**, finalized Oct 16 | **30** (24 images) | **30 × 576 = 17,280** positions | Compare original input `E` with teacher-forced answer response to zeroing one post-projector state | Complete; original measure was whole-answer log score, later rescored for matched first-piece probabilities |
| **High/mid/low `E` group and paired-question analysis**, Oct 17 | Same **30** | Reanalysis of saved 17,280 zeroing effects; **no new GPU inference** | Test group trends and same-image maps; inspect basis/padding/selected-slot roles | Complete, exploratory |
| **First-piece/whole-answer reanalysis**, Oct 17 | Same **30** | Recompute score summaries from archived per-target-token records | Test whether target-token scoring explains apparent weak correlations | Complete, post-hoc |
| **Perturbation and grounding validity experiment**, Oct 17 | **6** questions for perturbations, **6** MMStar cases for grounding checks | 6 × 32 positions × 3 operators = **576 operator/position scores**, plus 72 image/option trials and separate repeats | Zero vs half-strength vs neighbor replacement; image/blank/question-only and answer-option rotations | Complete, targeted |
| **After-block-16 zeroing and retain-one**, Oct 17/18 | **6 of the 30** | 6 × 576 = **3,456** layer-zeroing and **3,456** layer-retain-one scores; another 3,456 input-zeroing values rerun for parity | Compare input and layer-16 `E`/sensitivity maps; test sparse high-response states | Complete, two boundaries only |
| **Input retain-one full sweep**, Oct 18 | Same **30** | **30 × 576 = 17,280 new input retain-one scores**, plus exact zeroing audits/replays | Match `E` with both zeroing and retention effects on all 576 positions | Complete; definitive bounded RQ1 matched table |
| **Larger staged ApET vs. random**, Oct 21 | **2,400** across three cohorts | Input/decoder `E` logged with full-compression outcome records; **no 576-way answer interventions per question** | RQ2 operator comparison; reused for exploratory RQ3/image-level loss association | Complete for those cohorts, but **not** a larger token-sensitivity RQ1 sweep |
| **Large-cohort `E`/loss feasibility reanalysis**, Oct 25 | Saved **2,400** question outcomes, loss analyses conditional on baseline correctness | Image-level `E` statistics + five-fold image-group prediction; **no new GPU inference** | Explore whether reconstruction statistics predict later ApET failures | Complete exploratory reanalysis; RQ3-related, not evidence for token-level `E`–`I` alignment |

Core source notes: [input zeroing checkpoint](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-16-rq1-input-stage-summary-and-next-steps.md) · [saved-group analysis](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-saved-data-analysis.md) · [score reanalysis](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-score-reanalysis.md) · [validity review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-validity-review.md) · [full layer review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-layer-ih-full-review.md).

## 4. Main RQ1 quantitative results: all 30 questions at input

For each question, Spearman's rank correlation was computed **across its 576 visual positions**. The table aggregates the **per-question correlations** by dataset using the median; it does **not** pool all tokens into one giant independent sample.

| Dataset | Questions / images | Median ρ(`E`, zeroing) | Median ρ(`E`, retain-one) | Median ρ(zeroing, retain-one) | Positive `E`–retain-one per-question correlations |
|---|---:|---:|---:|---:|---:|
| GQA | 12 / 6 | **−0.087** | **−0.068** | **−0.016** | 6/12 |
| MMStar | 12 / 12 | **+0.009** | **−0.121** | **−0.005** | 4/12 |
| TextVQA | 6 / 6 | **−0.031** | **−0.136** | **−0.028** | 1/6 |

Across the 30 questions, `E`–retain-one correlations range from **−0.486 to +0.404**. Eleven are positive and nineteen negative. The result is **mixed alignment**, not that every question has zero correlation and not that `E` is always negatively related to useful information.

**Primary interpretation:** In these curated examples, native `E` does **not consistently rank signed reference-answer support** measured by either singleton intervention. This is a property of the **specified measurements and cohort**. It does **not** establish that `E` is useless for ApET or that the full compression method cannot outperform random pruning.

### 4.1 Robustness and sensitivity checks

- Excluding ten FPS basis sources per image gives median `E`–retain-one correlation **−0.044 (GQA), −0.115 (MMStar), −0.117 (TextVQA)**.
- Further restricting to **photo-only, nonbasis** positions changes these to **+0.087 (GQA), −0.065 (MMStar), −0.039 (TextVQA)**. Thus filtering **can change signs**; it does not reveal consistently strong positive alignment.
- Using **absolute** rather than signed retain-one effects gives different median correlations: **+0.008 (GQA), +0.113 (MMStar), +0.109 (TextVQA)**. Signed answer support and magnitude of intervention response are different scientific targets.
- For the earlier original **whole-answer log-score singleton zeroing**, high-`E` versus low-`E` mean-response comparisons were positive in only **2/12 GQA, 6/12 MMStar, and 3/6 TextVQA** questions. These are averages of **separate singleton effects**, *not* a measurement from simultaneously removing all high- or low-`E` tokens.
- For TextVQA, measuring the whole answer instead of only the first tokenizer piece sometimes changes rankings substantially. The `Cookery school` answer is a strong example: different subwords give different signed responses. Single-token targets do not have that particular discrepancy.

Sources: [full matched input review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-18-rq1-input-ih-full-review.md) · [group analysis](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-saved-data-analysis.md) · [scoring audit](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-score-reanalysis.md) · [30-question correlation CSV](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/rq1_correlations_30.csv).

## 5. Same image, different questions: six paired GQA examples

All six GQA pairs had matching input features, basis/error values and their underlying image encoding. **The `E` ranking was fixed for both questions about the image**, but the answer-response maps could be very different.

| Shared image | Two target answers | Between-question **original whole-answer zeroing** ρ over 566 nonbasis positions | Shared top-57 response positions |
|---|---|---:|---:|
| `n200692` | No / Stove | **−0.834** | 0 of 57 |
| `n455563` | No / Right | +0.505 | 13 of 57 |
| `n64959` | Right / Stove | +0.577 | 27 of 57 |
| `n125122` | Television / No | −0.138 | 2 of 57 |
| `n319845` | No / Table | +0.319 | 9 of 57 |
| `n578564` | Bottom / Window | **−0.332** | 0 of 57 |

Note: this table is the **historical whole-answer zeroing** same-image comparison from the saved-data analysis. The later matched **retain-one** between-question correlations across the six GQA pairs range from **−0.443 to +0.655**; those are **different maps and not the same numbers**.

**Interpretation:** The same image and `E` ranking can produce different answer-response patterns. However, the question, target-answer token(s), baseline confidence and sometimes sequence geometry change together, so this is **question/target-conditioned behavior**, not pure causal isolation of question wording or proof that compression will fail on either question.

Example: `n578564` asks about a pan in front of a **Window**, and where paper towels lie (**Bottom**). Their input `E` arrays are identical. For the retain-one experiment, the null probability for `▁Window` is **0.01390**, but for the first `▁B` piece of “Bottom” it is **0.71827**. Differences in null priors matter when interpreting the maps.

Source: [same-image analysis](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-saved-data-analysis.md) · [paired-figure provenance](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/README.md) · [illustration](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/same_image_pair/same_image_comparison.png).

## 6. Selected depth comparison: six questions, two boundaries

The six questions were:
- GQA `202218829` (**Window**) and `202218593` (**Bottom**) — same image.
- MMStar `895` (**Bedroom/B**) and `136` (**Soccer/B**).
- TextVQA `35962` (**Cookery school**) and `35670` (**Dominion**).

The two measurement boundaries were **before the first LLM block** (input) and **after block 16 / before block 17**. At layer 16 we measured all 576 positions for **zero-one** and **retain-one** interventions, each with the corresponding layer's `E`. The input-stage comparisons for those questions were reused/repeated for identity checking.

| Measured feature | Observed result |
|---|---|
| Input vs. layer-16 `E` ranking agreement | Spearman **about +0.59 to +0.79** across six matched cases |
| Input vs. layer-16 zeroing response ranking agreement | **about −0.04 to +0.30** (historical whole-answer comparison) |
| Input vs. layer-16 retain-one response ranking agreement | **about −0.04 to +0.20** |
| `E` vs. signed answer response at layer 16 | **No consistent improvement** relative to input; individual cases vary |

The reconstruction ranking is relatively more stable across these two representation boundaries than the measured response rankings. This is **not** evidence that answer relevance always decreases with depth, nor a complete 32-layer curve. The representations, interventions and `E` scales are different at the two boundaries.

### Sparse, strong retained states missed by calibration

A previous **32-position calibration** often missed the strongest retained state in the full **576-position** grid:

| First-piece target | Earlier 32 positions: best retained first-piece probability | Full 576 positions: best retained probability | Boundary |
|---|---:|---:|---|
| **Dominion** | 0.00150% | **20.19%** | Input |
| **Dominion** | 0.0768% | **98.10%** | After block 16 |
| **Window** | 12.12% | **94.06%** | After block 16 |
| **Cookery school** (`▁Cook`) | 15.24% | **74.19%** | After block 16 |

At layer 16, **Dominion's state 131** had very large positive retain-one response (`I_IH` ≈ **+0.9804**) despite `E` at only the **11.3rd percentile** of nonbasis states. That is a **low-`E`/strong-response exception**, not a universal inverse relationship. Other high-response states have high `E`, so one cannot infer a simple consistent low-`E` rule either.

At this late boundary, text states have **already processed the image in preceding LLM layers**. “Remove all visual states at layer 16” is therefore not the same as “the model saw no image.” Do not interpret it as an ordinary question-only baseline. The peak retained probability describes the **first gold tokenizer piece**, not freely generated answer accuracy.

Sources: [full after-block-16 study](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-layer-ih-full-review.md) · [depth interpretation](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-18-rq1-consolidation-and-scope.md) · [depth retain-one protocol](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/protocols/rq1_layer16_ih_v1.json).

## 7. Validity experiments: robustness is not the same as a universal importance ranking

### 7.1 Different perturbations

Six questions × **32 positions** were tested with (i) full zeroing, (ii) half-strength replacement, and (iii) nearest-neighbor-feature replacement. The 32 positions per question included a **16-position random stratum** and targeted high-`I` and low-`E`/high-`I` strata. Only the random stratum supports the less-selected rank-comparison summaries.

The zeroing results reproduced, but the half/neighbor responses sometimes **reordered tokens or changed signs**. For example, for GQA **Window**, position 211 had:

| Intervention | Signed reference-score effect |
|---|---:|
| Full zeroing | **+0.30548** |
| Half-strength | **−0.12530** |
| Neighbor replacement | **+0.23267** |

These are intervention-specific effects, not a demonstration of universal numerical instability, mathematical discontinuity, or a defect in zeroing. A changed teacher-forced reference score also does not prove that the model's freely generated answer changed.

### 7.2 Grounding/answer-choice checks

The validity run also performed **72 image/option trials** on **six selected MMStar questions**, using real/blank/question-only images and cyclic option rotations, plus six reference repeats. Five diagram-related cases raised concerns about whether the original correct multiple-choice output reflected visual grounding; the **soccer** control consistently followed the correct answer content across the four real-image option rotations. These are **flags about particular diagnostic examples**, not a representative MMStar conclusion.

Source: [validity review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-validity-review.md).

## 8. What the *larger* dataset analysis really measured

After the RQ1 full token-intervention phase, the project ran three broader staged compression studies:

| Cohort | Questions | Distinct images or image hashes | What was saved for later `E` analysis |
|---|---:|---:|---|
| **TextVQA100** | 100 | 100 | Input 576-position `E` arrays; FPS basis indices; decoder-stage `E` arrays for both schedules; compressed/uncompressed answer outcomes |
| **GQA800** | 800 | 100 (8 questions per image) | Same kinds of arrays; input encoding shared across each image's eight questions; per-question decoder-stage arrays |
| **MMStar1500** | 1,500 | 1,430 distinct image-byte hashes | Same kinds of arrays; repeated image bytes can have distinct per-case encoding seeds |
| **Total** | **2,400** | Not 2,400 independent images | **1,700 encoding records**, each represented at two staged compression schedules |

The [availability inventory](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/E_availability.json) reports **200 + 200 + 3,000 = 3,400** input-`E` array records across the two schedules, each with 576 positions. It also records per-question decoder-`E` arrays (200 TextVQA + 1,600 GQA + 3,000 MMStar = **4,800 decoder-`E` records**) at the two compression schedules. Many input arrays are duplicated across schedule arms; they are not independent `E` measurements. Input `E` and basis indices match exactly between schedules for all 1,700 corresponding encoding records.

### 8.1 The exploratory `E` → compression-loss analysis

From the saved large-cohort records, we asked a related but **different** question:

> Can simple **image-level statistics of input reconstruction error** predict when **native ApET will lose a formerly correct answer**?

The analysis used only questions correct under the uncompressed baseline, with **three image-level `E` features**: log mean error, coefficient of variation, and the fraction of error mass in the top 10% of positions. It used **five image-grouped folds** and out-of-fold (OOF) prediction, without new GPU inference.

| Cohort | Compression schedule | Eligible baseline-correct questions / ApET losses | Three-feature OOF AUROC |
|---|---|---:|---:|
| TextVQA | 96→32 | 44 / 10 | **0.568** |
| TextVQA | 288→96 | 44 / 7 | **0.498** |
| GQA | 96→32 | 552 / 78 | **0.477** |
| GQA | 288→96 | 552 / 34 | **0.410** |
| MMStar | 96→32 | 521 / 79 | **0.568** |
| MMStar | 288→96 | 521 / 36 | **0.562** |

An AUROC near **0.5** is chance-level ranking. The reported model did not show a clear Brier-score improvement over simply predicting the baseline loss rate; all six Brier improvement intervals include zero. This does **not support a robust cross-dataset image-only rule for predicting lost answers**, but it does not prove that no other feature or adaptive budget could work.

The analysis also found that in GQA at 96→32, **56 of 100 images** had both baseline-correct answers that were preserved and baseline-correct answers that were lost **under the same shared image encoding**. That observation motivates question-conditioned analysis: an image-only statistic cannot distinguish different outcomes for those questions when its inputs are identical.

**Important classification:** This is a **larger-scale observational `E`/failure-risk feasibility analysis** (best described under RQ3 motivation and RQ2 outcomes). It is **not** a larger-scale measurement of `ρ(E_i, I_i)` because individual visual-token zeroing/retention responses `I_i` were **not** computed for those larger cohorts.

Sources: [large-cohort evidence README](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/README.md) · [availability inventory](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/E_availability.json) · [full feasibility metrics](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/e_loss_feasibility_metrics.csv) · [GQA staged review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-gqa800-review.md).

## 9. Consolidated findings: what we know, what we do not

| Supported by the experiments | Not established |
|---|---|
| The measured input `E` rankings have **weak/mixed alignment** with **signed first-piece token-response measurements** on the curated 30-question panel | That `E` is useless, or that all visual tokens have one true universal importance ranking |
| Same-image GQA questions can have different zeroing/retain-one maps despite the same input `E` | That question wording alone caused the difference or that a specific token necessarily caused an ApET error |
| At the six tested cases, the two boundaries show relatively more stable `E` rankings than response rankings | A universal law over all 32 decoder blocks, or that question relevance monotonically increases/decreases with depth |
| The 32-position pilots sometimes missed very strong retain-one responses discovered in complete 576-position sweeps | That every large retain-one first-piece probability yields a correct freely generated answer |
| Different interventions and answer-score definitions can reorder response ranks | That one intervention is invalid, or that the measurements are random noise |
| Large cohorts contain detailed `E` logs and answer loss/gain records; simple image-level `E` features weakly predict loss | A 2,400-question **token-level** `E`–`I` correlation test, or a proven adaptive-budget improvement |

**The central methodological limitation:** The RQ1 panel was outcome-enriched and not representative. Tokens within an image, questions about one image, and repeated experiments on the same questions are not independent statistical samples. The reported `E`–`I` correlations should not be generalized as population estimates without new, independently selected question-level intervention measurements.

**How this connects to RQ2:** Native ApET selects features by *linear reconstructability* but actually combines discarded information by *cosine-based merging and averaging*. Individual token zeroing/retention does not reproduce that operation. We therefore cannot infer ApET's aggregate success or failure directly from the RQ1 correlations. Controlled RQ2 studies of selection, basis handling, and merging remain necessary.

## 10. Recommended wording for a report or presentation

**Short (plain English):**

> For RQ1, we measured how each of 576 visual tokens affects the reference-answer probability on 30 selected questions, using two different interventions. Reconstruction error did not consistently predict these effects. The larger 2,400-question experiments recorded reconstruction errors and compression outcomes, but did not repeat the full token-by-token sensitivity test, so our RQ1 conclusion remains exploratory.

**More precise (academic):**

> On an outcome-enriched diagnostic panel of 30 questions from 24 images, we compared native visual-token reconstruction error with signed answer sensitivity under singleton zeroing and a 1×1 retain-one adaptation. Median per-question Spearman correlations were weak or mixed across GQA, MMStar and TextVQA. A matched six-question input/after-block-16 study found no systematic improvement at the deeper boundary. Separately, saved `E` statistics from 2,400 broader compression-evaluation questions supported an exploratory image-level failure-prediction analysis, but did not supply token-level `E`–sensitivity correlations on that larger sample.

## 11. GitHub source map and raw evidence locations

**Tracked, directly readable source files** (stable links on the project branch):

1. [RQ1 consolidated scientific scope and findings](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-18-rq1-consolidation-and-scope.md)
2. [Canonical 30-question input zeroing results](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-16-rq1-input-stage-summary-and-next-steps.md)
3. [Canonical 30-question input retain-one results and correlations](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-18-rq1-input-ih-full-review.md)
4. [Matched six-question layer-16 study](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-layer-ih-full-review.md)
5. [High-/middle-/low-`E` and same-image analyses](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-saved-data-analysis.md)
6. [Intervention and visual-grounding validity](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-validity-review.md)
7. [Whole-answer versus first-piece scoring audit](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-17-rq1-score-reanalysis.md)
8. [Frozen 30-case manifest and exact per-question correlations](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/rq1_correlations_30.csv)
9. [Larger-cohort `E` inventory](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/E_availability.json)
10. [Larger-cohort `E` / outcome feasibility evidence](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/README.md)
11. [TextVQA100 staged results](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-textvqa100-review.md) · [GQA800](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-gqa800-review.md) · [MMStar1500](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-mmstar1500-review.md)

**Original raw artifacts**: Experiment ZIP/log pairs were archived locally under the project's Git-ignored `research/results/` directory and verified in the review notes. Representative canonical archive names:

- `rq1-input-ih-thirtycases-v1-20261018T140840768523Z.zip` — 30-question input retain-one measurement.
- `rq1-layer16-ih-sixcases-v1-20261018T014830278963Z.zip` — six-question layer comparison.
- `rq1-validity-all-20261017T204446432120Z.zip` — perturbation and grounding checks.

This Markdown file is a **local audit output**, not an additional experiment, and has **not** been committed or pushed to GitHub.
