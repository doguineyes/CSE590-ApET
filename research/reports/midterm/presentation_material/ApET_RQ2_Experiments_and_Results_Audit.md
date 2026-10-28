# ApET RQ2: Complete Experiment and Results Audit

**Project:** Understanding Reconstruction-Based Visual Token Compression: When and Why Does ApET Work?  
**Audit date:** October 28, 2026  
**Repository:** [`doguineyes/CSE590-ApET`](https://github.com/doguineyes/CSE590-ApET), `apet-analysis` branch, reviewed at commit [`f65ec142`](https://github.com/doguineyes/CSE590-ApET/commit/f65ec1422a8421864530e8e3bfa38206b2b07083).  
**Evidence basis:** Tracked GitHub experiment reviews, protocols, and exported compact data tables. Original large ZIP/log experiment archives are documented in the notes but stored under the Git-ignored `research/results/` directory; **I did not download or re-audit those raw archives**. All counts and checks below are attributed to the repository's saved review records. No new inference was run for this document.

## Executive answer: what did RQ2 actually establish?

RQ2 asks: **How much of ApET's effectiveness comes from reconstruction-error (`E`) selection, and how much from similarity-based merging?**

**We made substantial progress, but have not identified a universal additive split between selection and merging.** The completed work has two distinct evidence levels:

1. **Controlled input-stage component diagnostics:** On **30 deliberately selected, baseline-correct questions across 24 images**, we studied native merging versus no merging, error-ranked versus random *ordinary* retained positions with the FPS basis fixed, different routing/protection policies, and a single-target restoration. These identify conditional, case-level effects, but **cannot estimate general benchmark performance**, because the panel was enriched for previous ApET failures.
2. **Broader complete-procedure comparisons:** We evaluated **2,400 questions** (TextVQA100, GQA800, MMStar1500) using full-576 LLaVA, native **two-stage ApET**, and **pure random pruning** at two matched count schedules (three random seeds). This yields **21,600 primary predictions**, not 21,600 independent questions. The whole ApET procedure decisively outperforms the random baseline on the fixed TextVQA100 cohort, shows a smaller advantage on GQA, and has no clear accuracy advantage on MMStar. This study changes **selection, FPS/basis handling, and merging together**, so it does **not** isolate `E` selection alone.

The clearest qualitative result is that **similar aggregate accuracy can conceal many baseline-correct → wrong losses and baseline-wrong → correct gains**. For example, moderate-schedule GQA has **34 losses and 26 gains**, a net decline of only eight correct responses. MMStar moderate compression has **36 losses and 40 gains**, a net increase of just four.

**RQ2 status:** Bounded component and broad whole-procedure evaluations **completed**; end-to-end **mechanism attribution, stage interactions, and generalization to new evaluation cohorts remain open**.

---

## 1. Definitions: which versions of “random” and “merging” did we test?

### Native ApET

LLaVA-1.5-7B starts with **576 visual features**. Native ApET selects ten **farthest-point-sampled (FPS) basis representations**, computes approximate linear reconstruction residuals (`E`), and retains top-error positions. It inserts the FPS source features into designated retained slots, while merging unselected source features into nearest eligible retained features using **cosine-based routing and averaging**. The retained sequence is then passed to the downstream language model. Input and decoder-stage compression can both be enabled.

**Source:** [`llava/model/llava_arch.py`](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/llava/model/llava_arch.py) and [`llava/model/modeling_llama_x.py`](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/llava/model/modeling_llama_x.py).

| Term used in this audit | What is held fixed / changed | Scope |
|---|---|---|
| **Input-only 288** | Compress **576 → 288** before the decoder; later decoder compression disabled | All the early 30-question mechanism panels |
| **Native E merge** | Original E-based selection + native FPS/basis-slot handling + cosine merging | Baseline compressed condition in controlled panels |
| **E no-merge** | **Identical** E-selected positions and native FPS/basis mapping; omit ordinary-target averaging | Isolates an input-stage merging effect *conditional on the fixed selection and basis* |
| **Basis-fixed random ordinary selection** | Keep ten FPS basis sources and their native slot mapping; randomize the remaining 278 ordinary retained positions, with and without merging | Early crossed 30-case study; **not** pure random pruning |
| **Shuffled routing** | Same E selection and basis; change which dropped sources go to retained recipients, preserving recipient loads in the designated shuffles | Routing study |
| **Protected-target routing** | Same E selection; shield selected ordinary merge recipients, rerouting sources | Protection study |
| **Pure random pruning** | Match retained counts/stages, choose positions uniformly, **no FPS, E calculation, basis replacement, or merging** | Later 2,400-question complete-procedure comparison |

**Important:** Native ApET's “random basis” ablation mentioned in its paper is different from both our **basis-fixed random ordinary retention** and our **pure random pruning**.

### Two-stage schedules in the broad experiment

- **96 → 32:** 576 input visual features become **96 after input compression** and **32 after decoder block 16**, averaging **64 states across the 32 decoder layers**.
- **288 → 96:** Correspondingly **288 after input** and **96 after block 16**, averaging **192**.

The second stage recomputes its selection on the current decoder states. Neither schedule means “64/192 states immediately after the CLIP encoder” and neither should be conflated with the **input-only 288** component panel or the separate **input-only 64** calibration.

### Metrics

- **Accuracy:** frozen dataset-specific scorer, including TextVQA's **partial-credit ten-annotator consensus** for the main reported percentage.
- **Full-credit correctness:** binary TextVQA definition used for preservation/loss/gain counting; **not** the same as TextVQA's mean consensus score.
- **APR:** answer preservation rate = baseline-correct questions still correct after compression / baseline-correct questions.
- **Loss:** baseline correct → compressed wrong.
- **Gain:** baseline wrong → compressed correct.
- **Identity:** `compressed_correct = baseline_correct − losses + gains`.
- **Uncertainty:** paired **2,000-draw image-group bootstrap** for native-vs-random differences in the broad runs, conditioning on the fixed evaluated cohort, three random seeds, and first native realizations. TextVQA has one question per image, GQA groups eight questions per image, and MMStar groups by 1,430 distinct image-byte hashes. These intervals do **not** incorporate full-run numerical variation.

---

## 2. Chronological experiment inventory

Do **not** sum all these questions as new independent examples: experiments repeatedly revisit the same selected panel or reused calibration cases.

| Date / stage | Actual evaluated scope | Intervention or comparison | Outcome/status |
|---|---|---|---|
| **Oct 19: first input merge/no-merge attempt** | First question reached, **0 admitted questions** | Native merge vs no-merge, exact cross-execution feature parity gate | **Failed parity gate** due to post-merge feature-byte differences; preserved as diagnostic, **not** scientific outcome |
| **Oct 19: native parity diagnostic** | **1 GQA question**, 9 captures | Raw native, instrumented native, no-merge repeated captures | Numerical variation localized to **post-merging** with identical pre-merge state and assignments; exact GPU cause unproven |
| **Oct 19: v2 capture/replay + merge/no-merge** | **30 questions / 24 images**; 3 main conditions → **90 predictions** | Full576, native input-only288 merge, fixed selection/basis without merge | **19/30 vs 21/30** correct; no-merge rescues 2 cases |
| **Oct 19: crossed selection × merging** | Same **30**; 9 conditions → **270 predictions** | E selection and 3 basis-fixed random ordinary-position seeds, each merge/no-merge, plus baseline | Random selected sets often rescue selected cases, but merging effect depends on set |
| **Oct 19: Brolio one-target oracle** | **1 selected TextVQA question**, 6 primary conditions | Restore only one pre-merge retained target, matched control, sham | Restoring target **317** changes **Bottoli → Brolio**, while control target **141** does not |
| **Oct 19/20: routing/protection** | Same **30**; 10 conditions → **300 predictions** | Native nearest, no-merge, three load-preserving shuffles, high-E protection, three matched-count random protections, baseline | Routing changes many assignments but few answers; high-E protection is mostly **inactive** |
| **Oct 20: pure-random input-only calibration** | **6 selected questions / 5 images**; **66 primary predictions** | Full576, native input288/64 with/without merge, pure random at each retained count, 3 seeds | Verified native 576→64 is real; **native64 2/6 vs random64 4/6, 3/6, 2/6**; descriptive only |
| **Oct 21: staged two-boundary calibration** | **6 selected questions / 5 images**; **54 primary predictions** | Full576, native 96→32 and 288→96, corresponding pure-random 3-seed schedules | **Native 1/6 and 4/6**; random scores vary; system/control checks pass |
| **Oct 21: staged TextVQA100** | **100 questions / 100 images**; **900 predictions** | 9 arms: full576, 2 native schedules, 3 random seeds × 2 schedules | **Large native advantage** at both budgets; losses and gains persist |
| **Oct 21: staged GQA800** | **800 questions / 100 images**; **7,200 predictions** | Same 9-arm experiment | **Modest native advantage**, especially at 96→32; 34 losses and 26 gains at 288→96 |
| **Oct 21: staged MMStar1500** | **1,500 questions / 1,430 unique images**; **13,500 predictions** | Same 9-arm experiment | **No clear native advantage**; 288→96 has 36 losses and 40 gains |
| **Oct 22–25: saved-result synthesis** | Same **2,400 broad question records** | Recheck CSVs, bootstrap tables, score transitions and image-level E associations | No new GPU runs. Some E-based loss prediction belongs to **exploratory RQ3 feasibility**, not evidence that an RQ2 component was isolated |

Principal chronological reviews: [v1 parity failure](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-first-parity-failure.md) · [parity localization](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-native-parity-diagnostic-review.md) · [v2 merge/no-merge](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-v2-full-review.md) · [crossed study](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-crossed-full-review.md) · [routing](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-routing-full-review.md) · [input-only 64 calibration](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-20-rq2-pure-random-calibration-review.md) · [staged calibration](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-calibration-review.md).

---

## 3. Input-only 288 merge versus no-merge: what changed?

This was the first **admitted** controlled RQ2 experiment, using the same 30 baseline-correct questions from RQ1. All runs kept the **E-ranked retained positions, ten FPS basis sources and their assigned output slots** fixed. Only the ordinary retained-feature averaging step was disabled in the no-merge arm.

| Dataset | Full576 correct | Native E merge | E no-merge |
|---|---:|---:|---:|
| GQA | 12/12 | 8/12 | 9/12 |
| MMStar | 12/12 | 8/12 | 8/12 |
| TextVQA | 6/6 | 3/6 | 4/6 |
| **Selected-panel total** | **30/30** | **19/30** | **21/30** |

No-merge **rescued two** original answers and introduced **zero new correctness losses** on this panel: `gqa:201636988` (cake-versus-plate color) and `textvqa:37533` (**Brolio**). Nine baseline-correct cases were still wrong even without merging. This does **not** prove that merging generally hurts accuracy.

### Three illustrative TextVQA outputs (input-only288)

| Reference/full576 | Native merge | Same selection, no merge | Interpretation |
|---|---|---|---|
| **Brolio** | Bottoli | **Brolio** | Merge-dependent loss in this fixed configuration |
| **Cookery school** | Culinary school | Culinary school | Omitting ordinary merging does **not** fix this case |
| **Give up** | Gone | Go up | Merging changes the wrong answer; both remain incorrect |

The Brolio case is especially informative because source/retained target **317** was actually selected as an ordinary retained target; it was **not simply excluded**. Dropped sources **103, 109 and 331** merged into it. The failure cannot be explained solely as “ApET dropped the important token.” However, the global no-merge test modifies many recipients, so the initial rescue does not by itself locate a single causal representation.

**Source:** [v2 first merge/no-merge full review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-v2-full-review.md).

## 4. Crossed E/random ordinary selection × merging

Next we retained the **same ten FPS source features and native basis-slot placement**, but randomized the other **278 ordinary retained positions**. Each selected set was tested with ordinary merging on and off. The **three random seeds** were prespecified. This is a controlled partial selection ablation, **not** basis-free pure random pruning.

| Ordinary selection | GQA merge / no merge (of 12) | MMStar merge / no merge (of 12) | TextVQA merge / no merge (of 6) | Total merge / no merge (of 30) |
|---|---:|---:|---:|---:|
| **E-ranked (native basis)** | 8 / 9 | 8 / 8 | 3 / 4 | **19 / 21** |
| **Random seed 101** | 9 / 9 | 11 / 10 | 5 / 5 | **25 / 24** |
| **Random seed 202** | 7 / 10 | 11 / 11 | 5 / 6 | **23 / 27** |
| **Random seed 303** | 10 / 11 | 11 / 10 | 5 / 5 | **26 / 26** |

The random ordinary positions frequently preserved known failures that the E-selected arrangement did not, but the panel **intentionally includes earlier ApET failures**. These are not representative results and must not be used to claim that random pruning is generally superior. Random selected sets also caused new losses in some previously preserved cases.

### Merge on versus off, within exactly the same selected set

| Selected set | Cases **rescued** by enabling merge | Cases **lost** by enabling merge | Net correct-count change |
|---|---:|---:|---:|
| E-ranked | 0 | 2 | −2 |
| Random101 | 2 | 1 | +1 |
| Random202 | 0 | 4 | −4 |
| Random303 | 1 | 1 | 0 |

This is direct evidence that **merging's observed effect interacts with which tokens were selected**. It is neither universally harmful nor universally helpful, even within this small diagnostic cohort.

**Examples:** Random202 restores exact **Cookery school** with and without merging; the E arms answer **Culinary school**. All random no-merge arms restore **Give up**, while random202 merging loses it. For a GQA television question, E gives **Phone** with or without merging, whereas the random ordinary-position arms answer **Television**. For an appliance-location question, random101/202 merging produces **Dishwasher/Microwave** but their no-merge conditions remain correct. These are controlled examples, not a general ranking.

**Source:** [crossed full30 review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-crossed-full-review.md) · [fixed crossed protocol](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/protocols/rq2_crossed_input_v1.json).

## 5. Routing/shuffling/protection: can the merge rule be isolated?

With **native E selection and FPS bases fixed**, we compared nearest-cosine recipient routing, no-merge, three **incoming-count-preserving routing shuffles**, protection of **28 highest-E ordinary targets**, and protection of 28 random ordinary targets (three seeds). All retained **288 input states**.

| Policy | GQA /12 | MMStar /12 | TextVQA /6 | **Selected-panel total /30** |
|---|---:|---:|---:|---:|
| Full576 baseline | 12 | 12 | 6 | **30** |
| Native nearest-cosine merging | 8 | 8 | 3 | **19** |
| No merge | 9 | 8 | 4 | **21** |
| Shuffled101 | 9 | 8 | 3 | **20** |
| Shuffled202 | 9 | 8 | 3 | **20** |
| Shuffled303 | 8 | 8 | 3 | **19** |
| Protect highest E: 28 ordinary targets | 8 | 8 | 3 | **19** |
| Protect random101: 28 | 8 | 8 | 4 | **20** |
| Protect random202: 28 | 8 | 8 | 3 | **19** |
| Protect random303: 28 | 8 | 8 | 4 | **20** |

The shuffled arms changed **5,207 / 5,213 / 5,190** assignments (about 75% of the **6,912** total dropped-source assignments) while preserving each recipient's incoming count. Despite large routing changes, comparatively few answers changed. **This does not prove that semantic/cosine routing never matters; it describes this selected panel.**

### Why the 28-high-E protection test barely acted

| Protection policy | Images with **zero** changed assignments | Total rerouted dropped sources |
|---|---:|---:|
| Highest-E 28 | 15 / 24 | **21 / 6,912 = 0.304%** |
| Random protection 101 | 0 / 24 | 980 |
| Random protection 202 | 1 / 24 | 720 |
| Random protection 303 | 0 / 24 | 556 |

A saved routing-graph analysis showed that among the **278 ordinary retained targets**, the **lowest-E quartile received 4,569/6,912 = 66.10%** of incoming sources, and the **highest-E quartile only 126/6,912 = 1.82%**. The **lowest-E 28** targets received **3,092** incoming sources; the **highest-E 28** received only **21**.

The null result for high-E protection is thus largely an **intervention-activity limitation**: few sources were being routed into those high-E targets in the first place. It does not establish that protecting important representations cannot help, or that low-E targets are unimportant.

**Source:** [routing/protection full30 review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-routing-full-review.md) · [routing protocol](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/protocols/rq2_routing_input_v1.json).

## 6. Local controlled mechanism illustration: restore one Brolio state

This **post-hoc, gold-informed, single-question oracle** starts from one frozen native merged input tuple. It changes **only target 317's projected representation** back to its pre-merge value. All other projected token states, selected slots, masks, native basis insertion, generation policy and scoring remain unchanged. A sham replay and count-matched other recipient **141** provide controls.

| Input condition | Generated answer | Correct? | First gold tokenizer-piece probability |
|---|---|---|---:|
| Uncompressed full576 | **Brolio** | Yes | 0.811604 |
| Native E merging | Bottoli | No | 0.058736 |
| Same selection, all ordinary merges disabled | **Brolio** | Yes | 0.709970 |
| Sham / unchanged merged tuple | Bottoli | No | 0.058736 |
| Restore **only target 317** | **Brolio** | Yes | 0.556764 |
| Restore count-matched **control 141** | Bottoli | No | 0.058997 |

Restoration of target 317 alone is **sufficient** to recover the original answer in this frozen condition. This is stronger local evidence that **an altered retained representation**, not just outright token omission, can cause an answer-preservation failure. The result does **not** show target 317 is uniquely necessary, identify exactly which letters it represents, or validate a deployable gold-blind repair policy. It has one selected case and one matched control; the restoration arms were not independently repeated as fresh full runs.

**Source:** [Brolio oracle review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-brolio-oracle-review.md).

---

## 7. Calibration between input-only and complete two-stage studies

### 7.1 Input-only 576→64 sanity experiment (October 20)

A six-question/five-image calibration verified that native ApET really exposes only **64** visual states to the decoder: **ten basis slots plus 54 ordinary retained targets**; **512** dropped sources are routed/merged in the native merge arm. Native input-only64 was **2/6**, and pure-random input-only64 was **4/6, 3/6, 2/6** for seeds 101/202/303. All were selected cases; the comparison did not establish population-level random superiority.

Important individual outputs: **Brolio → Chianti** under native input64; **Dominion → Will** (from a scene containing a prominent *WE WILL ROCK YOU* poster); **Cookery school → Cooking school**. No-merge also failed all three native64 text cases. A no-merge rescue observed at **288 input states** cannot be assumed to survive **64 input states**.

**Source:** [pure-random/input64 calibration review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-20-rq2-pure-random-calibration-review.md).

### 7.2 Actual two-stage (96→32 / 288→96) calibration (October 21)

On **six selected questions** (five images), the complete native procedure scored **1/6** at 96→32 and **4/6** at 288→96; pure random scored **2/6, 2/6, 3/6** at 96→32 and **5/6, 5/6, 4/6** at 288→96. These results only validated execution and described selected cases. The calibrated second stage truly compressed **after decoder block 16**, and cache/replay controls passed. These questions were deliberately chosen for testing and include known failures; they are not a broad ranking of either method.

**Source:** [staged calibration review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-calibration-review.md).

---

## 8. Broad RQ2 comparison: complete native ApET versus pure random pruning

### Experiment contract and cohorts

Same **LLaVA-1.5-7B**, Kaggle **2 × NVIDIA T4**, FP16, Python **3.10.18**, PyTorch **2.1.2+cu121**, Transformers **4.37.2** and pinned source/model checks. At each of the two schedules, **pure random uses the same exact visual state counts** as native ApET but does **not** perform FPS, reconstruction, basis insertion or merging. Seeds **101, 202 and 303** are fixed in advance. Each dataset uses its own frozen question sets, answer scoring, and accuracy definition.

| Dataset | Cohort | Questions / images | Role and primary score |
|---|---|---:|---|
| **TextVQA100** | 100 development/screening questions | **100 / 100** | Mean **ten-annotator partial-credit consensus**; **full-credit binary** only for losses/gains |
| **GQA800** | 800 development questions | **800 / 100** (eight questions per image) | Frozen short-answer accuracy |
| **MMStar1500** | Whole pinned 1,500-question MMStar set | **1,500 / 1,430 unique image hashes** | Project's frozen explicit-choice scorer; not a claim of official leaderboard reproduction |
| **Total** | Three different cohorts | **2,400 questions** | **9 arms per question = 21,600 primary predictions**; don't pool the three metrics into one accuracy |

The broad runs are **not identical to the original paper's evaluation subsets, prompts, or scoring** and are not a numerical reproduction claim. They test **within-project matched comparisons**. Controls and repeat runs are outside the primary denominators. The reserved GQA evaluation split remained unused.

### 8.1 Summary of all six native-vs-random comparisons

**Scores are percentages.** Each “random” number is the **mean ± sample standard deviation across three prespecified seed scores**. The paired 95% confidence interval is for `native − mean(random across seeds)` using **image-group resampling**, not a confidence interval estimated from just three seed values.

| Dataset | Schedule (avg. tokens) | Full576 | Native ApET | Pure random mean ± SD | Native − random, pp [paired 95% CI] |
|---|---|---:|---:|---:|---|
| TextVQA100 | 96→32 (64) | 49.70 | 41.50 | 20.63 ± 3.12 | **+20.87 [13.93, 29.03]** |
| TextVQA100 | 288→96 (192) | 49.70 | 44.20 | 34.77 ± 4.80 | **+9.43 [3.20, 15.43]** |
| GQA800 | 96→32 (64) | 69.00 | 63.75 | 61.29 ± 0.79 | **+2.46 [0.25, 4.71]** |
| GQA800 | 288→96 (192) | 69.00 | 68.00 | 66.83 ± 0.62 | +1.17 [−0.58, 2.83] |
| MMStar1500 | 96→32 (64) | 34.73 | 33.40 | 33.53 ± 0.53 | −0.13 [−1.35, 1.07] |
| MMStar1500 | 288→96 (192) | 34.73 | 35.00 | 35.20 ± 0.29 | −0.20 [−1.25, 0.85] |

**Interpretation:** On the evaluated TextVQA100 cohort, **whole ApET beats pure random** by a substantial margin under both schedules. On GQA800 the effect is smaller; at 288→96 its interval includes zero. On MMStar1500 neither comparison establishes a detectable native advantage; zero-containing intervals **do not prove equivalence or random superiority**. Cross-dataset differences could reflect multiple model/task/protocol factors, not a demonstrated “OCR-specific mechanism.”

### 8.2 Per-seed scores: not only the random mean

| Dataset / score | Schedule | Random seed 101 | Seed 202 | Seed 303 | Native primary |
|---|---|---:|---:|---:|---:|
| TextVQA consensus | 96→32 | 17.10 | 23.00 | 21.80 | **41.50** |
| TextVQA consensus | 288→96 | 35.60 | 39.10 | 29.60 | **44.20** |
| GQA accuracy | 96→32 | 60.38 | 61.75 | 61.75 | **63.75** |
| GQA accuracy | 288→96 | 66.13 | 67.13 | 67.25 | **68.00** |
| MMStar accuracy | 96→32 | 33.53 | 34.07 | 33.00 | **33.40** |
| MMStar accuracy | 288→96 | 35.07 | 35.53 | 35.00 | **35.00** |

Selecting whichever random seed wins an individual question *after seeing the answer* is an oracle and **not** the evaluated random algorithm. Keep these three fixed seeds rather than reporting the best seed per question.

**Sources:** [TextVQA100 staged review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-textvqa100-review.md) · [GQA800 staged review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-gqa800-review.md) · [MMStar1500 staged review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-mmstar1500-review.md) · [compact score export](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/materials/rq2_staged_results.csv) · [seed/SD audit](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/materials/rq2_random_seed_summary.json).

---

## 9. Preservation, losses and gains: the accuracy number hides answer turnover

This table compares **native ApET to the uncompressed baseline** for every broad dataset and schedule, not native to random. TextVQA “correct” here means **full ten-annotator consensus credit**; its primary partial-credit score is in Section 8.

| Dataset | Schedule | Baseline correct | Native preserved | **Losses** | **Gains** | Native correct | APR |
|---|---|---:|---:|---:|---:|---:|---:|
| TextVQA100 | 96→32 | 44 | 34 | **10** | **3** | 37 | 77.27% |
| TextVQA100 | 288→96 | 44 | 37 | **7** | **3** | 40 | 84.09% |
| GQA800 | 96→32 | 552 | 474 | **78** | **36** | 510 | 85.87% |
| GQA800 | 288→96 | 552 | 518 | **34** | **26** | 544 | 93.84% |
| MMStar1500 | 96→32 | 521 | 442 | **79** | **59** | 501 | 84.84% |
| MMStar1500 | 288→96 | 521 | 485 | **36** | **40** | 525 | 93.09% |

For **GQA800 moderate**, accuracy changes only from **69.0% to 68.0%** (−8 net correct), but **34 + 26 = 60 question outcomes switch correctness**. For **MMStar1500 moderate**, accuracy increases from **34.73% to 35.00%** (+4 net correct), yet **36 + 40 = 76 outcomes switch correctness**. For aggressive GQA, **78 + 36 = 114** switch; aggressive MMStar, **79 + 59 = 138**. This is **real, recorded score turnover under the frozen scorer**; it is *not* proof that new visual reasoning skills were gained.

The moderate MMStar native arm preserves **485** of 521 baseline-correct questions, slightly more than random mean **482**, but pure random mean acquires **46** gains versus native **40**. Aggregate accuracy obscures this difference in preservation versus gain behavior. Similar effects occur in other rows.

**Same-image GQA:** All 78 aggressive and all 34 moderate native losses had at least one other baseline-and-native-correct question about the **same image**. On the shared image encoding, 56 of 100 GQA images at aggressive compression and 27 at moderate compression contained both lost and preserved baseline-correct questions. That supports question-conditioned outcome heterogeneity; it does not identify which feature or reasoning stage caused the loss.

**Sources:** [preservation-by-condition CSV (all 27 conditions)](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/preservation_by_condition.csv) · [six-row native/random export](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/table2_preservation.csv) · [supporting-data readme](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/README.md).

### 9.1 Broad TextVQA concrete cases

These are **staged two-boundary** cases and must **not** be merged with the **input-only288 Brolio → Bottoli** oracle outcomes above:

- **ID 35962 (Cookery school):** baseline **Cookery school**; native 96→32 **Cooking school**; native 288→96 **Culinary**. Both count as exact-text failures under the frozen scorer.
- **ID 36001 (Give up):** becomes **Gonna** or **Gone** under the two native schedules.
- **ID 37533 (Brolio):** becomes **Travel** (96→32) or **Castello di rivoli** (288→96); the 288→96 pure-random seeds 101 and 202 return **Brolio**, while seed 303 returns **Bodega**. This does **not** overturn the strong *average* native advantage across 100 questions.
- **ID 39308 (Orcid.org):** 96→32 still outputs the correct **Orcid.org**, but 288→96 becomes **Wwwidcom**. **More retained tokens do not guarantee monotonic correctness for every question**, because the realized selected sets/merge graphs change.
- **ID 39323 (Sky mall):** 96→32 outputs **Sky** (wrong); 288→96 preserves **Sky mall** while all three random 288→96 runs fail.
- Some native “gains” were mainly lexical/scoring changes: **Um → Umd**, **Monday night → Monday**, and at moderate compression **Steigl → Stiegl**. These score transitions are valid under the fixed consensus scorer, but not automatically evidence of new visual reasoning competence.
- Comparing the two native schedules, the higher-budget schedule **recovers seven** cases that the lower-budget one misses but **loses four** cases the lower-budget one gets right (examples in the TextVQA review). This is a property of two different compression realizations, **not** a controlled intervention that only adds back identical visual states.

**Source:** [Staged TextVQA review, “Failures, gains and nonmonotonic answers”](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-textvqa100-review.md).

### 9.2 Task/category observations (exploratory slices)

- **GQA:** The random mean is *higher* on structural **Compare** questions at both budgets (96→32: random **67.95%** vs native **62.82%**; 288→96: random **68.80%** vs native **67.95%**), despite native's overall advantage. The slices were descriptive, not separate significance-tested subgroup findings.
- **MMStar:** Every one of its six categories has **250 questions**. The native-vs-random pattern is mixed by category; some improve and some decline, and native's own uncompressed baseline is weak on several categories. No category-specific mechanism has been isolated.
- **MMStar baseline caveat:** A correct multiple-choice letter is not necessarily proof of visually grounded reasoning. Prior targeted option/image checks showed A-choice bias on some selected chart cases. Do not infer a specific visually grounded competence or its disappearance just from aggregate MMStar accuracy.

**Sources:** [GQA task patterns](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-gqa800-review.md) · [MMStar categories](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-mmstar1500-review.md).

---

## 10. Infrastructure, correctness verification and run-to-run variation

### 10.1 Why the original parity test failed

The first RQ2 runner demanded **byte-identical features from separate native compressed executions**. On GQA cake/plate case `201636988`, CLIP inputs, E, FPS source choices, selected slots and merge assignment graph agreed exactly, but the resulting post-merge/projected values did not. The initial pilot stopped on its gate and admitted **zero completed questions**. Treating those first three observed answers as an accepted merge experiment would be incorrect.

The follow-up one-question diagnostic found:

| Repeat comparison | Max absolute feature difference | Changed retained states |
|---|---:|---:|
| Repeated **raw unwrapped native projected** states | **0.0078125** | **28 / 288** |
| Repeated observed native **before merge** | **0** | **0 / 288** |
| Repeated observed native **after merge** | **0.015625** (largest pair) | **30 / 288** |
| Repeated **no-merge** at all captured stages | **0** | **0 / 288** |

This strongly localizes the observed variation to arithmetic **during native merging**, despite identical inputs/assignments. CUDA FP16 accumulation using `scatter_add_` is a **plausible but unproven root cause**; there was no controlled deterministic-kernel swap that isolated this operation as uniquely responsible. The diagnostic recorded only one free generated answer, so its score variability should not be described as newly observed answer flips in that diagnostic.

### 10.2 Correction: versioned capture/replay, not relaxed tolerances

The accepted v2 implementation captured the **first native realization per image/condition**, then reused its frozen representation for same-image questions and scoring, while preserving exact replay checks. Independent fresh-native repeats were reported separately. This removed an invalid exact-cross-execution-equality assumption **without hiding genuine numeric variation**, changing a tolerance post hoc or modifying native ApET's source.

The staged broad experiments additionally verified selected counts/masks, basis identities, merge assignments, cache geometry, full-576 no-op controls and fixed-feature replays. Each broad run included **six full-576 identity checks and 18 fixed-feature replay controls**, which matched generated IDs/answers/scores exactly. All primary answers were rescored; raw archive hash verifications are documented in the corresponding reviews.

### 10.3 The limitation remains real

In the staged GQA800 experiment, the predeclared fresh-native case `201636988` at **288→96** had a primary correct **No**, one fresh repeat **No**, and another **Yes** (incorrect), while exact fixed-feature replays were stable. Historical independently captured native results also exhibited a few raw-answer differences at unchanged nominal settings. These observations warn against overinterpreting *small differences or individual answers* from a single fresh native realization. They do **not** establish a dataset-wide answer-flip frequency or invalidate the properly archived first-run outcomes.

**Sources:** [failed first gate](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-first-parity-failure.md) · [parity localization](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-native-parity-diagnostic-review.md) · [accepted v2 result](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-v2-full-review.md) · [staged GQA repeats](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-gqa800-review.md).

---

## 11. What can we conclude scientifically? What is still unproven?

### Supported conclusions

1. **The complete ApET pipeline has task-dependent advantages over pure random pruning** in our controlled development/benchmark cohorts: large on TextVQA100, smaller on GQA800, no clear native advantage on MMStar1500. The effect is strongest at the more aggressive 96→32 schedule for TextVQA.
2. **Whole-procedure success does not imply that the E ranking alone is the reason.** The pure-random comparison simultaneously changes E selection, FPS/basis insertion and similarity merging. This does not contradict the RQ1 result that E poorly/inconsistently ranks some singleton answer-response measurements.
3. **Input-stage merging can damage a previously correct answer even when the crucial retained slot survives.** Brolio is the strongest bounded illustration, supported by merge/no-merge, frozen-tuple target317 restoration and negative controls.
4. **Selected sets and merging interact.** Enabling merging rescued some questions and harmed others under different selected sets in the 30-question crossed experiment.
5. **Average accuracy hides extensive correctness turnover.** All six native broad conditions contain both losses and gains. Gains should not be automatically interpreted as improved perception or reasoning.
6. **A simple highest-E-target protection policy was barely exercised by the actual merge graph**, so its near-null result is not an effective general test of protective aggregation.
7. **Native compressed inference is not universally bitwise stable across fresh executions**, although fixed-feature replay can be exact. The precise arithmetic root cause is not established.

### Not established by completed work

- A universal percentage attribution of ApET gains to **E ranking versus basis handling versus merging**.
- Whether **linear reconstructability** mathematically predicts downstream answer preservation after **nonlinear model computation**.
- A generally superior no-merge, random-selected, or randomized-routing policy on unbiased unseen data.
- Which input or decoder stage (or their interaction) dominates errors in the **two-stage** setup. Input-stage changes alter later hidden states and subsequent E/selection.
- A rigorously validated per-token intrinsic “importance” ranking, a general OCR-specific causal explanation, or a proven letter-level “blur” mechanism.
- A training-free RQ3 compression improvement, held-out success at matched average token budget, or a measured controlled **latency/speedup** improvement. Observed notebook runtimes include loading, tracing and evidence export and are **not** performance benchmarks.
- Exact reproduction of the paper's overall published TextVQA/GQA/MMStar scores, or population claims beyond the specified cohorts.

### Most useful follow-up (proposed, **not yet run**)

A focused matched-budget **staged merge intervention**, or a controlled comparison of native merged vs retained/pre-merge features at a fixed stage, would address the largest remaining RQ2 question. One candidate is **input merging on/off × decoder-stage merging on/off**, carefully specifying whether later E/selection is held fixed or recomputed. Because changing input merging changes decoder states and later selections, this would estimate **stage-conditional and interaction effects**, not a simple additive stage decomposition.

A second diagnostic direction is to compare native averaging with **no merge or controlled merge-strength interpolation** at the same selected positions, and measure downstream answer response. A more direct low-E vs reconstructed-state test would be a **diagnostic intervention** unless token count and conditions are carefully matched; it is **not an already evaluated improvement**. Both ideas are current plans, not results.

**Source:** [stage-one closeout and future ideas](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-stage1-closeout-and-future-ideas.md) · [midterm evidence checklist](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/evidence_checklist.md).

---

## 12. Evidence map / reproduce or extend this audit

All URLs below refer to tracked GitHub files. The raw ZIP/log archives mentioned inside these notes are in the developer's separate **Git-ignored `research/results/`** storage; a GitHub checkout by itself does **not** include their contents. Some note links to `results/` directories consequently resolve only in the original local project environment.

| Evidence block | Tracked review or export | Original archived experiment (ZIP basename) |
|---|---|---|
| Merge/no-merge selected 30 | [v2 full review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-v2-full-review.md) | `rq2-input-merge-v2-full-20261019T224044177722Z.zip` |
| Crossed E/random × merge/no-merge | [crossed full30 review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-crossed-full-review.md) | `rq2-crossed-full-v1-20261019T234435011493Z.zip` |
| Single-state Brolio restoration | [oracle review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-brolio-oracle-review.md) | `rq2-crossed-oracle-v1-20261020T000115185532Z.zip` |
| Routing and protected recipients | [routing review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-routing-full-review.md) | `rq2-routing-full-v1-20261020T004043438261Z.zip` |
| Input-only64 pure random calibration | [pure random calibration](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-20-rq2-pure-random-calibration-review.md) | `rq2-pure-random-calibration-v1-20261021T001951260192Z.zip` |
| Full two-stage six-case calibration | [staged calibration](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-calibration-review.md) | `rq2-staged-calibration-v1-20261021T130721550505Z.zip` |
| TextVQA100 broad | [TextVQA review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-textvqa100-review.md) | `rq2-staged-textvqa100-v1-20261021T134751714241Z.zip` |
| GQA800 broad | [GQA review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-gqa800-review.md) | `rq2-staged-gqa800-v1-20261021T142303483991Z.zip` |
| MMStar1500 broad | [MMStar review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-rq2-staged-mmstar1500-review.md) | `rq2-staged-mmstar1500-v1-20261021T153730115642Z.zip` |
| Six-row main-score summary | [CSV](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/materials/rq2_staged_results.csv) | Derived from the three broad archives; no extra inference |
| Per-condition losses/gains | [CSV](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/preservation_by_condition.csv) | Derived from broad saved primary predictions |
| Cross-dataset native/random preservation stats | [CSV](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/reports/midterm/supporting_data_2026-10-25/table2_preservation.csv) | Derived from broad saved primary predictions |
| Native numerical/parity investigation | [diagnostic review](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-19-rq2-native-parity-diagnostic-review.md) | `rq2-native-parity-diagnostic-v1-20261019T193300319406Z.zip` |
| Research status and possible next protocols | [closeout/future ideas](https://github.com/doguineyes/CSE590-ApET/blob/apet-analysis/research/notes/2026-10-21-stage1-closeout-and-future-ideas.md) | Planning document; **no claimed new experiment** |

**Suggested one-paragraph summary for a future report or presentation:**

> We investigated ApET's selection and merging mechanisms using controlled input-stage diagnostics and larger two-stage evaluation cohorts. On 30 selected questions, disabling ordinary merging rescued two known failures, while crossed E/random-retention conditions showed that merging can help or harm depending on the selected set. A targeted Brolio experiment recovered the correct name by restoring one retained representation altered by merging. We then compared complete ApET with pure random pruning on 2,400 questions. ApET showed a large advantage on TextVQA, a smaller advantage on GQA and no clear advantage on MMStar. However, accuracy hid many transitions between correct and incorrect answers. These experiments establish task-dependent whole-procedure performance and conditional component effects, but do not yet isolate how much benefit comes from reconstruction-error ranking, basis handling, or merging across the full model.

**Audit boundary:** This is a synthesis of **existing experiments**, not a newly run evaluation, a complete causal explanation, or a claim of reproducing the original ApET paper.
