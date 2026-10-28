# ApET Can Both Lose and Gain Correct Answers: Evidence from Our Experiments

**Project:** CSE590 — Understanding Reconstruction-Based Visual Token Compression: When and Why Does ApET Work?  
**Evidence checked:** 2026-10-28  
**Repository:** [`doguineyes/CSE590-ApET`](https://github.com/doguineyes/CSE590-ApET), branch `apet-analysis`, inspected at commit [`f65ec142`](https://github.com/doguineyes/CSE590-ApET/tree/f65ec1422a8421864530e8e3bfa38206b2b07083).  
**Data provenance:** GitHub-tracked review notes and exported summaries from experiments performed October 19–21, 2026. No new inference was performed for this note. The large original result ZIPs are stored outside Git and were **not independently re-audited here**; the reviewed notes document their earlier audits.

## 1. The main observation

**Overall accuracy hides substantial turnover in which questions the model answers correctly.** Relative to full-576-token LLaVA-1.5-7B, native two-stage ApET changes answers in *both* directions:

- **Loss:** uncompressed model correct → compressed model wrong.
- **Gain:** uncompressed model wrong → compressed model correct.
- **Preserved:** uncompressed correct → compressed correct.
- **Both wrong:** uncompressed wrong → compressed wrong.

For an exact-match/choice task:

\[
\text{Compressed correct}=\text{Baseline correct}-\text{Losses}+\text{Gains}.
\]

For TextVQA, the transition counts use **full ten-annotator consensus credit** (a binary criterion), while the headline 49.7%, 41.5% and 44.2% scores use **partial-credit consensus averages**. These two measures must not be mixed.

### All six native ApET comparisons against the uncompressed model

The schedule `96→32` means retaining 96 image tokens after the vision encoder and 32 after decoder block 16 (average **64** across the 32 decoder layers); `288→96` averages **192**. These are **native two-stage ApET** experiments, not input-only interventions.

| Cohort | Schedule (mean tokens) | Baseline score | Native ApET score | Baseline-correct | Lost | Gained | Correctness flipped (lost + gained) | Net correct-count change |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TextVQA100¹ | 96→32 (64) | 49.70% | 41.50% | 44 | **10** | **3** | 13/100 | −7 full-credit |
| TextVQA100¹ | 288→96 (192) | 49.70% | 44.20% | 44 | **7** | **3** | 10/100 | −4 full-credit |
| GQA800 | 96→32 (64) | 69.00% | 63.75% | 552 | **78** | **36** | 114/800 | −42 |
| GQA800 | 288→96 (192) | 69.00% | 68.00% | 552 | **34** | **26** | 60/800 | −8 |
| MMStar1500 | 96→32 (64) | 34.73% | 33.40% | 521 | **79** | **59** | 138/1500 | −20 |
| MMStar1500 | 288→96 (192) | 34.73% | **35.00%** | 521 | **36** | **40** | 76/1500 | **+4** |

¹ **TextVQA caution:** The score columns are mean partial-credit consensus; all counts and `net correct-count change` use binary full-credit correctness. Thus 44→37 full-credit answers is *not* calculated by subtracting 41.5% from 49.7%.  
**All counts are per-question transitions from the first recorded native primary realization.** The table is descriptive for the specified cohorts, not a population-wide claim.

**Direct source data:** [All-condition transition CSV](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/reports/midterm/supporting_data_2026-10-25/preservation_by_condition.csv); [six-row preservation summary](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/reports/midterm/supporting_data_2026-10-25/table2_preservation.csv); [supporting evidence README, §2](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/reports/midterm/supporting_data_2026-10-25/README.md).

## 2. The strongest illustrations of hidden turnover

### A. GQA800, moderate compression: almost unchanged accuracy, 60 flipped outcomes

- Baseline: **552/800 = 69.0%** correct.
- Native ApET 288→96: **544/800 = 68.0%** correct.
- **34 losses + 26 gains = 60 questions whose correctness changed (7.5% of all questions).**
- The net accuracy difference is only **−1.0 percentage point** because losses and gains offset.

This is a clean example of why a summary accuracy number alone cannot describe answer preservation. It does *not* establish that the model's visual reasoning improves on those 26 gains.

[Primary GQA800 review — “Accuracy and preservation”](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/notes/2026-10-21-rq2-staged-gqa800-review.md).

### B. MMStar1500, moderate compression: accuracy improves while correct answers are lost

- Baseline: **521/1,500 = 34.73%** correct.
- Native ApET 288→96: **525/1,500 = 35.00%** correct.
- **36 losses + 40 gains = 76 changed correctness outcomes**, but **net +4** correct answers.
- Therefore, **higher aggregate accuracy does not imply that the compressed model preserved the original model's knowledge or evidence on every question**.
- The native-versus-baseline 95% interval reported in the review is **−0.93 to +1.39 percentage points**: the +0.27-point aggregate increase is not compelling evidence of a general accuracy improvement.

[Primary MMStar1500 review — “Accuracy, preservation and uncertainty”](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/notes/2026-10-21-rq2-staged-mmstar1500-review.md).

### C. GQA800, aggressive compression: numerous gains coexist with an overall decline

- Baseline: **552/800** correct; native 96→32: **510/800**.
- **78 losses + 36 gains = 114 changed correctness outcomes (14.25%)**.
- Native ApET is **2.46 accuracy points above pure-random pruning** on this cohort at the same average token budget (paired image-bootstrap 95% interval **+0.25 to +4.71**), **yet still loses 78 answers the uncompressed model got right**.

This is the clearest demonstration of the *two different comparisons*: being better than random pruning does not mean being lossless relative to the original model.

[Primary GQA800 review](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/notes/2026-10-21-rq2-staged-gqa800-review.md).

## 3. Concrete TextVQA examples

The TextVQA100 review records **actual changed generated answers**, in addition to the counts. These cases should be labeled as examples, not attributed to one causal token without additional controls.

**Previously correct → incorrect (native staged ApET):**

| Question ID | Recorded answer change | Schedule / qualification |
|---|---|---|
| `35962` | **“Cookery school” → “Cooking school”** or **“Culinary”** | 96→32 / 288→96 respectively |
| `36001` | **“Give up” → “Gonna”** or **“Gone”** | Native staged runs; schedule-specific outputs in review |
| `37533` | **“Brolio” → “Travel”** or **“Castello di rivoli”** | Native staged runs; **do not confuse** with the input-only “Bottoli” case below |
| `39308` | **“Orcid.org” → “Wwwidcom”** | At 288→96, while the 96→32 run retains “Orcid.org” |
| `39323` | **“Sky mall” → “Sky”** | At 96→32; 288→96 retains “Sky mall” |

**Previously wrong → full-credit correct (scored gains):**

- **“Um” → “Umd”**.
- **“Monday night” → “Monday”**.
- At 288→96, **“Steigl” → “Stiegl”**.

These gain examples are reported explicitly in the TextVQA review as *scorer-defined full-credit gains*, but that note does not list their question IDs. They illustrate why gains need careful qualitative inspection: some may be spelling or answer-format changes, not improved visual understanding.

**Non-monotonic budget effect:** comparing the two native staged schedules in this same TextVQA cohort, the larger-token schedule recovers **seven** full-credit answers but loses **four** that the smaller-token schedule got right. Thus *more retained tokens* did not guarantee preservation of each individual correct answer. The two schedules also change selected sets, basis placement and merging; this is **not** a controlled experiment that only adds tokens to a fixed representation.

[Primary TextVQA100 review — “Failures, gains and nonmonotonic answers”](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/notes/2026-10-21-rq2-staged-textvqa100-review.md).

## 4. A narrower mechanistic example: Brolio → Bottoli (input-only experiment)

A **separate, selected input-only diagnostic** provides evidence that merging itself can cause an individual answer failure *under fixed selection and retained positions*. This is not part of the two-stage table above.

| Condition (input-only 576→288 tokens) | Answer | Correct? |
|---|---|---|
| Full 576-token baseline | **Brolio** | Yes |
| Native E selection **with merging** | **Bottoli** | No |
| Same selection and basis, **without merging** | **Brolio** | Yes |
| Restore the pre-merge representation of only target `317` | **Brolio** | Yes |
| Unchanged sham / restore count-matched control target `141` | **Bottoli** | No |

The native merge routed three non-retained sources (`103`, `109`, `331`) into retained target `317`. Restoring the **single chosen target's pre-merge feature**, while keeping all other projected positions and masks fixed, recovered the answer. This is stronger *local evidence* of a merge-dependent error than a raw before/after result.

**Important limits:** The target was selected after inspecting the failure, the case is a single cherry-picked diagnostic, and no general protection strategy was validated. It does **not** prove that a particular letter was stored at one token position or that merging generally hurts TextVQA.

[Single-target oracle restoration review](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/notes/2026-10-19-rq2-brolio-oracle-review.md); [selected crossed E/random × merge/no-merge review](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/notes/2026-10-19-rq2-crossed-full-review.md).

## 5. Evidence about *why* this happens: established vs open

**Established in the recorded experiments**

1. Native compression changes the identity of correct and incorrect answers in both directions in all six cohort/schedule conditions.
2. On MMStar 288→96, native aggregate correctness rises by four while 36 formerly correct answers are lost.
3. On GQA 288→96, accuracy drops by only eight net answers while 60 correctness outcomes flip.
4. Input-only component tests show that merging can help or harm depending on the selected set and question. In the selected 30-question panel, native E selection preserved **19/30 with merging** versus **21/30 without merging**; under fixed random-selected sets merging sometimes rescued questions and sometimes introduced failures.
5. A post-hoc one-target restoration recovers “Brolio” in a fixed input-only setting.

**Not yet established**

- That a gain reflects recovery of genuinely useful visual evidence rather than changed linguistic preference or benchmark scoring.
- That each failed answer was caused *specifically* by error-based ranking, cosine routing or averaging; the native two-stage procedure changes several coupled operations.
- That increasing token count monotonically increases correctness for individual questions.
- That the gain/loss rates or causal mechanisms generalize beyond these specific tested cohorts, frozen prompts and model.
- That compression improves overall MMStar accuracy in a statistically reliable way; the relevant baseline interval includes zero.
- That exact numerical outcomes are invariant across fresh compressed runs. The project recorded occasional fresh-run numerical variation; reported counts use the predeclared first native realizations.

### How this connects to our RQ2

ApET uses **linear reconstructability** to rank tokens, but performs **cosine-based averaging** to aggregate discarded representations. Neither operation directly guarantees that the compressed decoder preserves the same answers as full-576 inference. The gain/loss turnover motivates **controlled experiments on selection and merging**, particularly the difference between preserving geometric feature information and preserving downstream computations. The observed transitions alone do **not** identify the cause.

## 6. Concise sentences suitable for slides or the report

> **Short:** “Similar overall accuracy can hide many changed answers. On GQA800 at 288→96, ApET lost 34 previously correct answers but gained 26 previously incorrect ones, resulting in only a one-point accuracy drop.”

> **Stronger example:** “On MMStar1500 at 288→96, ApET's total accuracy rose slightly (34.73% to 35.00%), even though 36 baseline-correct answers became wrong and 40 baseline-wrong answers became correct. The small aggregate increase is uncertain.”

> **Research implication:** “These paired losses and gains show that visual-token compression changes model behavior in both directions. Accuracy alone cannot establish answer preservation, and component-controlled tests are needed to understand the mechanism.”

## 7. Primary repository references and evidence scope

- [Tracked results CSV: all 27 conditions, including native and random](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/reports/midterm/supporting_data_2026-10-25/preservation_by_condition.csv).
- [Tracked six-row preservation summary](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/reports/midterm/supporting_data_2026-10-25/table2_preservation.csv).
- [TextVQA staged primary review](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/notes/2026-10-21-rq2-staged-textvqa100-review.md).
- [GQA staged primary review](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/notes/2026-10-21-rq2-staged-gqa800-review.md).
- [MMStar staged primary review](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/notes/2026-10-21-rq2-staged-mmstar1500-review.md).
- [Brolio fixed-context oracle review](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/notes/2026-10-19-rq2-brolio-oracle-review.md).
- [Full30 selected component-ablation review](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/notes/2026-10-19-rq2-crossed-full-review.md).
- [Midterm evidence checklist explaining which raw artifacts are ignored by Git](https://github.com/doguineyes/CSE590-ApET/blob/f65ec1422a8421864530e8e3bfa38206b2b07083/research/reports/midterm/evidence_checklist.md).

**Source-access note:** The tracked CSVs and notes were read directly through the GitHub connection. Large underlying experiment archives were previously audited and summarized in those notes but are not committed to the GitHub tree; they were not fetched or re-executed for this evidence document. The Markdown file was prepared locally and **was not pushed to GitHub**.
