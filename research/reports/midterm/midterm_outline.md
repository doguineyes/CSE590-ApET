# Midterm report outline — conservative progress framing

**Updated October 22, 2026, following the user's scope clarification.**

This is the plan for revising the [October 21 draft](midterm_draft.md), not a
submission-ready report. It supersedes the earlier closeout language as the
report's framing. The user wants an account of progress and unresolved problems,
without suggesting that the whole project is nearly finished or that a promising
new compression algorithm has already been identified. Typesetting and page
counting are deferred until the narrative is agreed.

## 1. Position the report as a progress update

Keep the submitted title:

**Understanding Reconstruction-Based Visual Token Compression: When and Why Does
ApET Work?**

Add the subtitle **Midterm Progress Report**. No new algorithm name or stronger
claim is needed.

Suggested framing:

> We have established a reproducible experimental workflow and largely completed
> the planned bounded RQ1 analysis. For RQ2, several controlled diagnostics and
> larger comparisons have been conducted, but their mechanism-level interpretation
> remains incomplete. A concrete, evidence-supported direction for RQ3 has not
> yet been selected.

| Question | Current status | Report boundary |
|---|---|---|
| RQ1: reconstruction error versus answer contribution, across tasks and depths | Planned diagnostic experiments largely complete; interpretation is qualified by the measurement definitions and limited cohort/depth coverage | Report intervention-specific response patterns, not intrinsic token importance or a universal depth law |
| RQ2: selection versus merging | Substantial experiments completed; analysis and mechanism attribution ongoing | Report observed component effects and complete-procedure comparisons separately; no fixed percentage attribution |
| RQ3: lightweight improvement | Still exploratory; no concrete implementation direction selected | Describe the decision still to be made, rather than advertise an algorithm or promise a gain |

Being conservative means limiting explanations to what was tested. It does not
require hiding the clear numerical observations or the work already completed.

## 2. Suitable report structure

### Abstract — brief progress summary

Write this after the body. State the problem, model and datasets; summarize the
preliminary RQ1/RQ2 observations; explicitly leave RQ2 interpretation and RQ3 open.
Avoid “we explain why ApET works” or “the project is complete.”

### 1. Motivation and original research questions

- Introduce the cost of visual tokens and ApET's reconstruction-error selection
  and similarity merging.
- Explain the research motivation: a feature can be reconstructable without its
  answer-level role being obvious; selection and aggregation may affect behavior
  differently.
- Preserve all three original RQs, with the status distinction above.
- Present analysis of the existing method as the present objective. A successful
  improvement has not been demonstrated or selected.

### 2. Experimental setup and study design

Keep this compact and include what a reader needs to interpret the results:

- Native ApET/LLaVA-1.5-7B on Kaggle T4 ×2, full576 reference, and the two-stage
  96→32 / 288→96 schedules. A locked environment resolves the changing host
  dependency problem; exact versions and source records can stay in the evidence
  materials.
- Distinguish the **RQ1 diagnostic panel** (GQA12, MMStar12, TextVQA6) from the
  **larger RQ2 cohorts** (GQA800 development, MMStar1500, TextVQA100 screening).
- Define error E, singleton zeroing and the source-informed 1×1 IH retention
  adaptation. Primary matched scores use the first frozen reference tokenizer
  piece, not full-answer correctness. Depth coverage is input and after block16
  on six matched questions.
- Separate basis-fixed ordinary-position randomization from staged pure-random
  pruning. Define dataset-specific scoring, APR and clustered paired uncertainty.
- Explain that reviewed diagnostic examples are not representative benchmark
  samples. Raw predictions and archives are kept locally outside Git; compact
  tables, protocols and review notes are tracked.

### 3. RQ1: completed diagnostic experiments and preliminary observations

Use the existing three-row correlation table as the quantitative anchor.

The observations we can report are:

1. E does not consistently rank positive reference-answer support under either
   measured counterfactual on the curated panel. Correlations vary across
   questions, operators, filtering and scores; do not replace this with “E is
   unrelated to task relevance.”
2. Same-image questions can have different answer-response maps with the same
   input E. The question and reference target both change, so this is not pure
   isolation of question wording.
3. Zeroing and retention produce different rankings and answer different
   counterfactual questions. This is a measurement limitation and finding, not
   proof that either operational measure is invalid or ground truth.
4. Selected input/layer16 comparisons reveal sparse strong states but no
   consistent improvement in E–response agreement. Contextual states and late
   visual removal require careful interpretation.

End with a bounded status sentence: **the planned diagnostic measurements are
largely complete; general task relevance and the causes of compression failures
are not fully established.** One same-image or exact-text illustration is enough;
the report need not inventory all galleries and calibrations.

### 4. RQ2: experiments conducted, with attribution still open

Organize this section by experimental question, not notebook chronology.

**A. Controlled input-only component diagnostics.** Summarize the selected
30-case, 288-token panel. The basis-fixed crossed experiment should appear,
because it directly bears on selection and merging:

| Retention policy | With merging | Without merging |
|---|---:|---:|
| Native E selection | 19/30 | 21/30 |
| Random ordinary positions, seed101 | 25/30 | 24/30 |
| Random ordinary positions, seed202 | 23/30 | 27/30 |
| Random ordinary positions, seed303 | 26/30 | 26/30 |

The native ten-basis machinery and placement are fixed here. Merging's effects
depend on the retained set and question. This panel was failure-enriched, so it
does not establish random superiority on the task populations. Routing and
protection can be summarized briefly; the high-E protection test was largely
inactive, so its lack of improvement is inconclusive.

**B. Larger staged comparisons.** Keep the existing six-row results table for
full576, native and the three-seed pure-random mean at both schedules. Say:

- Native has a substantial advantage over random on the evaluated TextVQA100
  cohort, while still losing score relative to full576.
- GQA differences are smaller, and the moderate-schedule interval spans zero.
- MMStar has no clear aggregate native/random difference under these conditions.
  This is not an equivalence result.

These compare whole procedures: selection, FPS/basis representation and merging
all differ. They cannot isolate E's independent contribution. The contrast with
the selected input-only panel motivates further analysis; it is not a numerical
contradiction needing a rerun.

**C. One case study and open interpretation.** Brolio's fixed-context restoration
is useful evidence of a conditional merge-related failure. Keep its sham/control
and post-hoc scope visible. Similar total accuracy can also conceal losses and
gains. Use the MMStar choice-bias/stable-answer observations only as descriptive
context, not proof that the model ignores images or as a complete explanation
of the native/random gap.

Conclude: **many RQ2 conditions have been tested, but we do not yet have a general
account of how selection, basis preservation, merging and task demands interact.**

### 5. Roadblocks and limits on interpretation

Prioritize substantive issues over implementation history:

- Reproducible environments and native-versus-project source separation were
  necessary before experiments could be trusted.
- Frozen encoded replays are stable, but some fresh compressed runs change
  answers. Current intervals do not include all whole-run numerical variation.
- Answer-response scores depend on intervention, null context and tokenizer
  scoring. They are not universal importance labels.
- GQA wording/exact-match issues and MMStar baseline competence/choice bias
  complicate failure analysis. Screening and curated subsets limit generality.
- The broad tests do not separate all components; two depths do not define a
  full layer trend; recorded run costs are not controlled speedup benchmarks.

A short conceptual sentence is reasonable: reconstruction-based selection and
cosine averaging use different criteria, and neither guarantees preservation
through a nonlinear decoder. This motivates an open question; it is not a new
theoretical result or evidence for a particular replacement algorithm.

### 6. Current status and remaining research

Keep this short and explicitly uncertain:

- RQ1's present diagnostic scope is largely complete; consolidate its findings
  and limitations.
- For RQ2, interpret the existing comparisons and decide which targeted
  experiment would best resolve a specific ambiguity. Stage-wise merge controls
  and exact-text failure analysis are candidates, not an agreed full program.
- For RQ3, no implementable direction has been chosen. The next research decision
  is whether the analysis supports a feasible modification at all. Adaptive
  budgets, alternate bases and merge changes remain possibilities rather than
  commitments.

The closing sentence should say what has been learned so far and what remains
unresolved, without announcing project completion. Keep the six relevant paper
references. Defer layout, precise page allocation and presentation slides.

## 3. Do we need to rerun anything before writing?

**No mandatory new GPU run was identified for this outline's preliminary claims.**
The saved RQ1 table reductions reproduce; all five numerical source hashes match
the prior report snapshot. Existing component/oracle and three staged-run audits
record passed/verified status. This review checked saved data and prior audits;
it did not repeat tensor computations or re-audit every raw archive.

The feedback's extra MMStar statistics were independently recomputed from saved
primary rows and the frozen reference:

- Baseline predicts A on **1,059/1,500** questions; the gold label is A on
  **429/1,500**.
- Across full576, native288→96 and its three matched random seeds, **1,248**
  questions have the same parsed option in all five conditions; **798** of those
  are wrong in all five.

Agreement here means the frozen scorer's parsed option, not necessarily identical
raw strings. The counts establish stability and option imbalance; causes remain
unresolved. The compact [saved-data review](materials/2026-10-22_saved_data_review.json)
preserves sources, hashes and definitions.

Additional fresh repeats would be relevant if we wanted stronger numerical
reproducibility claims, and new controlled ablations would be needed for stronger
mechanism attribution. Neither is a prerequisite to an honestly scoped midterm.
Do not rerun the full cohorts just to make preliminary findings look final.

## 4. Assessment of the supplied feedback

Accept the suggestions to include the basis-fixed selection result, distinguish
model competence from compression sensitivity, and clarify what evidence is
available outside GitHub. The independently checked MMStar counts are useful
optional context. Keep any nonlinear-aggregation discussion explicitly tentative.

Defer its PDF/page-check instruction at the user's request. Do not adopt its
pressure to name a concrete extension now: the user has not selected one, and
the present evidence does not establish a clearly effective implementation.
The next revision should make uncertainties and research decisions visible,
rather than make the existing experiments sound like a finished explanation.
