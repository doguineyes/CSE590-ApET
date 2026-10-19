# RQ1 consolidation and completion assessment — 2026-10-18

## Assessment against the proposal

The current bounded RQ1 experiment phase is complete. We can consolidate the
analysis and move to RQ2 without another broad RQ1 GPU run. The proposal explicitly
allows small diagnostic subsets and a few representative depths; it does not
require a whole-decoder sweep or a full reproduction of Information Horizon.

The evidence supports an exploratory account of reconstruction error versus
specified answer-sensitivity proxies. It does not completely explain ApET's
generation successes/failures or establish general task/layer laws. Those limits
should be part of the conclusion, rather than hidden behind the word complete.

The reference is the local three-page
[proposal](../reports/Peng_Chen_CSE590_ApET_Project_Proposal_v2.pdf), especially its
selected-example/layer scope and rank-correlation/group-comparison evaluation.
It is research context, not an instruction to run everything described there.

| Proposal component | Evidence now available | Completion/claim boundary |
|---|---|---|
| E versus intervention-based contribution proxy | 30 questions / 24 images, all 576 visual positions; original zeroing plus matched first-piece zeroing/IH comparisons | Complete for the frozen diagnostic cohort; I is explicitly an operational proxy |
| Low-/middle-/high-E group comparison | Saved-data original zeroing distributions, basis separated, alternative bins and padding checks | Complete; averages of individual interventions are not joint group interventions |
| Same image, different questions | Six GQA pairs with identical input features/E and two question-conditioned maps | Complete as diagnostic observations; target/confidence differences prevent pure wording attribution |
| Task requirements | GQA12, MMStar12, TextVQA6, native labels and case review | Exploratory coverage complete; sample counts do not establish general category differences |
| Selected model depths | Matched six-question input / after-block-16 zeroing and IH grids | Complete for two boundaries; no full layer curve or general depth law |
| Proxy interpretation and reproducibility | Reference/no-op/repeat controls; first/whole scoring; half/neighbor replacement; MMStar image/option checks | Completed bounded checks; no proxy certified as intrinsic importance |
| Actual selection/merge mechanism behind compression errors | Baseline/native compression outcomes plus observational source/slot records | Still unresolved; controlled algorithm ablations belong to RQ2 |

The three datasets are not three interchangeable estimates of the same population.
Their questions, target formats and selection roles differ. There are 30 questions,
not 17,280 independent examples; six GQA image groups also contain paired questions.

## Finding 1: E does not consistently rank positive answer support

The matched input-stage medians of per-question Spearman correlations are:

| Dataset | rho(E, zero-first) | rho(E, IH-first) | rho(zero-first, IH-first) |
|---|---:|---:|---:|
| GQA12 | −0.087 | −0.068 | −0.016 |
| MMStar12 | +0.009 | −0.121 | −0.005 |
| TextVQA6 | −0.031 | −0.136 | −0.028 |

Individual E–IH correlations range from −0.486 to +0.404: some questions align,
others reverse. The earlier high-/low-E zeroing comparison also lacks a uniform
positive ordering. This supports a mismatch between reconstructive distinctiveness
and signed answer support under the specified counterfactuals in these cases.

It does not show that E contains no useful information. Signed support differs
from response magnitude: absolute IH can align positively with E on individual
cases, including layer-16 Window (+0.652) and Dominion (+0.556). Nor does it show
that any alternative scoring rule would produce the same result. Photo-only
nonbasis filtering changes the input GQA median E–IH sign to +0.087, while the
overall relationship remains mixed. Preserve these qualifications.

## Finding 2: the same E ranking does not determine a question's sensitivity map

All six GQA pairs have exactly matching input features and E. Between-question
IH rank agreement ranges from −0.443 to +0.655; maps can differ substantially or
share substantial ordering. The historical whole-answer zeroing map for the
cake/plate versus stove pair is strongly reversed (nonbasis rho about −0.834).

This is a useful demonstration of question/target-conditioned responses at fixed
image features. The pair changes question, answer identity and confidence too;
it is not a pure causal isolation of wording, and different maps do not themselves
show that compressed answers will fail. Some illustrative pairs remain correct
after native compression.

## Finding 3: zeroing and retention answer different counterfactual questions

Zeroing asks how a specified vector replacement changes the full-context reference
score. IH-style physical retention asks what one retained visual state adds relative
to deleting all visual states at that boundary, with nonvisual states retained.
They change different context and sequence geometry. Both can be reproducible
without agreeing on a universal token ranking.

Input Sky mall state 522 illustrates this: IH is +0.81667 probability units,
while zeroing sensitivity is only +0.00267. Its best retained first-piece
probability is 81.70%, versus 94.75% with full context. The same state can show
large support relative to a null and a small effect when zeroed in full context.
Redundancy, competing evidence and nonlinear response are candidate explanations,
not mechanisms identified by these measurements alone.

Half-strength and neighbor-replacement checks add another boundary on interpretation.
Their rankings do not reliably match full zeroing. A Window example even reverses
sign at half strength. This is evidence of intervention-specific behavior, not
evidence that our recorded zeroing was numerically random, that IH is ground
truth, or that all token interventions are invalid.

Our retention operator is a source-informed 1×1 IH adaptation. The original
zeroing experiment was inspired by IH but is a different measurement. Neither
is an exact reproduction of the full original IH experimental setup.

## Finding 4: global correlation can hide sparse strong states

The full grid corrected an important calibration limitation. For input Dominion,
best retained first-piece probability increases from 0.0015% on the 32 sampled
positions to 20.2% on all 576. At layer 16 it rises from 0.0768% to 98.1%.
Every layer-panel best index was outside its earlier 32-position sample.

Layer-16 Dominion state 131 has E at the 11.3rd nonbasis percentile and large
positive IH, whereas strongest Window and Cookery states have high E. Thus we
have both alignment and low-E/high-response exceptions, not a universal rule
that useful states always have low E. Rank summaries, effect sizes and concrete
exceptions should appear together in the report.

These probabilities concern the first tokenizer piece (Dominion begins with
`▁Domin`), not freely generated full-answer accuracy. A retained state is already
contextualized by CLIP and, at layer 16, the preceding decoder blocks. Its grid
coordinate does not establish that only those raw pixels contain the evidence.
Searching all 576 positions for the strongest example is a descriptive search,
not independent confirmation or a population tail estimate.

## Finding 5: depth changes the maps without producing uniformly better alignment

In the six matched cases, input/layer E-rank agreement is about +0.59 to +0.79.
Zeroing-map agreement is about −0.04 to +0.30; IH-map agreement is about −0.04 to
+0.20. Representation reconstruction rankings are more stable across these two
boundaries than the corresponding measured sensitivity rankings.

There is no consistent improvement of E–I alignment at layer 16. However, late
text states have already received image information. A late visual-null condition
therefore does not remove all prior image information from the computation.
Different spaces, nulls and retained contexts prevent a pure causal depth claim.

**Correction to the preceding full-review note:** it previously suggested decoder
visual E could become question-conditioned merely because the decoder processes
the prompt. Our actual prompt places the image before the question, with causal
attention (`rq1_native.reference_tokens`, `smoke_original_apet.generate_native_answer`).
Visual states cannot attend to later question tokens. Native `layer_prune` computes
E only on the visual slice. With fixed image/prefix/seed, the later question has
no semantic route into these E values. Whole-sequence shapes can still produce
numerical differences; the same-image layer E agreement is nearly exact, not
necessarily byte-identical. The full-review wording is now corrected. This does
not change any score or require inference. Answer-side I remains question/target-
conditioned, and the late text states do see the image.

## Finding 6: baseline correctness and token coordinates need careful interpretation

MMStar grounding checks found three selected diagram cases always answered the
displayed letter A under image/blank/question-only conditions and rotated choices.
Five diagram cases therefore have explicit grounding concerns; the soccer control
followed the correct answer content through all real-image rotations. A correct
baseline option alone is insufficient evidence of grounded visual competence.
Keep flags and original cases visible rather than silently removing them.

TextVQA also exposed answer-score dependence. Cookery school's first-piece and
whole-answer rankings differ, and many effect signs reverse between them. A
partially predicted subword is not necessarily semantic answer preservation.
This limits broad task-comparison claims while making the measurement audit useful.

Native source identity also matters. Input Cookery's strongest retained state 149
is an FPS basis source with tiny E. Native ApET inserts basis features into selected
output slots and merges unselected sources into other retained features. Missing
an original source coordinate is therefore not equivalent to deleting its
information. This is an important constraint for RQ2's design.

## Closing RQ1 and proceeding

The later [literal-text failure review](2026-10-18-textvqa-exact-text-failures.md)
adds an important answer-level finding: baseline Brolio, Cookery school and Give
up become Bottoli, Culinary school and Gone under native input compression.
Original images and actual saved model canvases confirm the printed text. These
are concrete preservation failures even though singleton E/I maps do not isolate
their cause. Brolio's highest-zeroing-sensitivity state 317 is retained but
averaged with three other sources, making it a useful RQ2 merge-ablation case.

A supported report-level statement is:

> In a curated LLaVA-1.5-7B diagnostic panel, native ApET reconstruction-error
> rankings do not consistently align with signed answer sensitivity under
> singleton zeroing or source-informed 1×1 retention. Sensitivity maps vary with
> the question/target, intervention and measurement boundary. Full-position
> analysis reveals sparse strong responses that small calibration samples miss.

This is a substantive exploratory result. It does not establish that ApET is
ineffective or identify selection versus merging as the cause of a generated
answer failure. The empirical task/depth coverage is deliberately narrow.

The remaining RQ1 work is presentation and consolidation: retain one primary
matched table, the same-image comparison, a few qualified examples, and a compact
limitations/status table. Existing saved data are enough for those deliverables.
No new obligatory layer sweep or original-IH benchmark reproduction is implied
by the proposal's selected-example design.

Next prepare RQ2's input-only native-merge/no-merge pilot at 288 tokens, keeping
FPS sources, selected positions and basis source-to-slot mapping fixed. Compare
actual answers and preservation alongside reference scores. Only after a small
controlled pilot should it expand to development GQA800 and MMStar1500. Changing
selection and merging together would not isolate their contributions. Reserved
evaluation remains untouched for later confirmation.

Evidence notes: [full input/IH synthesis](2026-10-18-rq1-input-ih-full-review.md),
[full layer/IH comparison](2026-10-17-rq1-layer-ih-full-review.md),
[validity checks](2026-10-17-rq1-validity-review.md),
[score reanalysis](2026-10-17-rq1-score-reanalysis.md), and
[group/same-image analysis](2026-10-17-rq1-saved-data-analysis.md).

This consolidation reads saved evidence only. It changes no native algorithm,
runtime lock, measurement protocol, original archive, cohort or model prediction.
