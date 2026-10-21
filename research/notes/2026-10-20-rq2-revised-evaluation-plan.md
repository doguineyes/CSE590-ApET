# RQ2 revision: pure random retention and compression severity — 2026-10-20

Implementation update: the first [six-question calibration](../RQ2_PURE_RANDOM_CALIBRATION.md)
is now implemented and its [Kaggle run passed](2026-10-20-rq2-pure-random-calibration-review.md).
Actual 64-token inputs, 26 prerequisites and 66 primary predictions verify. Native64
gets 2/6 correct versus 2–4/6 for pure random in this selected panel, providing no
evidence yet of an ApET advantage under stronger compression. The broader
TextVQA100/multi-dataset runner remains a later
step. Existing native sources, frozen protocols and old results are unchanged.

This note records the revised design. It supersedes the earlier
recommendation to make the existing basis-fixed random arm the principal broad
random baseline.

## Why revise the next step

The [proposal](../reports/Peng_Chen_CSE590_ApET_Project_Proposal_v2.pdf), section 6,
calls for random token selection at the same retained-token budget as a simple
baseline. Its RQ2 asks about reconstruction-error selection versus similarity
merging. These call for both a simple whole-procedure comparison and controlled
component ablations; one cannot substitute completely for the other.

Our completed random arm is a component ablation: keep native FPS basis sources,
E-dependent replacement slots and basis assignment fixed, then randomize the
ordinary retained positions. It still computes E. Call it **random ordinary
retention with native basis fixed**, rather than pure random pruning. Its results
remain valid with that scope. They do not measure the benefit of the entire ApET
pipeline over an E-free, basis-free random compressor.

The 30-question panel also includes known compression failures, and all its
uncompressed answers are correct. Its results support diagnostic observations,
not a general ranking of compressors. All current RQ2 algorithm comparisons use
288 input tokens. We need broader predefined pools and a more aggressive budget.

## What is already complete

- Native merging versus no merging at fixed selection/basis, all 30 questions.
- E/random ordinary retention crossed with merging/no merging, all 30 questions.
- Fixed-selection shuffled routing and high-E/random target protection, all 30.
- Separate post-hoc Brolio restoration: restoring only target 317 recovers the
  answer in the recorded context; sham/control 141 do not. This is conditional
  sufficiency, not proof of a unique necessary repair or a deployable method.

Routing/protection is not awaiting calibration or a full run. Both passed; advice
that describes them as pending is stale. Their small effects and low high-E
protection activity are already recorded in the
[full review](2026-10-19-rq2-routing-full-review.md).

No completed panel requires repetition merely because this plan changes.

## Primary new comparison

Use one full 576-token reference and these methods at **K = 288 and K = 64**:

| Method | Basis handling | Ordinary selection | Merging |
|---|---|---|---|
| Native input ApET | Native ten FPS sources and replacement | Native E ranking | Native nearest |
| Native input ApET without merging | Identical to its native partner | Identical to its native partner | Off |
| Pure Random-K | None | Uniform K of the original 576 positions | Off |

Pure Random-K must perform no FPS, reconstruction solve, E ranking or basis
insertion. Its K original CLIP feature vectors pass through the same native
projector and multimodal insertion/position policy, in original spatial order.
It physically supplies K visual states; this is not an IH intervention or a
576-state sequence with zeros. At K=576 it must reproduce the full reference.

Use three prespecified random replicate IDs, 101/202/303, with an isolated RNG
derived from image identity and a new protocol version. Generate one permutation
per image/replicate and retain its first K entries, then sort by spatial index.
Thus the 64-position sample is contained in the 288-position sample. Same-image
questions reuse the same realized representation. Seeds never depend on answers.

Keep native basis count at ten for both budgets. Native ApET at 64 has 54
ordinary targets plus ten basis slots. The basis share changes from 3.47% to
15.63%; report this rather than attributing every budget effect to E ranking.
Native basis slots are chosen by the native rule at each budget; they need not
coincide across budgets. Do not force native selected sets to be nested.

All initial comparisons are **input-only**, with decoder pruning disabled. K
means the number entering the decoder, not a final-layer or average-per-layer
token count. This isolates a fixed compression boundary and continues our
288-token experiment. It does not reproduce a paper's staged 64-token configuration.

## Keep the questions distinct

Native ApET versus Pure Random-K compares complete compression procedures.
Differences combine selection, basis replacement and merging; they cannot be
credited solely to E. Native versus native-no-merge isolates merging conditional
on that native selection and basis assignment.

The existing basis-fixed crossed experiment supplies the narrower ordinary-E
selection ablation at 288. A small new 64-token component panel can be added if
the broad results motivate it; the old 288-token protocol must remain frozen.
Do not automatically expand every old routing/protection arm to both budgets.

A useful optional second-stage control is **Pure Random-K + nearest merging**:
use exactly the same sampled original targets as Pure Random-K, but merge dropped
sources into all K retained targets using a declared cosine/averaging rule, with
no basis machinery. This tests merging for a pure random selected set. If added,
freeze its definition and paired controls before running it; a comparison with
native ApET still does not isolate E alone.

Differences between basis-fixed random retention and pure random retention would
also include basis placement, eligibility and possibly merge-target differences.
They would not, by themselves, establish an FPS-specific benefit.

## Budgets and relation to the literature

K=288 retains 50% of 576 states; K=64 retains 11.1% (88.9% removed). This tests whether
the incremental benefit of specialized compression changes under greater
compression. It does not assume that random must fail or E must improve at 64.
Add an intermediate budget only to resolve a specific ambiguity in these results.

ApET reports 192/128/64-token settings and evaluates random **basis sampling** in
Table 5; that random basis comparison is not Pure Random-K.
[ApET sections 4.2–4.3](https://arxiv.org/html/2602.19870v1#S4).
IH reports competitiveness of random pruning in deep layers and uses staged
schedules with average-per-layer retention in its supplementary Table 5. Those
results motivate a random control, but do not establish competitiveness of pure
random input pruning at 64 in our setup.
[IH sections 4.1 and 7](https://arxiv.org/html/2512.07580v2#S4).

## Dataset and execution sequence

1. Validate a new pure-random implementation with the current original runtime:
   full 576-token parity, exact selected sets/count/order, skipped FPS/E, native 64
   identity, unchanged native source hashes, same-image reuse and bounded repeats.
   New pure-random and 64-token paths need a small calibration; old passed experiments
   are not retroactively rerun. Measure cost before choosing broader run sizes.
2. Run the core comparison on the entire existing **TextVQA100** at both budgets.
   This is a screening/development cohort, including the previously studied six
   cases, not an independent held-out or full-TextVQA evaluation. Keep all cases,
   not just baseline-correct examples or known spelling failures. Export the
   complement of the six studied cases as a secondary sensitivity check.
3. Freeze the next broad comparison and run **GQA development800**. Same-image
   grouping matters for paired uncertainty estimates. Keep the reserved GQA
   evaluation images untouched until protocol choices are settled.
4. Include **MMStar** as the proposal's broad task comparison: use all 1,500 if measured
   runtime permits, or freeze an outcome-independent subset before inference.
   Report accuracy and baseline-correct preservation separately; retain known
   qualitative grounding flags. It should not silently disappear from RQ2.

Record per-seed results, means/ranges, paired gains/losses and dataset-appropriate
scores. For TextVQA distinguish consensus mean from full-credit preservation.
Replicate seeds are not independent question samples. Use fixed prompts, image
preprocessing, precision, generation settings and native position policy.

Broader runs should preserve predictions, scores, input identities, selected
indices, basis/merge summaries where applicable, seeds, source/lock provenance
and logs. Reserve large full tensor snapshots for calibration and declared
diagnostic cases, rather than automatically producing multi-GB evidence for
every broad condition. Original archives stay intact.

## Interpretation and stopping points

- ApET wins only at 64: evidence for a budget-dependent advantage in these pools,
  without identifying which component is responsible by itself.
- Pure random remains competitive at 64: examine paired uncertainty, seeds,
  task dependence and basis-fixed ablations before questioning E generally.
- No-merge effects vary by budget/task: report the interaction and exact-text
  preservation failures; do not impose a universal merge benefit/harm story.
- Apparent random advantage disappears on broader pools: report diagnostic
  selection effects openly. The earlier rescues remain valid case evidence.

This revision continues RQ2; it does not require a new ApET improvement. It also
does not assume that reconstructive E, either intervention-specific RQ1 score,
or incoming merge load is ground-truth token importance.
