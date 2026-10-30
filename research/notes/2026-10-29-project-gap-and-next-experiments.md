# Project gap and candidate next experiments — October 29, 2026

This compares the actual three-page proposal, completed experiment reviews, saved-data feasibility analysis, and submitted midterm report. It is a planning assessment, not a new frozen protocol or an instruction to run every candidate. No model experiment was run for this note.

## Where the remaining gap lies

The project already has enough infrastructure and broad benchmarking. The main remaining scientific gap is **component attribution on outcome-unselected data**: which parts of ApET help, at which boundary, under which fixed conditions? A complete theory of every answer change is outside the bounded project scope.

| Component | Evidence already available | Remaining gap |
|---|---|---|
| Execution/reproduction | Locked native LLaVA environment; actual two-stage counts; archived identity/replay checks | Essential verification for new operators; controlled timing only if making our own speed claims |
| RQ1 | E/zeroing/retain-one measurements across 30 selected questions and three datasets; six paired depth cases; same-image comparisons; operator-validity checks | Restricted cohort, depths and operational measures. These limits need clear reporting, not an automatic new full-token sweep |
| Whole-procedure RQ2 | TextVQA100, GQA800 and MMStar1500; both schedules; three pure-random seeds; APR/loss/gain records | These comparisons change ranking, basis handling and merging together, so they do not explain the source of the advantage |
| Component RQ2 | Selected-panel E/random ordinary retention × merge/no-merge; routing/protection; Brolio restoration | Primarily input-only288, 30 baseline-correct cases chosen using earlier native outcomes. Conditional effects are real evidence, but population ranking and staged attribution remain open |
| RQ3 | Exploratory image-E loss predictor with weak, dataset-dependent results | An actual adaptive policy was not evaluated. Either justify a bounded negative feasibility conclusion with revised scope, or test one motivated extension after the mechanism study |

The proposal's success criterion explicitly permits informative negative results; outperforming ApET is not required. It does propose exploring one lightweight extension, so a final report should distinguish a negative feasibility probe from a completed policy evaluation and explain any scope change.

## Minimum useful finish for an analysis-focused project

1. Complete one matched component comparison on at least 100 questions sampled without conditioning on native wins/losses.
2. State exactly what is held fixed and which stage is changed. Report conditional effects and interactions, rather than percentages of effectiveness assigned to components.
3. Give a concrete explanation supported by the comparison, or a bounded negative result with adequate checks that the intervention actually acted.
4. Check any claim intended to generalize on another fixed cohort. A proposed improvement needs unused evaluation cases after its rule is frozen.
5. Consolidate figures, uncertainty, answer transitions and limitations. Keep RQ3 optional unless the analysis produces a defensible candidate; explain this change from the original extension ambition.

This does not require another model, another large benchmark suite, or another complete 576-position intervention sweep.

## Priority 1: a clean selection × merging comparison

**Question:** Conditional on the native basis and upstream representation, does E-ranking of ordinary retained states help, and does its value depend on merging?

Use the existing whole TextVQA100 screening pool first: it has a clear native–pure-random gap and is not a pool selected from the 30 diagnostic failures. It is already inspected development data, not a new blind test set. Keep the original prompts, scorer and environment.

A particularly clean staged starting point is the **second boundary**:

- Keep the native first-stage 576→96 compression and its realized representation fixed.
- After block 16, always produce 32 visual states, including ten native basis representations.
- Hold the basis source identities and their output-slot assignment fixed within each comparison.
- Compare E-ranked versus random ordinary retained sources, each with native merging versus no merging.

| Ordinary source selection at block 16 | No merging | Native similarity merging |
|---|---|---|
| E-ranked, native basis fixed | E/drop | E/merge: native reference |
| Random, same basis fixed | Random/drop | Random/merge |

Three random seeds give six random arms plus the two E arms. Random-source eligibility, order, duplicates of basis sources, position IDs and cache behavior must be defined before results. Reuse each selected set for its merge/no-merge pair. This is a **conditional ordinary-ranking ablation**, not a replacement for the already completed basis-free pure-random baseline.

Why start at the second boundary? The upstream representation can be identical across arms, and no later compression stage needs to be reselected. Thus it gives a clear stage-specific test while retaining the actual 96→32 schedule. It does not identify all first-stage effects or the contribution of basis placement itself.

Report generated-answer scores, APR, losses/gains, paired image-group intervals and:

- E-minus-random under each merge rule.
- Merge-minus-drop under each selection rule.
- Their interaction: whether the merge effect changes with selection.

Possible outcomes are useful in either direction. E can help under both merge rules, help only with merging, or show no clear advantage conditional on the fixed basis. None alone licenses a universal percentage decomposition or proves that the basis explains the remaining whole-method gap.

If a first-stage test follows, distinguish an input-only mechanism diagnostic from a native staged policy test. In a staged test, changing input compression also changes later features and possibly E/basis/selection. Keeping the decoder *rule* fixed does not keep its realized selection fixed. That is a total upstream-policy effect, not an isolated direct effect of one input feature.

Do not infer anything about native96→32 from a new input-only64 test; these are different computations.

## Priority 2: test the emerging aggregation-role explanation

There is already an interesting structural observation in the selected routing panel:

- Lowest-E quartile **among ordinary retained targets** receives 4,569/6,912 incoming sources, or 66.10%.
- Highest-E quartile receives 126/6,912, or 1.82%.
- Protecting the highest-E 28 targets reroutes only 21 sources (0.304%), with no routing change on 15/24 images.

These are relative ranks within the retained targets, not a claim that these targets have the lowest E among all 576 original positions.

**Hypothesis:** native compression may preserve distinctive high-E states relatively unchanged while using other retained states as aggregation recipients. The FPS basis provides another protected representation path. This could help explain why a global reconstruction procedure works despite weak singleton E–sensitivity agreement. It is a hypothesis, not an established mechanism.

**Cheap next step:** use saved E and routing/count records to test whether the rank–incoming-mass pattern persists in TextVQA100 and the other broader cohorts, separately by boundary and schedule. Count shared image encodings once where appropriate. This count/rank audit needs CPU analysis, not new inference. Feature-change norms require actual saved vectors or new instrumentation where those vectors are absent.

**Possible GPU follow-up:** with one selected set and basis fixed, compare no merging, half-strength merging and native merging:

`h(alpha) = (1-alpha) h_unmerged + alpha h_native_merged`, for alpha = 0, 0.5, 1.

All retained counts and slots stay unchanged. Report full-answer outcomes, losses/gains and actual representation changes. This probes an averaging effect; it does not guarantee smooth, monotonic answer behavior. A norm-restoring variant could be a later diagnostic, not another required axis now.

A null protection test is difficult to interpret when it changes almost no assignments. Any future protection study should record how often the policy acts and how many states/sources it changes, alongside answer scores.

## Priority 3: distinguish reconstructability from safe replacement

**Question:** If a state has small reconstruction error, does replacing it with its actual linear approximation preserve answer behavior better than removing or zeroing it?

On a small prespecified sample, construct the native linear approximation before projection, replace selected nonbasis states with it, and keep the sequence length and positions unchanged. Start with a few E-stratified states per question; no full-grid sweep is necessary. Compare same-image questions where useful.

Include identity/reference controls, generated-answer correctness and a fixed whole-reference score. Keep first-token scores as auxiliary if used. Small E also means a smaller perturbation, so include a matched-norm perturbation control before attributing a result to the special reconstruction direction rather than perturbation size.

This is a representation diagnostic, not a compression algorithm: the count is unchanged. Native ApET does not explicitly reconstruct discarded states during inference, but their information may remain represented through the basis or correlated retained features, as well as merged averages. A replacement result therefore cannot be equated with the benefit of actual token dropping or merging.

This candidate is useful if the component experiment leaves the linear-reconstruction assumption as the main open issue. It is not a prerequisite for finishing the selected RQ2 comparison.

## Complementary case analysis with saved predictions

Review the native TextVQA losses and gains at both schedules, with preserved controls, using a written rubric. Useful categories include literal text substitutions, omissions, semantic answer changes, acceptable wording variants penalized by the scorer, and questionable labels.

Keep real spelling corrections separate from scoring artifacts: a different word becoming the correct printed word is not automatically a scorer problem. Likewise, a similar-looking wrong word does not alone prove that specific letter features were merged away.

Brolio supports a local effect of changing a retained representation under input-only288. It does not automatically explain Brolio's different staged96→32 output. Two or three additional targeted restoration cases could document distinct failure modes if needed; they remain post-hoc diagnostics, not an unbiased estimate of how often restoration works.

## When to consider RQ3

Only after choosing the explanatory comparison:

- A consistent aggregation problem could motivate a fixed conservative merge-strength rule.
- A reproducible question-type difference could motivate a question-aware budget policy using inference-available features, without gold answers.
- An E-only predictor with weak performance can be reported as a limited negative feasibility result; it does not rule out all adaptive budgets.

Any adaptive rule needs a matched-cost comparison. Higher accuracy from allocating more tokens is not by itself a policy improvement. Compare with fixed-budget native ApET at a comparable average budget and a random budget allocator with the same budget histogram, then confirm on unused cases. Token-count equality alone is not a measured latency speedup.

## Recommended order

**Saved routing/case audit → one frozen component calibration → TextVQA100 component comparison → one bounded confirmation → final synthesis.** Choose the reconstruction-replacement diagnostic or an RQ3 candidate only if the preceding result gives a reason. The research contribution can be a clear explanation of one conditional mechanism and its limits, including a well-controlled negative result.

Sources: the project proposal v2; October 18 RQ1 full review; October 19 crossed/routing/oracle reviews; October 21 staged reviews; October 25 saved E feasibility exports; submitted midterm report v8-2. The input-only and staged scopes remain distinct throughout.
