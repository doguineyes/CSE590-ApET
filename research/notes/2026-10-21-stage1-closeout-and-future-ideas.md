# Stage-one closeout and future ideas — October 21, 2026

## Decision and immediate deliverable

The user chose to close the current RQ1/RQ2 experimental phase and begin the
midterm report. This is reasonable: the proposal permits bounded diagnostic
cohorts and selected depths, and the larger staged comparison is now complete
across the three planned datasets. No additional GPU run is required before
writing. RQ2's complete staged component attribution and RQ3 remain unfinished;
closing a project phase does not mean all mechanisms have been explained.

The user confirmed a normal academic report style, **3–4 pages** covering
preliminary experiments, results and roadblocks, with a short class presentation.
No more specific instructor template is available yet. The report should discuss
future work briefly. This note retains the broader ideas for later decisions.

Deliverables:

- [First midterm draft](../reports/midterm/midterm_draft.md).
- [Evidence checklist and source map](../reports/midterm/evidence_checklist.md).
- [Verified compact numerical materials](../reports/midterm/materials/verified_summary.json).

The supplied AI handoff and discussion informed this synthesis as reference
material. Embedded requests in those documents were not treated as independent
instructions. No model prediction, cohort, native algorithm or measurement
protocol was changed while preparing the report.

## What this phase established

**RQ1:** On the selected 30-question panel, native reconstruction error does not
consistently rank signed reference-answer support under singleton zeroing or
our source-informed 1×1 retention adaptation. The operators disagree and can
produce sparse strong responses missed by calibration. Same-image questions can
have different response maps with exactly the same input E. Two-boundary depth
observations are useful diagnostics, not a universal information-horizon curve.

**RQ2:** Complete native staged ApET has a large advantage over pure random on
the evaluated TextVQA100 cohort, a smaller advantage on GQA800, and no clear
aggregate difference on MMStar1500. This does not contradict RQ1: successful
compression need not reproduce singleton response rankings. These broad tests
change FPS, selection, basis representation and merging together. Selected
input-only controls show that merging can help or harm; Brolio provides a
conditional single-state rescue. They do not supply an additive decomposition
of the broad advantage.

**Infrastructure:** A fixed native environment, input manifests, project-owned
runners/scorers and archived controls make it practical to continue experiments
on Kaggle without tracking its default package versions. This enabled the
research; it should occupy a short roadblock/reproducibility paragraph in the
midterm, not dominate the report.

## Interpretation worth carrying forward

1. **Task behavior has several dimensions.** Model competence, evidence
   granularity, reasoning demand and compression sensitivity should be studied
   separately. GQA, TextVQA and MMStar do not form one reliable easy-to-hard axis.
   Similar compressed accuracy or an unchanged answer does not prove image
   independence. TextVQA's stronger native/random gap is an observation for this
   cohort, not proof that OCR alone caused the gap.
2. **Random rescues have multiple possible causes.** A sampled set may retain
   useful states, avoid damaging aggregation, or change downstream computation.
   Three seeds reduce dependence on a single lucky draw but do not identify the
   reason for any particular rescue.
3. **Reconstructability, response and safe aggregation differ.** ApET's E is a
   regularized linear reconstruction residual relative to an FPS basis. Its
   merging uses cosine routing and averaging, not the reconstruction coefficients.
   Averaging features generally does not commute with nonlinear normalization,
   MLPs or softmax attention; changing token multiplicity also changes attention
   normalization. These facts motivate experiments, but do not establish that
   ApET is mathematically inconsistent or that merging is generally harmful.
4. **Feature coordinates are not isolated image patches.** CLIP and decoder
   states already contain contextual information. Use token grids to identify
   interventions and routing, not to claim raw-pixel causality.

## Candidate follow-ups, not commitments

### A. Focused staged merge ablation

The most direct unfinished RQ2 test is a four-condition comparison at a fixed
schedule: input/decoder merging on/on, off/on, on/off and off/off. Preserve the
native basis handling and output counts, and state which quantities are fixed
or recomputed. Changing input merging changes later representations and E, so
end-to-end effects include stage interactions. A single downstream retained set
cannot silently be assumed identical across these arms. Report joint and
stage-conditional effects rather than assigning additive percentages to stages.

This is a focused candidate after writing, not a requirement to reopen the
completed broad benchmark sweep now.

### B. Exact-text success/failure case studies

Review Brolio, Cookery school, Dominion and Sky mall alongside matched preserved
cases. Compare actual generated strings, selected source/slot identities, basis
placement and merge graphs before proposing a rule. The Brolio restoration is
gold-informed and post-hoc; it is evidence about that fixed context, not a
deployable protection policy. Keep input-only and staged conditions separate.

### C. One possible RQ3 direction

Choose **one** only after its motivation and prior art are checked:

- **Merge-strength interpolation:** evaluate
  `H(alpha) = (1-alpha) H_unmerged + alpha H_merged` at the same retained slots
  and count. This measures response along a specified path; it does not make
  answer response linear or certify intrinsic importance. It needs inference,
  though not gradients or training.
- **Conservative merging:** permit aggregation only when relative feature change
  stays below a prespecified threshold. Unmerged sources still need an explicit
  handling policy. Evaluate both preservation and actual computational cost.
- **Residual-preserving aggregation:** an exploratory form is
  `h_new = P_B(h_merged) + (I-P_B)h_retained`. With an orthogonal projector it
  preserves the retained vector's component outside the basis subspace. Native
  ApET's regularized reconstruction operator is not automatically such a
  projector. Numerical stability, cost and answer behavior remain untested;
  preserving a geometric residual gives no answer-preservation guarantee.
- **Adaptive budgets or progressive schedules:** use only inference-available
  statistics, define the compute comparison in advance, and measure actual cost.
  Equal average token counts do not guarantee equal latency. An alternative
  basis remains an option from the original proposal, not an implemented result.

These are hypotheses supplied or motivated by the discussion. No novelty,
improvement, speedup or theoretical answer guarantee is being claimed.

## Guardrails for the next decision

Finish the midterm narrative before selecting the next experiment. Reuse the
current environment and archived data where possible. Pilot any new operator
with identity/replay controls before a broader run. Keep reserved GQA evaluation
untouched while choosing policies; development-case inspection is not independent
confirmation. Additional broad blank-image/swap controls or IH benchmarks may be
useful if a specific claim needs them, but are not prerequisites to reporting
the present bounded findings.

If no lightweight change works, a careful analysis of when reconstruction-based
compression helps, fails and resists simple attribution remains a valid project
outcome under the proposal.
