# RQ1 input-stage checkpoint and next steps — 2026-10-16

The first input-stage measurement milestone is complete across all three
proposal datasets. This is an initial RQ1 result, not the completed layer/task
study and not an evaluation of a new compression method.

## Verified results

Only the canonical full-grid runs are counted below. Calibrations and the
earlier three-case MMStar sweep are reproduction checks, not extra samples.

| Dataset | Questions / images | Patch interventions | Median signed rho | Rho range | Median excluding FPS basis |
|---|---:|---:|---:|---|---:|
| GQA | 12 / 6 | 6,912 | −0.08684 | −0.30401 to +0.09272 | −0.09266 |
| MMStar | 12 / 12 | 6,912 | +0.00912 | −0.08087 to +0.17725 | +0.00873 |
| TextVQA | 6 / 6 | 3,456 | −0.02299 | −0.03970 to +0.04255 | −0.02162 |
| Total | **30 / 24** | **17,280** | — | — | — |

All 30 fresh uncompressed reference answers were correct under their respective
scorers and aligned with the fixed measurement targets. Each run passed its
recorded numerical controls. The original source/lock are preserved; source
commits differ as project instrumentation evolved. TextVQA uses the documented
full-teacher-forced control policy: later cached-generation score differences
are diagnostic, while full-forward controls and target alignment are enforced.

The [combined local gallery](../results/rq1-input-stage-summary-2026-10-16/index.html)
collects all 30 image/E/I plots. Its
[JSON summary](../results/rq1-input-stage-summary-2026-10-16/summary.json) records
source hashes and cohort statistics; its
[question CSV](../results/rq1-input-stage-summary-2026-10-16/questions.csv) keeps
identities, targets, roles and per-question statistics. Raw and derived results
are ignored by Git. Detailed reviews:
[GQA](2026-10-16-rq1-gqa-input-pilot-v2-review.md),
[MMStar](2026-10-16-rq1-mmstar-twelve-case-review.md),
[TextVQA](2026-10-16-rq1-textvqa-full-pilot-review.md).

## What we can currently say

**In these reviewed cases, native input reconstruction error is not a consistently
strong positive ranking of signed reference-answer sensitivity to singleton
projected-feature zeroing.** Weak or negative alignment occurs in all three
datasets; some MMStar cases show weak positive alignment. Excluding FPS basis
sources does not materially change the cohort medians.

GQA additionally holds the image/E fixed across six question pairs. Sensitivity
maps can differ substantially for different questions about the same image.
This is useful evidence for question/answer-dependent effects, though target
length, answer confidence and task also differ. It is not an isolated estimate
of visual ground-truth relevance.

Several cases deserve detailed comparison: GQA window versus paper-towel
questions on image `n578564`; MMStar 895 (bedroom) and 136 (soccer); TextVQA
Cookery school and Brolio. The last example already cautions against explaining
all compression losses by omission: its highest positive-I patch is directly
retained. Joint selection, merging and context changes remain candidates.

Limits to preserve in the report:

- These are curated baseline-correct, outcome-enriched mechanism subsets, not
  random dataset samples. The loss/preserved roles come from archived native
  compression runs; interventions score the full uncompressed reference model.
- E is in pre-projector CLIP feature space; I is measured by zeroing one
  post-projector embedding. Sequence length, mask and positions remain fixed.
  Native ApET instead selects/merges features and shortens the sequence.
- Signed I measures support for the fixed reference answer. Negative I means
  zeroing increased its score. Absolute-I statistics are an auxiliary measure
  of influence magnitude, not a replacement for the preregistered signed score.
- Small rho does not establish that ApET is ineffective. Redundant evidence,
  distributed features, token interactions, padding context and the zeroing
  proxy can all affect this measurement. Basis and merge roles matter.
- The answer formats differ across datasets. Do not interpret these median
  differences as a demonstrated task effect, pool all patches as independent
  samples, or treat the 30 questions as 30 independent images.

## Recommended next experiment

**Do not rerun large baselines or expand these input sweeps now.** Keep the current
input evidence fixed, review the maps, and implement a small decoder-stage
calibration next. No new environment or dataset construction is needed.

1. Fix a small panel from the existing cases before decoder measurements:
   a GQA same-image loss/preserved pair, one loss and one preserved MMStar case,
   and one loss and one preserved TextVQA case. Save their IDs, selection reasons,
   target IDs and seeds. This is a diagnostic panel, not a new random sample.
2. Start with native **layer 16**, which is after decoder block 16 (module index
   15), before block 17. The existing native forward calls `layer_prune` at this
   boundary. Keep all 576 visual tokens upstream and disable actual input and
   decoder pruning in the measurement reference, to isolate depth.
3. Observe native reconstruction E in that layer's visual hidden states using
   the original routine on a protected copy, discarding its compressed output.
   Intervene on the corresponding visual hidden-state position at the same
   boundary and continue the unchanged downstream model. Keep positions, masks,
   targets and scoring fixed; save stage, mapping, features and provenance.
4. First run only **32 patches per question**. Require unchanged-forward repeats,
   read-only observation parity, a no-op intervention, fixed-target alignment and
   final-reference controls before interpreting any rho. Inspect runtime and
   maps; only then expand the same panel to all 576 positions.
5. Compare input versus layer-16 E/I rankings **within the same questions**. Raw
   reconstruction errors are in different feature spaces and should not be
   compared numerically across stages. Visual positions remain traceable but
   their hidden states can contain distributed contextual information.

The layer-stage runner is **not implemented yet**; the current CLI only supports
input interventions. This note is the plan for the next code change, not an
instruction to run a nonexistent `--layer` option. Native ApET source should stay
unchanged; project instrumentation observes or temporarily hooks the boundary.
In this image-before-question causal layout, visual decoder states cannot
attend later question tokens, so deeper visual E is not automatically
question-conditioned. Downstream answer sensitivity is question-conditioned.

Before drawing a broad mechanism conclusion, add a bounded robustness check
on a few already identified examples (and controls): compare zeroing with a
feature replacement or small group intervention, then test relevant native
selection/merge behavior. Freeze that perturbation and panel before execution.
This connects the sensitivity proxy to actual compression and later RQ2; it
does not require another full GQA/MMStar baseline run now.

All needed MMStar12 ZIP/log evidence is saved. The previously documented newer
TextVQA audit's separate console log remains optional and unavailable; its
report/details are embedded in the full pilot. No rerun is needed for that gap.
The Kaggle GPU can be stopped while the next measurement is implemented and
reviewed; retain the downloads and cached inputs through the normal workflow.
