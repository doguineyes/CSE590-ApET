# RQ1 saved-score reanalysis and audit — 2026-10-17

Completed without GPU inference or changes to original predictions, measurements,
accepted cohorts, native ApET sources or dependency versions. IH remains an
optional later comparison, not a prerequisite for the bounded layer pilot or RQ2.

The [complete local review](../results/rq1-score-reanalysis-v1-2026-10-17-complete/index.html)
contains the [summary/provenance](../results/rq1-score-reanalysis-v1-2026-10-17-complete/summary.json),
per-question and per-position CSVs, answer-piece statistics, same-image comparisons,
flagged-subset summaries, and a selected-panel norm audit. Earlier calculations
and original archives are preserved. This complete folder includes snapshots and
hashes of all analysis/scoring helpers. Its plan is explicitly post-hoc.

## Verified scope

Canonical sources are the completed GQA12, MMStar12 and TextVQA6 input runs:
**30 questions, 24 images, 17,280 singleton interventions**. Calibrations,
MMStar3 and later validity repeats are not additional canonical examples.
The [new analysis plan](../protocols/rq1_score_reanalysis_v1.json) fixes the source
ZIP hashes and records the scoring and sensitivity checks.

Rechecked source ZIP CRC/SHA256, all 77 original evidence-file hashes, 24 feature
file byte hashes, encoded image hashes, case/seed/feature identities, target
IDs/rendering/prompt/position alignment, complete unique patch grids, per-answer-
token score sums/means, and every stored reference-minus-perturbed effect. Fresh
reference answers still score correct under the existing scorers. All recorded
full-forward repeat/no-op/native/final comparisons have **zero drift**. Saved
feature tensors were not unpickled and E was not recomputed from those tensors.

Pinned LLaVA and CLIP revisions, Torch 2.1.2+cu121, Transformers 4.37.2,
FP16/SDPA, the original uv lock and eight monitored native source hashes agree
across the three runs. Source commits differ as project instrumentation evolved;
that history is preserved. The known TextVQA later cached-generation score gap
does not invalidate the full-teacher-forced comparisons under the accepted
control policy. No new arithmetic, identity or provenance failure was found.

## First-token versus whole-answer scoring

All three measures use the **same singleton-zeroing counterfactual**:

- `whole_log`: original mean reference-answer token log-probability difference.
- `first_log`: difference for the first fixed reference-answer tokenizer token.
- `first_probability`: difference of the corresponding first-token probabilities.

The original whole-answer score remains primary. The others are explicitly
labeled robustness checks, not replacement results and not IH reproduction.

Per-question E–I Spearman medians, excluding the ten FPS basis sources:

| Dataset | Questions / images | Whole-answer log score | First-token log score |
|---|---:|---:|---:|
| GQA | 12 / 6 | −0.09266 | −0.09266 |
| MMStar | 12 / 12 | +0.00873 | +0.00873 |
| TextVQA | 6 / 6 | −0.02162 | −0.03726 |

There are **21 one-token** and **nine multi-token** targets. All 12 MMStar
targets and nine GQA targets are single tokens, so first/whole scores are
identical by construction. For GQA's three multi-token targets, first/whole
non-basis rank agreement exceeds 0.99995. Thus averaging over answer tokens
does not explain the weak GQA/MMStar E–I rankings in these cases.

TextVQA has material case-level changes:

| Target | Whole E–I rho | First E–I rho | Whole/first I rank agreement |
|---|---:|---:|---:|
| Dominion | −0.04153 | −0.01792 | +0.97212 |
| Cookery school | +0.01407 | −0.15372 | +0.63347 |
| Give up | −0.05883 | −0.05573 | +0.99791 |
| Brolio | −0.01159 | −0.02855 | +0.96657 |
| Laughter | −0.03166 | −0.04597 | +0.95412 |
| Sky mall | +0.03991 | −0.02004 | +0.87340 |

For Cookery school, 358/566 non-basis positions reverse effect sign with both
magnitudes above the previously used 0.0001 descriptive cutoff. This cutoff is
not a measured noise floor. Its answer pieces have different E–effect relations:
`Cook` −0.15372, `ery` +0.17335, `school` −0.13764. Averaging their signed changes
can hide disagreement between pieces. We have not established that any one
piece's metric is the more faithful measure of semantic answer contribution.

The previous high-E versus low-E group-mean comparison also depends on scoring:
high exceeds low in GQA 2/12 and MMStar 6/12 under either log score; TextVQA changes
from 3/6 under whole-answer to 1/6 under first-token scoring. These are averages
of separate singleton effects, not jointly perturbed groups or population rates.

For every question, first-token log and probability scores give identical E–I
Spearman correlations, with no extra probability ties. This follows mathematically:
the full reference is fixed, and both scores decrease monotonically with the
perturbed first-token probability. It is not independent empirical confirmation.
Different raw magnitudes or group means can still arise under that transformation.

## Other interpretation checks

**Subword semantics.** A first token need not be a complete answer word: Bottom
starts with `B`, Laughter with `L`, Dominion with `Domin`, and Cookery school
with `Cook`. First-token scoring is therefore not automatically semantically
preferable. Whole-answer teacher forcing also gives later pieces the correct
preceding reference pieces; it is not a free-generated answer-quality measure.

**Same-image observations persist.** The cake/stove pair's between-question
non-basis I rho changes only from −0.83378 to −0.83327; the Window/Bottom pair
changes from −0.33151 to −0.33158. Their top-57 overlaps remain zero under both
scores. This preserves the observation of different question-conditioned maps
with identical E/features. Different targets, confidence and task still prevent
calling this a pure causal estimate of question relevance.

**Padding is not a simple explanation.** Restricting the secondary analysis to
photo-only non-basis positions gives whole/first median rhos of −0.04466/−0.04466
for GQA, +0.01929/+0.01929 for MMStar, and −0.06113/−0.05278 for TextVQA. These
remain weak/inconsistent. This does not establish that padding features are
irrelevant; CLIP features are contextualized. Canonical measurements keep every
position and retain separate basis, mixed-region and padding labels.

**MMStar grounding/quality flags do not explain everything.** Canonical n=12
remains unchanged. A labeled post-hoc sensitivity check without the five newly
flagged diagram cases has n=7 and median rho −0.00468. Removing those plus the
two earlier question-quality cases leaves n=5 and median rho −0.01517. Neither
subset is a new accepted cohort or a known clean/grounded sample. Most remaining
cases have not received the same image/option checks. There is no positive
alignment recovered by these exclusions in this small collection.

**Feature norm is a possible covariate, not an established cause.** The six-case
validity archive supplies norms for 32 selected positions per question, not for
the full grids. On its 16-position random strata, signed zero-I/norm correlations
range from −0.274 to +0.597, without a consistent positive pattern. Absolute-I/norm
associations also vary. Its targeted low-E/high-I groups often have larger norms,
but those groups were selected using prior I, so they cannot establish a general
norm explanation. Method, direction and strength remain entangled.

**Confidence and available records.** An almost certain reference has little room
for score improvement, bounding negative I. This matters for magnitude comparisons
between questions; subtracting a fixed reference does not itself change within-
question ranks. The archives retain perturbed reference-token scores, not full
perturbed vocabulary logits or free-generated answers for every patch. Thus we
cannot retrospectively recover all alternative-answer margins or count semantic
answer flips from these records. These are scope limits, not missing/corrupted
files; no rerun is required for the present RQ1 analysis.

## Consequence for the next step

The refined conclusion holds: **native input reconstruction error does not
consistently rank answer sensitivity under the specified singleton-zeroing
measurement in these selected cases**. Scoring choice affects some TextVQA
interpretations, while it cannot explain all weak associations or same-image
differences. No claim about ground-truth importance, IH outcomes, universal task
effects or ApET ineffectiveness follows.

Proceed with the **existing six-question × 32-position zero-only layer calibration**
as a bounded exploratory depth comparison. No need to add IH before that run.
Its saved per-answer-token scores will allow both scoring summaries afterward.
Raw E/I magnitudes are not calibrated across stages, and earlier layers may have
copied visual evidence into text states. The layer runner's pretrained controls
still await Kaggle; this CPU analysis does not validate that implementation.
Keep `RUN_FULL_LAYER_SWEEP = False`, review the calibration, then shift the main
effort to RQ2. Additional hypothesis-driven measurements can remain optional.

Reproduce in a new output folder, using only the existing editor environment:

```bash
environments/editor/.venv/bin/python scripts/analyze_rq1_scores.py \
  --output research/results/rq1-score-reanalysis-new
environments/editor/.venv/bin/python research/tests/test_rq1_score_reanalysis.py
```

Six meaningful CPU checks pass, including score/rank invariants, opposite
multi-token effects, corruption rejection, overwrite refusal and complete real
archive verification with original ZIPs unchanged. No packages were installed.
