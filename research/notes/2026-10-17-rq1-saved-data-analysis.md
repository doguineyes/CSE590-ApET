# RQ1 saved-data analysis — 2026-10-17

Completed the saved-data analysis proposed after the input-stage pilots. **No
GPU inference, new package installation, environment change or native ApET
modification was needed.** Original ZIPs, logs, extracted records and figures
were read without alteration. All 30 questions / 24 images / 17,280 canonical
singleton effects are included; calibration and MMStar3 repeats are not counted
again. This is post-hoc descriptive analysis, not a preregistered hypothesis test.

Start with the [local analysis gallery](../results/rq1-saved-analysis-v1-2026-10-17/index.html).
It contains the group tables, all six GQA pairs and selected image/E/I examples.
All 30 existing PNG figures are copied unchanged into its `figures/` directory,
with a SHA-256 manifest, so its main page can be viewed without external assets.
The [summary JSON](../results/rq1-saved-analysis-v1-2026-10-17/summary.json) and CSVs
preserve numerical details. These outputs are ignored by Git; this note, analysis
plan and CPU script are tracked. Keep the original exports backed up separately.

### Clearer figure gallery added later on 2026-10-17

The [padding-aware gallery](../results/rq1-padding-aware-gallery-v1-2026-10-17/index.html)
now provides 30 newly rendered PNG/PDF figures. Existing derived review pages
link to it through a banner and retain their previous HTML in an adjacent
`index-before-padding-aware-gallery.html` backup. Raw exported figures, ZIPs,
logs, numerical summary and CSVs remain unchanged.

- Cyan outlines show the photograph boundary on the model canvas and both maps;
  hatching identifies padding. All panels use aligned patch coordinates 0–23.
- Red outlines select high positive I; gold outlines select high E. The rules
  select two photo patches per score, with one additional positive-I padding
  patch and mixed-boundary patch when present. IDs and native roles are listed.
- `P` and dashed outlines identify entirely padded patches; `M` and dash-dot
  outlines identify mixed patches. Photo-coverage percentages explain borderline
  cases: Window patch 211 is mixed but 98.75% inside the photograph.
- GQA same-image questions use shared symmetric I color limits. Different images
  can have different limits. Padding measurements remain visible; the new region
  summaries are auxiliary and do not replace the primary full-grid analysis.

Start with **MMStar 136** for the distinction between photo patch 201 and padded
patch 21P; **TextVQA 35962** for high-I text patches 136/137 versus high-E patches
251/275; and the **GQA Window/Bottom pair** for question-dependent effects at a
fixed image boundary. The renderer is `scripts/render_rq1_gallery.py`, separate
from inference and numerical analysis. It uses Matplotlib 3.10.3 with Pillow and
NumPy in an isolated plotting environment; the editor and Kaggle locks are
unchanged. Four CPU geometry/marker tests pass. The geometry reconstruction
matched every saved 336×336 canvas exactly, including integer-centered padding.
The figures were visually checked after rendering. No GPU run is needed.

## Group comparison: high error does not consistently mean higher signed support

Within each image, exclude the ten native FPS basis sources and rank the remaining
566 patches by native reconstruction error E. Divide average ranks into thirds,
without splitting equal E values. In these data this gives **188 low, 189 middle,
189 high** patches. Report the ten basis sources separately. No group is selected
using I or correctness. The same image uses the same E groups for both questions.

For each question and group, report mean, median, interquartile range, extrema and
positive fraction of signed singleton I. Then calculate the question's contrast:

    delta = mean(I in high-E third) - mean(I in low-E third)

Positive delta supports the hypothesis that high-E patches have greater average
signed answer support under this intervention; negative delta reverses it. The
table reports the median of those within-question contrasts, **not** a difference
between separately aggregated group medians. Values are nats per reference token.

| Dataset | Questions with high mean I > low mean I | Median within-question delta | Median per-image mean delta | Four-group sensitivity: median high-minus-low |
|---|---:|---:|---:|---:|
| GQA | 2 / 12 | −0.002023 | −0.004692 | −0.002041 |
| MMStar | 6 / 12 | +0.000077 | +0.000077 | −0.000224 |
| TextVQA | 3 / 6 | −0.000146 | −0.000146 | +0.000101 |

GQA's paired questions share images, so the image column first averages the two
question contrasts within an image, then takes the median over six images.
MMStar and TextVQA each have one question per image. These summaries have no
pooled-token p-values and make no population claims.

The group comparison agrees with the weak/mixed rank correlations. There is no
consistent positive high-E/answer-support ordering in these selected input-stage
cases. Small MMStar/TextVQA median contrasts even change sign with four rather
than three bins: those near-zero signs should not be interpreted as stable
positive or negative effects. GQA's negative median persists with image-level
aggregation and four groups.

Primary scores remain **signed**. A negative I means zeroing increased the fixed
reference score; it does not mean the patch has no information. Means/medians of
separate singleton effects do not measure jointly removing or zeroing a group.
Neither adding effects nor averaging them gives a measured joint intervention.

Files: [question groups](../results/rq1-saved-analysis-v1-2026-10-17/question-groups.csv),
[question contrasts](../results/rq1-saved-analysis-v1-2026-10-17/questions.csv),
[dataset group summaries](../results/rq1-saved-analysis-v1-2026-10-17/dataset-groups.csv),
[all token assignments](../results/rq1-saved-analysis-v1-2026-10-17/tokens.csv).
Basis-source distributions and auxiliary absolute-I means remain visible, rather
than mixing basis sources into the low-E group or replacing the signed metric.

## GQA: fixed image features and E, different answer sensitivities

For each pair, verified identical CLIP/projected hashes, compression seed,
all 576 E values and native selection/basis/merge records. The table compares
signed-I rankings among the **566 non-basis patches**. Top sets contain the 57
highest signed-I scores per question, not the highest absolute values. There
were no ties at any observed top-set boundary.

| Image | Questions / fixed targets | Between-question non-basis rho | Top-57 overlap |
|---|---|---:|---:|
| n200692 | Cake/plate color (No); plate lies on what (Stove) | −0.834 | 0 / 57 |
| n455563 | Curtains/trash cans (No); medicine-cabinet side (Right) | +0.505 | 13 / 57 |
| n64959 | Refrigerator side (Right); appliance to its left (Stove) | +0.577 | 27 / 57 |
| n125122 | Device right of curtains (Television); sinks/mirror (No) | −0.138 | 2 / 57 |
| n319845 | Skateboards/cabinets (No); polished furniture (Table) | +0.319 | 9 / 57 |
| n578564 | Paper-towel position (Bottom); pan in front of what (Window) | −0.332 | 0 / 57 |

Some pairs share sensitivity rankings; others differ substantially or reverse
them. The same question-independent E ranking therefore need not match the
answer-support rankings of different questions about one image. Wording, answer
identity, answer length and reference confidence also differ. This establishes
question/target-conditioned proxy behavior, not isolated ground-truth relevance.

Detailed [pair table](../results/rq1-saved-analysis-v1-2026-10-17/gqa-pairs.csv)
includes answer lengths, reference scores, overlap/Jaccard and sign disagreements;
the [paired-token table](../results/rq1-saved-analysis-v1-2026-10-17/gqa-paired-tokens.csv)
preserves both effects and ranks. Sign disagreements require both magnitudes to
exceed 0.0001, an explicit descriptive cutoff, not a measured noise threshold.

## Visual examples and native roles

These examples were selected after observing the saved data and are illustrative,
not independent confirmation. The gallery reuses original aligned plots.

- **GQA Window / Bottom, n578564, patch 211:** non-basis E percentile 20.35%,
  low group. I is **+0.30548** for Window and **−0.000821** for Bottom. The
  original embedding is not directly retained, but can contribute through
  merging. Both archived input-only answers were correct. This illustrates
  question dependence; it is not a demonstrated compression-induced failure.
- **GQA Television, patch 215:** E percentile 50.62%, middle group; I=**+0.36821**
  versus −0.001593 for the same-image absence question. The native selected mask
  includes this slot, but a basis feature replaces its original embedding.
  A selected slot and a directly retained source are different objects.
- **MMStar bedroom 895, patches 344/347:** low/middle groups; I=**+0.29694 /
  +0.28952**. Neither original is directly retained. Sparse positive peaks can
  coexist with negative global rho and suggest targets for a later joint/native
  intervention; this does not prove the cause of the archived compression loss.
- **MMStar soccer 136:** high-E patch 201 is directly retained and has
  I=**+0.30513**. Low-E padded-border patch 21 also has I=+0.23529. Contextualized
  CLIP features and artificial zeroing prevent interpreting every coordinate as
  purely local pixel evidence. The compressed answer was preserved.
- **TextVQA Cookery school 35962:** patch 136 is middle-E (46.37th percentile),
  I=**+0.31589**; patch 140 is low-E (20.88th percentile), I=+0.07838. Both are
  outside direct-original retention. These are fine-detail candidates, not a
  proof that these individual removals caused the compressed answer failure.
- **TextVQA Brolio 37533:** high-E patch 317 has I=**+0.12858** and is directly
  retained, yet archived input compression answered incorrectly. Omitting the
  strongest supporting original patch cannot explain this particular loss;
  other selection changes, merging and joint/context effects remain possible.

Exact example identities, roles and paired effects are in
[illustrations.csv](../results/rq1-saved-analysis-v1-2026-10-17/illustrations.csv).

## Task and quality checks

The [strata table](../results/rq1-saved-analysis-v1-2026-10-17/strata.csv) separates
native task labels and archived loss/preserved roles. For example, all six GQA
relation questions (four images) have negative mean contrasts, while two of three
attribute questions have positive contrasts. Those counts are too small and
confounded by question/answer type to establish a general task effect. MMStar
categories have only 1–3 cases each; dataset-level contrasts do not isolate task.

Excluding MMStar's previously flagged cases 604 (two real choices) and 617
(option-name typo) leaves **5/10** with higher high-E mean I, with the same median
contrast +0.000077. This sensitivity check does not remove the flagged records or
change the frozen cohort. Correct option output remains insufficient proof of
visual grounding, especially for small chart labels and possible option priors.

## Validation and next work

The script rechecks original ZIP hashes, evidence/source hashes, cached-feature
tensor hashes, native 288-token role mapping, finite/complete E/I records,
reference-minus-perturbed arithmetic, accepted controls and target alignment.
Encoded image hashes and case/feature seed/hash agreement are also checked.
All 30 full-teacher-forced reference/repeat/no-op/native/final score comparisons
have zero drift. Cached-generation prefix differences are recorded separately:
TextVQA's previously documented later-token difference remains diagnostic under
its accepted control policy; no new scoring or tolerance change was made here.
Five meaningful CPU tests pass, including basis exclusion, signed effects, tied
E groups, anti-correlated same-image rankings, feature mismatch rejection,
overwrite refusal and the real saved cake pair.

**No additional Kaggle test is needed to complete or trust this saved-data
calculation.** Native code, original lock, reference targets and measurements are
unchanged. Further GPU work is needed for the remaining scientific questions:

1. A bounded robustness check on selected examples and controls, using a stated
   alternative perturbation or joint token group. This tests the zeroing proxy
   and interactions; it is distinct from this distribution analysis.
2. A small layer-16 calibration, then a matched-depth comparison, as described in
   the [input-stage checkpoint](2026-10-16-rq1-input-stage-summary-and-next-steps.md).
   Decoder-stage intervention instrumentation is still to be implemented.
3. Limited grounding checks if chart examples are used as key mechanism evidence.
   No need to rerun all baselines, expand datasets, or resume the deferred airplane
   numerical investigation for this step.

Reproduce from the repository root with only the Python standard library:

```bash
python3 scripts/analyze_rq1_saved.py \
  --output research/results/rq1-saved-analysis-v1-2026-10-17-new
python3 research/tests/test_rq1_saved_analysis.py
```

The output must be new and outside all original runs. Source ZIPs and extracted
folders must be present. The tracked
[analysis plan](../protocols/rq1_saved_analysis_v1.json) pins their identities,
hashes, grouping, sensitivity checks and illustrative-example selection.

The follow-up [validity diagnostics](../RQ1_VALIDITY_CHECKS.md) are now implemented:
72 MMStar image/option trials plus six reference repeats, then six fixed
questions × 32 patches × zero/half/neighbor replacement. One notebook can run
both parts with one model load and the unchanged v1 input bundle. The tracked
profile freezes the panel, patch strata and archived E/I before new outcomes.
Ten CPU protocol/export checks pass; Torch prerequisites and pretrained behavior
still require the first Kaggle execution. No validation result is claimed yet.

**Later checkpoint:** the complete validity run returned on 2026-10-17. The
[separate review](2026-10-17-rq1-validity-review.md) records exact zeroing
reproduction, image/option concerns on selected MMStar diagrams, intervention-
dependent rankings, and a locally repaired HTML export. The prior input results
and this saved-data analysis remain unchanged.

The review also records today's methodological interpretation and its connection
to later selection/merging/basis comparisons. Repeated measurements agree, while
the meaning of I depends on the intervention. Singleton scores cannot be assumed
to predict joint compression effects. Future records should identify conditions
with stable, weak or reversed effects; the present evidence does not establish
mathematical discontinuity or a universal failure of token interventions.
