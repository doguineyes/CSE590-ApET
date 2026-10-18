# RQ1 full input / layer-16 / IH comparison — 2026-10-17

The full six-question, 576-position comparison passed. **The Kaggle GPU can be
stopped.** Both supplied raw files are archived locally, and no inference rerun
is needed. This completes the planned bounded input/layer-16 comparison, not a
whole-decoder or population study.

The full grid changes one important interpretation of the 32-position
calibration: **strong retained states were missed by that sample**. Window,
Cookery school and Dominion each have a retained state with higher first-target
probability than the full-context reference. The sampled singleton/full gap
therefore does not establish that support must be distributed or require
synergy. The broader finding remains: native reconstruction error is not a
consistently positive predictor of signed sensitivity under either intervention.

## Archive and integrity

Original Downloads copies were retained. Copies in `research/results/`:

- `rq1-layer16-ih-sixcases-v1-20261018T014830278963Z.zip`: 59,953,110 bytes;
  SHA256 `e396b7a1c0f0c554e2642ac4e0ac9a83d4f9c4b2a4ecdfbfee6e86bb5a845b37`.
- Matching `.log`: 28,215 bytes;
  SHA256 `5a592375ad99aafd4545e493a1ca300302e82cd7d6c030f62b8a49362eddf219`.

UTC filenames are October 18; this review uses October 17 local time. Runtime
commit `08d0629c16da77269c9c8d3105af70805e7d39f4` was clean. Original native
ApET sources and lock are unchanged. The ZIP has 93 unique members, passes CRC,
and preserves 50 verified evidence files and 11 feature snapshots whose file
bytes were hashed without loading tensors.

The actual prerequisite was the user's **rerun** calibration ending in
`20261018T013751203709Z`. Its passed report is embedded as
`evidence/prerequisite-layer-calibration.json`, with SHA256
`fb7a35d5205124e98d3b61b615926e6a10eddf8864744ca915e4fb2e485612d5`.
Its source/model/profile/lock gate passes independently; a changed filename is
not a problem. Its separate ZIP/log were not supplied. This is an optional
archival gap, not missing measurement controls and not a reason to rerun.
The earlier `002129035074Z` calibration ZIP/log remain archived too.

All six fresh baseline answers are correct and target-aligned. All recorded
reference, no-op and retention controls have zero drift; within-question
boundary identity checks pass. Independent verification checks every raw score,
target, intervention index, native E value and retained original position map:

- Input zeroing: 3,456 scores, exactly repeating the six corresponding canonical
  input sweeps, including raw target-token scores and E.
- Layer-16 zeroing: 3,456 scores.
- Layer-16 retain-one: 3,456 scores.

Total: **10,368 measurements**, plus controls. All 384 zeroing and 192 IH scores
from the earlier 32-position calibrations repeat exactly, including raw score
records. These are reproducibility checks, not additional independent questions;
the canonical input-stage sample remains 30 questions / 24 images / 17,280 scores.

Wall time was **5,880.25 seconds = 98.00 minutes**. Zeroing phase: 72.64 minutes;
IH phase: 25.19 minutes; the remainder is shared overhead. IH-phase memory peaks
were 8.13/7.21 GiB. Those counters were reset for that phase and are not a
whole-run peak or a controlled speed/memory benchmark.

## Matched first-token comparison

At the boundary after block 16, before block 17:

`I_zero_first = p(first target piece | full) − p(first target piece | zero one)`

`I_IH = p(first target piece | retain one) − p(first target piece | remove all at layer 16)`

All 576 positions are included. These are signed **probability differences**,
not whole-answer accuracy or interchangeable intrinsic token importance.

| Case / target | E–zero-first rho | E–IH-first rho | Zero-first–IH-first rho |
|---|---:|---:|---:|
| GQA 202218829 / Window | −0.205 | −0.334 | +0.326 |
| GQA 202218593 / Bottom | +0.019 | −0.110 | +0.097 |
| MMStar 895 / Bedroom (B) | +0.014 | −0.101 | +0.110 |
| MMStar 136 / Soccer (B) | +0.006 | −0.094 | +0.004 |
| TextVQA 35962 / Cookery school | −0.048 | −0.003 | +0.036 |
| TextVQA 35670 / Dominion | −0.005 | −0.099 | +0.098 |

Neither measure shows consistently strong positive alignment with native E.
Window is moderately negative, so “all correlations are zero” would be wrong.
Zeroing and retention also have limited shared rank agreement. Dropping the ten
FPS basis sources per question does not reverse the broad finding: IH rhos are
−0.346, −0.119, −0.101, −0.087, +0.016 and −0.089. Photo-only nonbasis IH rhos
are also nonpositive here; padding exclusion does not explain the result away.
These are sensitivity analyses, not changes to the primary cohort.

E does associate positively with **absolute** IH response in some cases:
Window +0.652, Bottom +0.359, Cookery +0.430 and Dominion +0.556. A relationship
with response magnitude is different from positive support for the target.
Do not claim E is unrelated to every possible output response.

## Strong states missed by the calibration

The values below are probabilities for the first fixed gold tokenizer piece.
`Bottom` starts with `▁B`, Cookery school with `▁Cook`, and Dominion with
`▁Domin`; a first piece is not necessarily a complete word or semantic answer.

| Target | Full | Remove all at layer 16 | Best of earlier 32 | Best of all 576 | Best index |
|---|---:|---:|---:|---:|---:|
| Window | 27.097% | 6.994% | 12.123% | **94.063%** | 211 |
| Bottom | 98.849% | 99.035% | 99.093% | 99.107% | 438 |
| Bedroom / B | 63.076% | 62.612% | 62.628% | 62.963% | 301 |
| Soccer / B | 79.718% | 80.017% | 80.066% | 80.102% | 417 |
| Cookery school | 70.205% | 14.311% | 15.237% | **74.192%** | 149 |
| Dominion | 94.716% | 0.0608% | **0.0768%** | **98.101%** | 131 |

Every best index was outside the earlier 32-position sample. For a descriptive
effect-size view, only 8/576 Dominion positions have IH greater than 0.1
probability units; seven exceed 0.5. Window has 24/14 and Cookery 11/1 at those
cutoffs. These are post-hoc summaries, not significance or precision thresholds.
Near-zero global rank correlation can coexist with a sparse, strong tail.

Dominion token **131** (row 5, column 11 in the 24×24 grid) is especially useful
for review: IH is +0.9804, while native E is only at the **11.3rd percentile**
among 566 nonbasis states. The marked raw-image patch overlaps the theatre-name
lettering in the saved input. Window's strongest state has E at the 96.1st
percentile and Cookery's at the 93.1st, however. The defensible claim is that
low-E/high-response exceptions exist, not that high-response states always have
low E or that one threshold separates them.

The retained state is contextualized by the first 16 decoder blocks and CLIP;
all retained text has already processed the image too. Its marked raw-image
location is an index reference, not proof that only those pixels caused the
answer. Retention also removes competing context and changes sequence geometry.
Probability above the full reference is not proof that a singleton gives a
better generated answer. No per-intervention generated-answer test was run.

For Bottom, Bedroom and Soccer, removing late visual states barely decreases
the first-target probability. This does not establish image independence: the
earlier blocks already used the image. Median absolute IH is about 0.000149,
0.00000125 and 0.00164 respectively. Soccer's full-grid median is much larger
than its earlier sampled median; sample effect-size summaries also need limits.
Ties, FP16 resolution, high reference confidence and context leakage limit
strong interpretations of small score/rank differences.

## Input versus decoder depth

This table preserves the historical **whole-answer mean log-score zeroing**
measure at both stages. It must not be substituted for the probability-unit
IH comparison above.

| Target | Input E–I rho | Layer16 E–I rho | Input/layer E rho | Input/layer I rho |
|---|---:|---:|---:|---:|
| Window | −0.101 | −0.205 | +0.587 | +0.296 |
| Bottom | +0.093 | +0.017 | +0.587 | +0.060 |
| Bedroom | −0.070 | +0.014 | +0.719 | +0.001 |
| Soccer | +0.177 | +0.006 | +0.790 | −0.037 |
| Cookery school | +0.010 | −0.218 | +0.687 | +0.219 |
| Dominion | −0.035 | −0.070 | +0.728 | +0.055 |

Native E ranks retain moderate agreement across representations, while zeroing
sensitivity ranks agree much less. There is no consistent improvement of E–I
alignment at layer 16 in this panel. E is measured in different feature spaces,
and zeroing magnitudes/context differ; these are descriptive stage comparisons,
not a controlled causal effect of depth. Two stages do not establish a complete
information-horizon curve or a result across all decoder layers.

The two GQA questions share image `n578564`: “What is the pan in front of?” and
“Are the white paper towels in the bottom part or in the top of the photo?”
Their layer E rank agreement is 0.9999976. Their whole-answer zeroing-map
agreement is −0.330 at input and +0.003 at layer 16; IH first-piece map agreement
is +0.221. The same visual reconstruction ranking can accompany quite different
question-conditioned maps. Input feature hashes match, but the two decoder
visual hashes differ across different forward sequence shapes; within-question
repeats are exact. Targets, confidence and shapes also vary, so this does not
isolate wording as the only cause.

## Saved review and next steps

Open the [full-grid review and maps](../results/rq1-layer16-ih-full-review-v1-2026-10-17/full-review.html).
It includes all six questions, actual padded CLIP canvases, top-five retained
states, E/zero-first/IH-first heatmaps with hover values, basis labels and photo
footprints. Colors use separate linear ranges per panel: compare printed values
before comparing colors. The [effect CSV](../results/rq1-layer16-ih-full-review-v1-2026-10-17/effect-summary.csv),
[top-token CSV](../results/rq1-layer16-ih-full-review-v1-2026-10-17/top-retained-tokens.csv),
[stage table](../results/rq1-layer16-ih-full-review-v1-2026-10-17/stage-comparison.csv)
and [verification](../results/rq1-layer16-ih-full-review-v1-2026-10-17/verification.json)
are separate derived outputs. No native model or measurement code changed.

The browser connection was unavailable for a rendered HTML preview; structural,
image-link, marker-grid and numeric checks were performed locally. The original
saved Dominion/Cookery canvases were also inspected directly. An initial local
CSV export failed; that partial derived folder is marked `-incomplete-export`.
The completed review linked above was rebuilt in a new folder; raw ZIP/log were
never changed.

No further GPU run is required today. Review a few strong and weak examples,
consolidate RQ1 as **intervention-specific sensitivity**, and then plan RQ2's
controlled selection/merging ablations. Do not redesign selection to optimize
these six outcomes. Preserve the current reference/protocol and dataset split.
Additional layers, 2×2 tests or a full IH reproduction are optional extensions,
not automatic prerequisites for moving on.

## Reproduce offline

```bash
environments/editor/.venv/bin/python -X utf8 scripts/analyze_rq1_layer_ih_full.py \
  --archive research/results/rq1-layer16-ih-sixcases-v1-20261018T014830278963Z.zip \
  --log research/results/rq1-layer16-ih-sixcases-v1-20261018T014830278963Z.log \
  --expected-sha256 e396b7a1c0f0c554e2642ac4e0ac9a83d4f9c4b2a4ecdfbfee6e86bb5a845b37 \
  --output research/results/rq1-layer16-ih-full-review-new
environments/editor/.venv/bin/python -X utf8 -m unittest discover \
  -s research/tests -p 'test_rq1_layer_ih*review.py'
```

Eight CPU review tests passed: original calibration verification, corruption and
geometry/control rejection, exact-repeat drift/missing-position rejection,
embedded rerun prerequisite verification, full-grid completion and sparse-tail
recovery, and original-file preservation. No tensor loading or inference.
