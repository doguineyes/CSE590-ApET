# RQ1 layer-16 IH calibration review — 2026-10-17

**Later checkpoint:** the [full 576-position comparison](2026-10-17-rq1-layer-ih-full-review.md)
has completed. It finds strong retained states outside these 32 positions:
Window 94.1%, Cookery school 74.2%, Dominion 98.1% first-target probability.
The sample below is unchanged historical evidence; its low sampled maxima
do not establish that support must be distributed. No new GPU run is needed.

The new calibration passed. Its 1×1 released-style IH measurement does **not**
restore a consistently positive relationship between native E and signed
first-answer-token sensitivity in these six cases. The two sensitivity rankings
also differ. A stronger finding is the gap between all-visual and the sampled
retain-one conditions on TextVQA: tiny sampled singleton effects can coexist
with a large collective visual-state effect under this counterfactual.

The bounded full sweep is reasonable now. It can determine whether a few strong
singletons were missed by the 32-position sample; the calibration cannot rule
that out. No extra debugging run or measurement-code fix is needed first.
Finish this six-case comparison, review it, then return to RQ2.

## Archives and verification

Both supplied artifacts were copied byte-for-byte from Downloads into
`research/results/`, with the original Downloads copies retained:

- `rq1-layer16-ih-calibration-v1-20261018T002129035074Z.zip`: 58,791,797 bytes;
  SHA256 `84aa8999e9593f9cd2b436cbce794c205f088083ff2a104b511989e201a40e84`.
- Matching `.log`: 6,479 bytes;
  SHA256 `35085fa2913bcd4124a24fe8eb8d52e59ccaa1a84e9973005bb518c0aa40a27c`.

The UTC filename is October 18; the run and download occurred October 17 local
time. Runtime commit `dad8fec1ac95606b5645ff9be0b39e7a5ed534ec` was clean.
The original environment lock remains
`6458cb0ab526857c483b41c19cb87e63179507ebaf6e48f4e9020d18d261f3a2`.

The ZIP's 92 unique members pass CRC. All 49 evidence-file hashes, 11 saved
feature-file byte hashes, three accepted manifests, pinned method profiles,
eight monitored native files and original uv lock verify. The native ApET
reference and its previously registered two-GPU compatibility patch are
unchanged. No tensor was unpickled and no inference was performed locally.

All six fresh baseline answers are correct and target-aligned. All original
input/layer no-op, native, repeat and final reference controls have zero drift.
The new keep-all no-op, repeated remove-all, repeated retained-case, final
reference and boundary checks also have **zero drift**. Every retained sequence
has the expected original position map, all nonvisual states retained, and one
visual state; the null condition removes all 576 visual states at this boundary.
The existing source gate accepts this calibration for the expanded IH run.

All **384 zeroing measurements** exactly repeat the previous layer calibration,
including raw target-token scores, E and perturbation norms. Input feature
hashes and full layer-boundary hashes match too. Those 384 are reproduction
checks, not new independent evidence. The newly added measurements are **192 IH
retain-one scores**, plus their null/reference controls. Existing canonical
input-stage sample counts are unchanged. Summary HTML/CSV exported successfully;
there is no encoding/export error in this ZIP.

The [derived review](../results/rq1-layer16-ih-calibration-review-v1-2026-10-17/index.html),
[verification](../results/rq1-layer16-ih-calibration-review-v1-2026-10-17/verification.json)
and [effect table](../results/rq1-layer16-ih-calibration-review-v1-2026-10-17/effect-summary.csv)
are separate saved-data outputs. Originals and source scores remain unchanged.
The separate prerequisite tiny-test console output is not part of this log;
the archive establishes the successful pretrained measurement controls.

## Comparable correlations

Both columns below use **first frozen reference-token probability differences**
at layer 16, on the same 32 randomly sampled positions. The historical
whole-answer log-score zeroing results are preserved separately. In particular,
Cookery school's first-token zeroing rho is −0.069, while its historical
whole-answer rho is −0.273; this is a metric difference, not new numerical drift.

| Question | E vs zero-first | E vs IH-first | Zero-first vs IH-first |
|---|---:|---:|---:|
| GQA Window | −0.183 | −0.491 | +0.266 |
| GQA Bottom | −0.216 | −0.228 | +0.074 |
| MMStar Bedroom | +0.181 | −0.162 | −0.142 |
| MMStar Soccer | −0.120 | +0.059 | −0.063 |
| TextVQA Cookery school | −0.069 | −0.280 | +0.106 |
| TextVQA Dominion | +0.191 | −0.059 | +0.443 |

Five IH E–I correlations are negative and the remaining one is close to zero.
Do not summarize this as every correlation being zero: Window has a moderately
negative association. FPS-source exclusion changes Bedroom's IH rho to −0.239
and its zero/IH agreement to −0.198; the other five comparisons are unchanged.
The two methods show limited or reversed rank agreement here, rather than a
common token ranking. No population sign test or pooled-token significance is
claimed for six curated, spatially related, partially same-image questions.

For the GQA pair, native layer E rank agreement over all 576 positions remains
0.9999976, while the sampled IH-map agreement across questions is +0.421
(zero-first agreement is approximately +0.192). These are question-conditioned
counterfactual measurements, not a pure causal isolation of wording; targets,
answer confidence and full-forward shapes also differ.

## The all-visual / sampled-singleton gap

These are probabilities for the **first fixed gold tokenizer piece**, not
semantic answer accuracy. Best retain-one is only over the 32 sampled positions.

| Question | Full visual states | No visual states at layer 16 | Best sampled retain-one |
|---|---:|---:|---:|
| Window | 27.097% | 6.994% | 12.123% |
| Bottom | 98.849% | 99.035% | 99.093% |
| Bedroom / B | 63.076% | 62.612% | 62.628% |
| Soccer / B | 79.718% | 80.017% | 80.066% |
| Cookery school / `Cook` piece | 70.205% | 14.311% | 15.237% |
| Dominion / `Domin` piece | **94.716%** | **0.0608%** | **0.0768%** |

Dominion is the clearest example. Removing the visual block produces a very
large loss of first-target probability, yet retaining any one of the 32 sampled
states recovers little. Cookery school and Window also show a substantial
full-to-none difference that the sampled best singleton does not recover.
This directly limits the inference that small singleton scores imply a small
collective contribution of visual states at that boundary.

Possible explanations include distributed/context-dependent support, several
states needing to work together, or a small number of strong **unsampled**
states. Sequence shortening/attention changes can also contribute. These data
do not isolate a specific synergy mechanism or prove that no singleton can
suffice. A full 576-position sweep is useful precisely to test the missed-strong-
singleton possibility. No sampled-score sum is used to predict a group effect.

Bottom and Soccer show little decrease of first-target probability when the
late visual states are removed. This does not establish that they can be solved
without seeing the image: the first 16 blocks and retained text states already
processed it. Nor do these probabilities supply per-intervention free generated
answers, full-answer correctness or a measured information-horizon curve.

## Signed scores, small effects and scope

IH's retain-one-minus-null score can be negative. For example, 22/32 Window
retentions and 25/32 Cookery school retentions lower first-target probability
relative to that layer's null condition. That is a counterfactual effect, not
negative Shannon information or an intrinsically harmful visible patch.

Magnitude is a different question from support direction. As a **post-hoc
auxiliary** summary, E vs absolute IH sensitivity is +0.650 for Window, +0.651
for Bottom and +0.521 for Dominion, but −0.195 for Bedroom and −0.039 for Soccer.
Thus we should not claim E is unrelated to every kind of output response.
Association with response magnitude does not establish positive answer support;
sign, null context, confidence and effect size must remain visible.

Within a question, subtracting the same null probability sets the signed score's
origin but cannot itself create its E–I ranking. That rank is the rank of
retain-one probabilities, with ties respected. The choice of retained context
still changes the function being measured; a shared baseline subtraction alone
does not explain away the observed negative associations.

Median absolute IH effects are only about 0.00000125 for Bedroom and 0.00000149
for Soccer in probability units. Bottom has 23 distinct sampled scores, Bedroom
30, Soccer 26, with tied ranks handled explicitly. Reproducible references do
not establish an arbitrary precision noise floor for such small effects.
All primary ranks remain descriptive. Basis and padding labels are retained in
the original comparison CSV; no locations or outcomes are silently excluded.

This is the documented **1×1, physical-deletion IH adaptation** in the original
ApET/SDPA environment. It differs from the released example's 2×2 setting and
from our shape-preserving zeroing in both context and operator. Differences do
not identify which design choice causes them. It neither disproves IH nor shows
ApET ineffective. The supported result is that native reconstruction E is not
a consistently positive predictor of either sampled operational sensitivity in
this panel, and the measured rankings depend on the counterfactual.

## Next run

The new calibration and measurement-source gate both pass. In the existing
notebook, reuse the helper and the new calibration path:

```python
layer_ih_full_dir = run_layer(
    tokens=576,
    calibration=layer_ih_calibration_dir,
    include_ih=True,
)
```

If the updated notebook named it `layer_calibration_dir`, use that variable
instead. Both should identify the newly passed
`/kaggle/working/rq1-layer16-ih-calibration-v1-20261018T002129035074Z` directory.
The old zero-only calibration is not the IH prerequisite.

This is the same six questions, 576 positions, 6,912 zeroing and 3,456 IH scores
(10,368 total), with one model load. The calibration took **456.52 seconds
(7.61 minutes)**; IH phase 87.04 seconds, zeroing phase 267.81 seconds. Timed
interventions extrapolate to 83.87 minutes before fixed controls, downloads,
growing exports and archiving. Budget **90–120 minutes** on the same T4 ×2.
The IH-phase memory peaks were 8.13/7.21 GiB; these are not whole-run peaks or
benchmark comparisons. Download the new ZIP and log. No code pull, new dataset,
environment install or repeated calibration is required in the current live
session for this run.

After the full comparison, inspect complete effect maps and the sampled/full
agreement, then move to RQ2. More layers, region sizes or counterfactuals are not
automatically required.

## Saved-data reproduction

```bash
environments/editor/.venv/bin/python -X utf8 scripts/analyze_rq1_layer_ih.py \
  --archive research/results/rq1-layer16-ih-calibration-v1-20261018T002129035074Z.zip \
  --log research/results/rq1-layer16-ih-calibration-v1-20261018T002129035074Z.log \
  --expected-sha256 84aa8999e9593f9cd2b436cbce794c205f088083ff2a104b511989e201a40e84 \
  --output research/results/rq1-layer16-ih-review-new
environments/editor/.venv/bin/python -X utf8 research/tests/test_rq1_layer_ih_review.py
```

Five CPU review tests check score/probability corruption, retained-position
geometry/completion, control drift/boundary failure, overwrite/hash rejection,
and a complete review with original-file preservation. No GPU rerun was used.
