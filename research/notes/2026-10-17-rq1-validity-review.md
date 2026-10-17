# RQ1 validity diagnostics review — 2026-10-17

The complete original ZIP and console log are archived in `research/results/`:
`rq1-validity-all-20261017T204446432120Z.zip` and `.log`. No inference rerun is
needed. The [local reconstructed review](../results/rq1-validity-review-v1-2026-10-17/index.html)
and [verification record](../results/rq1-validity-review-v1-2026-10-17/verification.json)
are separate derived artifacts; the originals are unchanged.

## Integrity and controls

- Passed execution: 72 option/image trials + six separate reference repeats,
  then six questions × 32 positions × three methods = 576 patch/method scores.
- Runtime: 523.47 seconds (8.72 minutes), including loading/downloading overhead.
  This instrumented diagnostic is not a generation-speed benchmark.
- All 32 evidence-file hashes, five saved feature-file byte hashes, accepted
  manifests, native reference sources and original uv lock verified. Feature
  tensors were not unpickled during this CPU review.
- Fresh baseline answers are correct and match fixed target token IDs. All
  repeated/no-op/native/full-forward/per-method-sham/final reference comparisons
  have zero token-score drift. TextVQA's known later cached-decode differences
  remain diagnostic under the previously documented full-teacher-forced policy.
- Each question exactly reproduces the archived CLIP/projector hashes, baseline
  score, native E and selected zeroing I values. All six free-answer original
  MMStar repeats also match their original-order responses.
- The exported HTML was empty because the process default encoding was ASCII
  and the writer used an en dash. Raw JSON/JSONL, CSV, features and logs are
  intact. Explicit UTF-8 is now used for validity/layer HTML and shared CSV
  exports. The derived HTML is rebuilt locally; no GPU work is required.

ZIP SHA256: `259441e1a772fc370aa07fb51d577da2d37f50d755b55e85b83b395b472b8901`.
Log SHA256: `266ec8ffd2b4cd586a057949609316020f86d849a2f889eaa0069f573d6bf3c7`.
Run commit: `1ab2879e9b3f192ba5332b32f25946e61d6b05c4`; original lock:
`6458cb0ab526857c483b41c19cb87e63179507ebaf6e48f4e9020d18d261f3a2`.

## Image/option reliance

Counts below score the correct **answer content** after four cyclic option
orders. Repeats are excluded. These are selected diagnostic cases, not a
benchmark-accuracy estimate or four independent questions per image.

| MMStar ID / task | Real image | Question only | Neutral blank image |
|---|---:|---:|---:|
| 1014 — count bars below 6 | 1/4 | 1/4 | 1/4 |
| 1030 — geometry angle | 1/4 | 1/4 | 1/4 |
| 1188 — minimum age group in plot | 2/4 | 1/4 | 2/4 |
| 776 — dark-blue bar difference | 1/4 | 1/4 | 1/4 |
| 1372 — sea-coral food-web relation | 1/4 | 2/4 | 3/4 |
| 136 — soccer control | **4/4** | 1/4 | 1/4 |

Cases 1014, 1030 and 776 always answer the displayed letter **A** in all twelve
conditions/orders. Their original-order correctness therefore does not establish
that LLaVA solved the visual problem. Case 1188 has identical displayed answers
for real and blank images across the four orders. Case 1372 is order/condition
sensitive and is not more accurate with the real image in this small check.
These are flags against using the five diagrams as strong evidence of visually
grounded competence. They do not prove a specific internal shortcut or that an
image has no influence on logits. Keep all original samples/results and report
the flags openly; do not silently change the frozen cohort or benchmark scores.

The soccer control follows **Soccer** through B, A, D, C with the real image,
while the image-free/blank conditions always answer A. It provides much clearer
evidence of image-dependent, option-content-consistent behavior in this panel.
The bedroom layer-panel case was not part of these image/option checks; no
equivalent grounding claim is established for it here.

## Does I survive other interventions?

The random stratum contains 16 positions per question, independent of previous
E/I. The additional high-I and low-E/high-I strata each contain eight positions
chosen from archived zeroing scores; keep those post-hoc strata separate. This
table uses **only the random stratum**:

| Question | rho(E, I_zero) | rho(I_zero, I_half) | rho(I_zero, I_neighbor) |
|---|---:|---:|---:|
| GQA Window | +0.003 | +0.085 | +0.206 |
| GQA Bottom | −0.024 | +0.600 | +0.260 |
| MMStar Bedroom | +0.029 | +0.296 | +0.115 |
| MMStar Soccer | +0.326 | +0.165 | +0.168 |
| TextVQA Cookery school | +0.203 | −0.088 | −0.406 |
| TextVQA Dominion | −0.129 | +0.303 | +0.385 |

The old zeroing measurement is reproducible, but its rankings do not consistently
carry over to the alternatives. Several signs change and many alternative scores
are much smaller. Half-strength is a smaller intervention along the same direction,
and neighbor replacement may retain evidence. Different effects therefore do not
by themselves invalidate zeroing or certify either alternative as the truth.
Small scores should not be overinterpreted, even with exact reference repeats.

There is some concrete persistence: GQA Window patch 211 has I_zero **+0.30548**,
I_neighbor **+0.23267**, and I_half **−0.12530**. This is the previously illustrated
low-E, non-basis, mostly-photo boundary patch. Its positive support survives the
neighbor replacement but not halving. This is more precise than saying the patch
has one perturbation-independent importance value. Across its eight low-E/high-I
positions, the Window question has median zeroing I +0.10510, half −0.00717 and
neighbor −0.00243; the original selected-group effect does not uniformly persist.

For Cookery school, the eight previously high-zero-I positions retain median
neighbor I +0.02332 versus zero I +0.07695, with within-stratum rank correlation
+0.690. Its low-E/high-I stratum has median neighbor I only +0.000662 versus
zero I +0.04561 and rank correlation −0.071. Report both the surviving and weak
patterns, with their post-hoc selection stated. The full
[method/stratum CSV](../results/rq1-validity-review-v1-2026-10-17/method-summary.csv)
records effect magnitude, sign, perturbation norm and correlations for every case.

The supported statement remains narrow: **native input reconstruction error is
not a consistently strong positive ranking of singleton-zeroing reference-answer
sensitivity in these selected cases.** We now also know that I depends materially
on intervention choice and that several baseline-correct MMStar diagrams have
weak grounding evidence. Neither finding establishes that ApET is ineffective,
that token zeroing measures ground-truth relevance, or that all chart cases fail.

## Interpretation recorded after discussion

Our operational definition should remain explicit:

\[
I_i^{(T)} = S(X) - S(T_i(X)),
\]

where S is the fixed reference-answer mean token log probability without EOS,
and T specifies the representation, position, replacement and strength. Zeroing,
halving and neighbor replacement define different counterfactuals. The recorded
quantity is **answer sensitivity under a specified intervention**. Calling it an
intrinsic, intervention-independent token importance or ground-truth answer
contribution would exceed our evidence. This is our project-owned measurement;
we have not shown that an inspiring paper's exact experimental procedure is wrong.

Keep two findings distinct. **Reproducibility passed** for the recorded cases:
reference controls and repeated zeroing effects match exactly. **Interpretation
depends on the intervention**: changes in rankings and signs limit the claim that
one sensitivity map identifies task-critical evidence under other manipulations.
This is not evidence that today's results are numerically random. Different
strengths and the possibility that replacement retains information also mean
disagreement alone cannot invalidate any one intervention.

Possible explanations to investigate, rather than established causes, include
unusual states introduced by zeroing, nonlinear responses, overlapping evidence
across contextualized tokens, and dependence on the chosen answer-score metric.
A small singleton effect can coexist with a large joint effect. Conversely, a
large response to an artificial vector change need not establish that a specific
visible object was essential. A changed reference score need not change the
generated answer or its semantic correctness. Related methodological limitations
are discussed in [Zhang and Nanda](https://arxiv.org/abs/2309.16042) and
[Heimersheim and Nanda](https://arxiv.org/abs/2404.15255); those papers do not
establish the mechanism behind our particular VLM results.

Discussion raised the possibility that consistent effects of token manipulation
will be difficult to find. We should distinguish **continuity**, **monotonicity**,
and **consistency across intervention choices**. We have not measured a dense
strength-response curve or established mathematical discontinuity. Window patch
211 does show a nonmonotonic sampled zeroing-strength response: I is 0 at the
unchanged reference, −0.12530 at half strength, and +0.30548 at complete zeroing.
That observation is specific to this patch/question/score. It does not establish
that every token has such behavior or that no useful regularity can be found.
The useful reporting goal is to identify conditions with stable, weak, or reversed
effects, preserving counterexamples and the scope of each observation.

## Connection to selection, merging and FPS bases

The same issue matters for the later selection/merging study. Native ApET changes
many representations together, replaces selected slots with FPS basis features,
merges contributions from other sources, and shortens the sequence. Singleton
zeroing holds the other input states, token count and positions fixed. We cannot
predict the combined native effect by simply adding singleton I values.

A source absent from direct retention can still contribute through merging or
as a basis feature. A retained slot may contain a replacement basis source, so
slot IDs and original source IDs must remain distinguishable. Choosing different
FPS bases can also change E, selection and merge assignments together. Any basis
comparison must record these coupled changes rather than attribute the result
to a basis choice in isolation. Low E–I correlation by itself does not identify
whether selection, basis handling, merging or the sensitivity proxy caused an
answer failure.

For future controlled comparisons, record the following before drawing a mechanism
claim; these are planning commitments, not additional completed experiments:

- Compare merge variants with the same source features, basis mapping, selected
  indices and total output budget, documenting any unavoidable difference.
- For a selection or basis variant, fix the other algorithm settings and report
  the resulting changes in selected sources, replacement slots and merge targets.
  Match total output budgets; changing basis count also changes the allocation
  between direct retained targets and basis slots.
- Preserve raw answers, correctness/preservation, fixed-reference scores, seeds,
  per-question and same-image outcomes, and native feature/selection/merge records.
  Report gains, losses and weak/reversed effects rather than only favorable cases.
- Keep exploratory diagnostic cases and later broader or reserved-set confirmation
  distinct. Do not choose a new I definition because it correlates better with E.

The project can contribute evidence about where reconstruction-based compression
and answer sensitivity align or diverge, including the limits of the measurement.
We do not need to demonstrate one universally smooth or monotonic token-importance
law, or conclude that compression/intervention research is generally unreliable.

## Next action

Proceed with the already implemented **six-question × 32-position input/layer-16
calibration**. That panel was fixed before these validity outcomes and uses
random positions independent of E/I. It includes the soccer control and two GQA
same-image questions, rather than the five flagged diagrams. Keep it unchanged.
Label its scores as stage-specific zeroing sensitivity. Do not start the 576-position
sweep until layer controls have been reviewed. There is no need for another
large baseline, another full input sweep, or a new environment/dataset bundle.

Rebuild the derived review with the local editor environment (uses the native
TextVQA scorer's existing `tqdm` dependency; no Torch/GPU required):

```bash
environments/editor/.venv/bin/python scripts/analyze_rq1_validity.py \
  --archive research/results/rq1-validity-all-20261017T204446432120Z.zip \
  --log research/results/rq1-validity-all-20261017T204446432120Z.log \
  --output research/results/rq1-validity-review-new
```
