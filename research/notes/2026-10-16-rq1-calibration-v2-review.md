# RQ1 calibration v2 passed — 2026-10-16

The corrected two-question × 32-patch calibration passed. Proceed to the frozen
12-question / six-image GQA input-stage pilot with all 576 singleton interventions
per question, using the same v2 measurement code and accepted inputs. No additional
inference-code, environment or dataset changes are needed for that run.

## Evidence and checks

Original ZIP copied unchanged from Downloads:
`research/results/rq1-gqa-calibration-v2-20261016T142220402681Z.zip`, with extracted
folder of the same name. All 37 ZIP entries passed CRC validation. All recorded
evidence hashes and the saved feature-tensor file hash were verified. Token tables
contain exactly 576 unique indices per question: 32 measured and 544 I=null.
Sensitivity arithmetic and Spearman values were independently recomputed.

Runtime commit `a3842946ccd91c35805c161dca1f23729000d325`, clean working tree.
Protocol `rq1-input-zero-v2`, original lock SHA256
`6458cb0ab526857c483b41c19cb87e63179507ebaf6e48f4e9020d18d261f3a2`.
Native source hashes match the original passed reference. Both questions share
the same image `n200692`, features, error basis and patch execution order as v1.

| Question | Fixed measured target | E–I Spearman | Positive / negative I | I range (nats per target token) |
|---|---|---:|---:|---:|
| `201636988`: cake vs plate color | `No` | -0.08542 | 11 / 21 | -0.02243 to +0.02301 |
| `201637316`: plate lies on what? | `Stove` | +0.04142 | 4 / 28 | -0.03595 to +0.00637 |

Both fresh baseline answers were correct. Both fixed target sequences matched
native generated prefixes. Unchanged-reference repeats, clone-only no-op,
cached/native multimodal parity and final reference differences were exactly
zero. Generation-vs-teacher-forcing target log-probability differences were zero
for `No` and at most 0.00000472 nats for `Stove`, below the 0.001-nat threshold.
The v2 targets' reference sequence probabilities reproduce the preceding audit:
0.515563 and 0.529833, respectively (without EOS).
Both exported PNG figures were visually inspected. The cake-question scatter
contains visibly tied/stepped sensitivity levels under the existing FP16 policy;
interpret fine token rankings cautiously even though reference repeats are stable.

Wall time with checkpoint reuse was 52.15 seconds, including model loading and
plot creation. Question measurements took 20.38/17.16 seconds. Mean measured
singleton-forward time was 0.5083 seconds, giving a rough 58.6-minute extrapolation
for 12 × 576 interventions before extra control/startup/plot/I/O costs. Plan for
about one hour; this is not a guaranteed duration or a speed benchmark.
Peak allocated GPU memory was 7.43/7.27 GiB.

## What the pilot can and cannot suggest

The two E–I correlations remain near zero on 32 sampled patches, but this is one
image and two selected questions, not evidence of a population-level absence of
association. The cross-question sensitivity rank correlation changed from v1's
+0.9363 to v2's **-0.8430**, despite identical image features and patch indices.
This demonstrates sensitivity to target rendering in this example and motivates
question-conditioned analysis; it does not establish the general RQ1 relationship.

Patch 290 has E≈0.0000173 and I=+0.01528 for the cake question; patch 98 has
E≈0.0000630 and I=+0.00762. Both are native FPS **basis source** patches. Native
diagnostics map them into replacement slots 353 and 325, respectively. They are
therefore not clean examples of ApET discarding low-error but important content.
Future interpretation must distinguish basis sources, directly retained original
features, replacement slots and merged/dropped-source paths. These role mappings
are already preserved in `features/*.json`; no new inference is needed to recover
them. Low-error non-basis examples 11 and 549 have positive I for `Stove`, but
singleton zeroing on full-context embeddings is still not native merging/deletion.

Negative I means that this zeroing intervention made the fixed reference string
more probable; it does not by itself mean the generated answer became correct,
or that the corresponding image patch is globally harmful. Keep the proxy and
the curated/image-dependent sampling limits explicit.

## Next run

In the existing live notebook, using its already defined helper:

```python
gqa_pilot_dir = run_experiment(
    "rq1-gqa-input-pilot-v2", "gqa/manifest.json",
    ["--questions", "12", "--tokens", "576"],
)
```

Run only this new cell, then download its ZIP and log. The environment, v1 input
ZIP and checkpoint cache are reusable. The helper creates a fresh timestamped
output directory, leaving all earlier evidence intact. Native alignment/numeric
checks still run for every question and stop on a failure; return the saved ZIP
and log if one occurs. After the GQA pilot, inspect per-image/question/patch-role
patterns, then continue with MMStar confirmation and the TextVQA baseline.
