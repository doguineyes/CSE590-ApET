# RQ1 input IH calibration review — 2026-10-18

The calibration passed. **Proceed to the fixed 30-question × 576-position full
input IH sweep.** All archived reuse checks are exact and the new retention
controls have zero drift. No measurement fix, reference replacement, setup
rerun or GitHub pull is needed in the same live Kaggle notebook.

The sampled results differ across questions and interventions. They do not yet
establish a dataset-level E–IH relationship. In particular, Dominion's positive
sample correlation concerns tiny probability changes; all sampled-singleton
maxima remain limited to 32 positions. The preceding layer full sweep already
showed that this sample can miss strong states.

## Preserved files and provenance

Both supplied files are copied byte-for-byte into `research/results/`; the
Downloads originals remain intact:

- `rq1-input-ih-calibration-v1-20261018T135403155797Z.zip`: **3,238,120 bytes**;
  SHA256 `05d1fc5c78682d94ab645a386f1dde7eb23f8e51fa72f3e70714dbf22c2a5a8c`.
- Matching `.log`: **3,170 bytes**;
  SHA256 `a7e762fb09d01d3cbeaa5f133459a79a105bf0c261799034c04d642c7d120c61`.

Runtime commit `967aa230b2df8bd9aec440d4baec0ed2e9682a34` was clean. The
121 unique ZIP members pass CRC. All **95 evidence-file hashes**, **5 fresh
feature-metadata hashes**, question hashes, three accepted manifests, embedded
reference index/files, current measurement/native sources and original lock
verify. No native ApET, dependency or measurement source changed during review.

Independent CPU checks recompute all **192 new IH measurements**, including raw
target-token score arithmetic, shortened sequence lengths, retained original
positions, first-piece probabilities and per-question correlations. All six
fresh baseline answers re-score correctly and match the frozen target prefix.
Every fresh CLIP/projector hash, seed, native E/basis, full target-token score
and **18 zeroing audit repeats** matches its archived reference exactly.
The 192 paired archived zeroing scores are reused evidence, not new observations.

The keep-all no-op, repeated null, first retained-case repeat, final reference
and unchanged-input controls all pass with **zero drift**. These repeats do not
prove arbitrary numerical precision for every tiny counterfactual. The full-run
source gate independently accepts this exact calibration and all six detailed
records; its report SHA256 is
`ed6dcd165d6e2fd1d93c03df7621a9caaaced8005a395c11584f1e758df4d7ab`.

The supplied log starts at pretrained input validation/loading. Separate tiny
native-test console output from the setup cell is not included; the archive
does establish successful pretrained measurement controls. No result or required
run log supplied in this turn is missing. No local inference or tensor loading
was used to review it.

## Matched scores on the sampled positions

Both I columns below are **first frozen gold tokenizer-piece probability
differences** at the input stage. E is native pre-projector CLIP reconstruction
error; the two interventions act on corresponding projected LLaVA embeddings.
Historical whole-answer zeroing log scores remain preserved separately.

| Question / target | E–zero-first rho | E–IH-first rho | Zero-first–IH-first rho |
|---|---:|---:|---:|
| GQA 201636988 / No | −0.085 | **−0.618** | −0.077 |
| GQA 201637316 / Stove | +0.042 | +0.071 | **−0.420** |
| MMStar 313 / D (4 buildings) | +0.284 | **−0.408** | −0.330 |
| MMStar 617 / A (curved beak) | +0.429 | +0.272 | +0.332 |
| TextVQA 37533 / Brolio | −0.161 | −0.237 | +0.205 |
| TextVQA 35670 / Dominion | +0.126 | **+0.536** | +0.132 |

The same token positions can have materially different rankings under zeroing
and retention. MMStar 313 changes from positive E alignment under zeroing to
negative alignment under retention, while MMStar 617 remains positive under
both. Neither method is thereby established as the correct token-importance
measure. They change both retained context and the zeroing/deletion operator.
MMStar 617 retains its previously recorded quality flag; it is not silently
excluded or promoted to a clean population example.

Removing sampled FPS basis sources leaves E–IH rhos at −0.630, +0.011, −0.351,
+0.272, −0.177 and +0.536 in the same order. The main negative GQA association
is not an artifact of its two sampled basis states. This is a descriptive
sensitivity check, not a different primary cohort or a significance test.

The GQA pair shares identical image features and reconstruction errors, but
asks “Is the color of the cake different than the plate?” and “What does the
plate lie on top of?” Its sampled IH-map rank agreement is **−0.250**.
Question/target-conditioned responses can differ despite the same visual E.
The targets and confidence levels also differ; do not attribute every difference
solely to question wording or pool the 32 positions as independent samples.

## Effect sizes and interpretation

All percentages below refer to the first fixed gold tokenizer piece, not
generated-answer correctness or the entire answer sequence. Stove starts with
`▁Sto`, Brolio with `▁Bro`, and Dominion with `▁Domin`.

| Target | Full image states | No visual input states | Best of 32 retained states |
|---|---:|---:|---:|
| No | 51.556% | 53.114% | 53.503% |
| Stove | 53.490% | 0.0653% | 11.097% |
| D / 4 buildings | 30.854% | 11.246% | 13.541% |
| A / curved beak | 45.159% | 42.879% | 48.002% |
| Brolio | 81.160% | 0.001775% | 0.002865% |
| Dominion | 94.716% | 0.001140% | 0.001500% |

For Dominion, median absolute IH is only **8.20×10⁻⁷ probability units**, and
the largest sampled positive effect is **3.60×10⁻⁶**. Its +0.536 E–IH correlation
is therefore a relative ranking of very small effects, not strong evidence that
high-E states recover the answer. Brolio also has tiny sampled effects: median
absolute IH **1.74×10⁻⁶**. Stable reference controls do not establish a universal
noise floor at these magnitudes. The full run's exact calibration-repeat checks
will add repeatability evidence without redefining the measure.

Stove has a larger sampled effect: its best retained state raises first-piece
probability from 0.000653 to 0.110966, while its E–IH rank correlation is only
+0.071. A sparse strong response can coexist with weak global rank agreement.
The full 576 positions are needed before describing which strong states exist.
No sampled maximum or sum establishes distributed support, synergy, or absence
of a sufficient singleton.

For No, the null probability slightly exceeds the full reference and 27/32
retentions reduce that probability. E vs **absolute** IH response is +0.606,
despite E vs signed IH being −0.618. Magnitude of response and support direction
are separate questions. The high No/A null confidences also caution against
treating every correct baseline as demonstrated visual grounding.

At this boundary no decoder block has processed the image, but the null retains
original positional gaps and is not an ordinary text-only prompt. Each retained
CLIP feature is already contextualized by the vision encoder. Layer-16 nulls
and retained text have a different context history; keep depth labels explicit.
The supported interpretation remains **intervention-specific sensitivity**.

## Full run and expected time

The calibration took **156.60 seconds = 2.61 minutes**, including checkpoint
downloads/loading and review export. The six measured question sections total
45.25 seconds; their 192 timed retain-one forwards total **16.28 seconds**.
Means are about **0.0835 s/position GQA**, **0.0885 MMStar**, and **0.0824 TextVQA**.

Weighting those rates by the full 12/12/6 panel and adding the measured question
controls gives a simple **27.01-minute** projection. It excludes fresh
generation/encoding, model loading, periodic JSON export and final packaging.
Other questions are longer: MMStar's shortened lengths range 90–148 versus
92/100 in calibration. Allow roughly **30–45 minutes in the same warm session**;
this is a provisional execution budget, not a measured full-run time or speed
benchmark. Input retention is cheaper than late retention because all decoder
blocks process the shortened visual sequence.

In the same notebook, use the already-defined helper and directory:

```python
input_ih_full_dir = run_input_ih(
    tokens=576,
    calibration=input_ih_calibration_dir,
)
```

If the variable was lost but the directory still exists, use:

```python
input_ih_full_dir = run_input_ih(
    tokens=576,
    calibration="/kaggle/working/rq1-input-ih-calibration-v1-20261018T135403155797Z",
)
```

Do not rerun setup, pull Git, change targets or replace cases. The full command
loads the model once, measures 17,280 retained states, audits 90 zeroing scores,
joins the 17,280 original E/zero pairs and checks all 192 calibration retentions
exactly. The remaining 24 questions still have to pass their fresh reuse gates;
a failure preserves evidence and stops. Download the full ZIP **and log**.

## Saved-data reproduction

The [verified review](../results/rq1-input-ih-calibration-review-v1-2026-10-18/index.html),
[effect table](../results/rq1-input-ih-calibration-review-v1-2026-10-18/effect-summary.csv),
[verification](../results/rq1-input-ih-calibration-review-v1-2026-10-18/verification.json)
and [timing data](../results/rq1-input-ih-calibration-review-v1-2026-10-18/timing.json)
are separate from the original ZIP/log. The initial derived review is preserved
in an adjacent `-initial` folder; the linked review adds fresh-answer rescoring
and explicit question/image identity checks without changing measurements.

```bash
environments/editor/.venv/bin/python -X utf8 scripts/analyze_rq1_input_ih.py \
  --archive research/results/rq1-input-ih-calibration-v1-20261018T135403155797Z.zip \
  --log research/results/rq1-input-ih-calibration-v1-20261018T135403155797Z.log \
  --expected-sha256 05d1fc5c78682d94ab645a386f1dde7eb23f8e51fa72f3e70714dbf22c2a5a8c \
  --output research/results/rq1-input-ih-review-new
environments/editor/.venv/bin/python -X utf8 research/tests/test_rq1_input_ih_review.py
```

Three CPU review tests passed, covering complete real-archive verification and
source-file preservation, overwrite/hash rejection, and rejection of a mismatched
measurement source before extraction. No GPU rerun or local package install.
