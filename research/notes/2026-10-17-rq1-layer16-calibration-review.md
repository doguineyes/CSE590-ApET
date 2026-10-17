# RQ1 input / layer-16 calibration review — 2026-10-17

The six-question, 32-position calibration passed. A bounded full sweep of the
**same six questions at 576 positions per stage** is now reasonable. This is
6,912 intervention scores, not a full-dataset accuracy run or a layer sweep over
the whole decoder. Finish this comparison, analyze it, then return to RQ2.

## Saved artifacts and integrity

Original files were copied byte-for-byte from Downloads into `research/results/`:

- `rq1-layer16-calibration-v1-20261017T230335470061Z.zip`: 58,701,880 bytes;
  SHA256 `8758b31819ef9f73992dcd46776247b7015107e583ca2615f691b64dd8249fd3`.
- Matching `.log`: 4,559 bytes;
  SHA256 `eccab4baff49b0717f842aaae172c71f57c5e93522179c629ddf259b9fe2596a`.

Both supplied artifacts are present. The ZIP has 78 unique members and passes CRC.
The [derived review](../results/rq1-layer16-calibration-review-v1-2026-10-17-complete/index.html)
and [verification record](../results/rq1-layer16-calibration-review-v1-2026-10-17-complete/verification.json)
are separate from those originals. No inference was rerun and no saved tensors
were unpickled during this review. Analysis source snapshots and hashes are
retained in the derived folder, alongside reconstructed CSVs and raw copies.

Runtime commit: `6dc0784d132d882e9d6cb6c8a6a3d8ac5a74d981`, with a clean checkout.
Original lock: `6458cb0ab526857c483b41c19cb87e63179507ebaf6e48f4e9020d18d261f3a2`.
The 39 archived evidence files, 11 feature-file byte hashes, three accepted
manifests, eight monitored native files and original lock all verify. Native
ApET is unchanged from the established reference, including its previously
registered two-GPU compatibility patch. These interventions and scoring remain
project-owned measurement code.

## Controls and repeatability

- All six fresh baseline answers are correct and match the frozen target IDs.
- All input reference/repeat/no-op/native/final and decoder observation/repeat/
  boundary-no-op/final comparisons have **zero reference-token score drift**.
- Full boundary states repeat exactly within each question before and after the
  intervention loop. Native E observation preserves the captured states, restores
  RNG and discards its compressed output.
- All 384 expected patch/stage measurements are present, in the fixed random
  order. Both stages use the same positions, answer targets and sequence mapping.
- All 192 fresh input sensitivities, all full input E arrays, CLIP/projector
  hashes and full reference scores exactly match the earlier canonical runs.
- Cookery school's cached-generation later-token gap remains 0.0054963;
  Dominion's remains 0.00008458. These are the already documented diagnostic
  differences: first-token prefill and full-teacher-forced controls pass exactly.

The separate notebook prerequisite-test output is not in this measurement log;
the ZIP establishes successful pretrained measurement controls, rather than an
archived transcript of every tiny native test.

## Exploratory stage comparison

Whole-answer signed sensitivity is still the primary score. The following
Spearman correlations use **all 32 randomly chosen positions per question**;
they are not the earlier full-grid statistics.

| Question / fixed target | Input E–I rho | Layer-16 E–I rho | Input/layer I rank agreement |
|---|---:|---:|---:|
| GQA 202218829 — Window | −0.033 | −0.183 | +0.186 |
| GQA 202218593 — Bottom | +0.092 | −0.222 | +0.038 |
| MMStar 895 — Bedroom / B | −0.120 | +0.181 | −0.070 |
| MMStar 136 — Soccer / B | +0.402 | −0.120 | +0.058 |
| TextVQA 35962 — Cookery school | +0.237 | −0.273 | +0.273 |
| TextVQA 35670 — Dominion | +0.047 | +0.026 | −0.448 |

Stage-specific FPS exclusion changes Bedroom's layer correlation to +0.118 and
Cookery school's input correlation to +0.165; the CSV also gives common-nonbasis
comparisons. No cases or basis/padding positions were silently removed. E rank
agreement across stages is +0.561 to +0.851, while I rank agreement is −0.448 to
+0.273 in this small panel. Thus reconstruction rankings persist more than the
zeroing-sensitivity rankings here; this is a descriptive observation, not a
population-level statistical conclusion.

Median absolute I is smaller at layer 16 in each case, despite larger median
zeroing norms in every case. For example, Soccer changes from 0.07145 to 0.000466,
and Bedroom from 0.01709 to 0.000001818. This does **not** establish that late
visual states are irrelevant, that compression is safe, or that influence simply
vanishes with depth. Other states already contain image information, zeroing
acts in different spaces, and several late effects are very small. Exact
unperturbed repeats do not establish arbitrary precision for perturbed effects.
No per-patch generated-answer changes were measured.

Saved first-answer-token scoring remains available. Cookery school's layer E–I
rho changes from −0.273 (whole answer) to −0.069 (first piece), and Dominion from
+0.026 to +0.191. Single-token MMStar scores agree by construction. Do not treat
the first subword as the entire semantic answer or change the primary metric
after seeing these values.

## Same-image qualification

The Window and Bottom questions reuse identical input features and input E.
Their layer-16 visual feature hashes are **not bit-identical across question
forwards**, which have different sequence lengths (638 and 650). The maximum
layer-E difference is 0.014969, but the full 576-position E rank agreement is
0.9999976; the FPS basis and selected sets match exactly. Shape-dependent
floating-point behavior is a plausible explanation, not an independently proven
cause. Within each question, repeated boundary hashes match exactly.

Their input I-map rank agreement on the 32 positions is −0.315; layer-16 I-map
agreement is +0.192. The near-identical E ranking and different sensitivity maps
remain interesting, but this is not a perfectly identical-hidden-state causal
comparison of question wording alone. Answer confidence, targets and sequence
shape also differ. No new numerical-debugging investigation is required before
the bounded full sweep; preserve and report this qualification.

## Export fix and full-run instructions

The original summary export failed with an ASCII **decode** error while reading
UTF-8 manifest text. Raw measurements, controls, images, features, source evidence
and log are intact. HTML and CSV were rebuilt locally. The earlier explicit
UTF-8 writer fix did not cover these implicit text reads.

The notebook helper now starts the isolated child with `-X utf8`, explicitly
decodes subprocess output as UTF-8, and writes its log as UTF-8. Only notebook
launch/I/O behavior changed; measurement scripts, protocol, native files and lock
are unchanged. The existing calibration therefore still passes the full-run
source-compatibility gate. `PYTHONUTF8=1` alone is insufficient because `-I`
ignores Python environment variables.

In the current live notebook, the minimum change is to edit the helper's command:

```python
command = [str(original_python), "-X", "utf8", "-I", "scripts/run_rq1_layer.py",
           # Keep all existing remaining arguments here.
]
```

Rerun the **helper definition cell**, then run the existing full-sweep call:

```python
RUN_FULL_LAYER_SWEEP = True
layer_full_dir = run_layer(tokens=576, calibration=layer_calibration_dir)
```

The calibration variable should still identify
`/kaggle/working/rq1-layer16-calibration-v1-20261017T230335470061Z`.
Keep the same dataset, native environment, checkpoints and T4 ×2. No calibration
rerun, new environment or input upload is needed in this live session. The updated
helper is also saved in the repository notebook for subsequent sessions; pulling
GitHub alone does not replace a helper already defined in notebook memory.

Observed wall time was **363.62 seconds (6.06 minutes)**, including downloads and
loading. The 384 timed interventions took 212.31 seconds, or 0.553 seconds each.
Linear extrapolation gives 63.69 minutes for 6,912 interventions before controls,
loading, growing JSON writes and archiving. Budget **about 70–90 minutes**; actual
runtime can differ. These instrumented timings are not speed-benchmark results.
Download the new ZIP **and** `.log` before ending the session.

## CPU verification

```bash
environments/editor/.venv/bin/python -X utf8 scripts/analyze_rq1_layer.py \
  --archive research/results/rq1-layer16-calibration-v1-20261017T230335470061Z.zip \
  --log research/results/rq1-layer16-calibration-v1-20261017T230335470061Z.log \
  --expected-sha256 8758b31819ef9f73992dcd46776247b7015107e583ca2615f691b64dd8249fd3 \
  --output research/results/rq1-layer16-review-new
environments/editor/.venv/bin/python -X utf8 research/tests/test_rq1_layer_review.py
```

Six saved-review/launch regression tests pass, covering incorrect targets,
incomplete positions, corrupted/nonfinite effects, boundary drift/failure,
overwrite/hash rejection, original-file preservation and a real isolated UTF-8
child launched by the actual notebook helper. The nine existing CPU layer
protocol tests also pass. No new native Torch tests or GPU inference were run
locally.
