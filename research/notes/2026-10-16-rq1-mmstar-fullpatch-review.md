# MMStar: full-patch RQ1 check — 2026-10-16

The same three MMStar cases completed **576 interventions each**, for 1,728
measured effects. All controls passed. The full maps show almost no monotonic
relationship between native reconstruction error E and signed answer sensitivity
I in these cases. The positive correlations in the earlier 32-patch calibration
did not persist across the full patch grid.

## Archived evidence and checks

Original ZIP/log were copied from the supplied Downloads paths into
`research/results/rq1-mmstar-threecases-v2-20261016T170443901294Z.{zip,log}`;
Downloads originals remain intact. ZIP: 18,296,408 bytes, 48 entries, CRC passed.
ZIP SHA-256:
`12c99ee7ba5fb33f7b320ce2c696b399df11114eeefe871b0925e7bcdd26ebf8`.
Log SHA-256:
`ae69083405687ce96d9d5c3beb922aee78784af8bc7b253367673b82e5241d65`.

Runtime commit `a3842946ccd91c35805c161dca1f23729000d325`, clean working tree,
protocol `rq1-input-zero-v2`. The accepted manifest, native reference hashes and
original dependency lock match the previous runs. All recorded evidence hashes,
three image hashes and three cached tensor-file hashes were checked. Every
question has 576 distinct, finite patch measurements; all 1,728 exported rows
agree with the raw question records and native E diagnostics. I arithmetic and
correlations were independently recalculated.

Fresh uncompressed answers were D, A and A, all correct and format compliant.
Fixed target token IDs match native generated prefixes. Reference repeats,
clone-only no-ops, native/cached parity, final references and generation-versus-
teacher-forcing score comparisons have **zero drift** on all three cases.

## Calibration versus all patches

| Case | Task | Archived input-only role | Rho, first 32 patches | Rho, all 576 | Rho, 566 non-basis sources |
|---|---|---|---:|---:|---:|
| 313 | Count buildings | loss | +0.28446 | −0.01266 | −0.01517 |
| 617 | Compare beak curvature | loss | +0.42925 | +0.02358 | +0.02213 |
| 1014 | Count bars below six | preserved | −0.02163 | −0.04203 | −0.03916 |

Median all-patch rho is **−0.01266**. Excluding the ten FPS basis sources per
image does not change the near-zero pattern. The auxiliary non-basis E versus
absolute-I correlations are −0.01223, +0.11314 and −0.04298, respectively;
absolute influence is a different statistic from the primary signed metric.

Crucially, **all 96 repeated interventions match the earlier calibration
exactly**, for both E and I. CLIP/projected feature hashes, prompts, targets,
seeds and reference scores also match. The full execution order begins with the
same earlier 32 indices. Thus the change in rho comes from adding the remaining
544 measurements, rather than a change in these repeated measurements. The
remaining-544 correlations are −0.02890, −0.00081 and −0.03990. This demonstrates
why the 32-patch run served as calibration, not a stable mechanism conclusion.
It does not establish general numerical determinism.

The archived roles describe the earlier native input-only answers. This run
measures zeroing effects in the uncompressed model; it does not rerun native
compressed generation or measure the causal effect of native merging.

## Interpretation and maps

All three saved figures were visually inspected. Reconstruction-error maps and
sensitivity maps differ substantially. The beak example has localized positive
effects up to +0.28758 nats for a single patch, even though its full-grid rank
correlation is near zero. Near-zero rho therefore does not mean no patch matters.
The chart includes negative effects down to −0.09737 nats: removing a patch can
increase the target score under this intervention, without implying a better
generated answer.

Some sensitive features correspond to padding or areas away from the apparent
answer evidence. CLIP patch embeddings are contextualized; their sensitivity
cannot be interpreted as the causal importance of the pixels in that patch
alone. Likewise, low-error FPS basis sources are represented separately in native
compression, and other original patches can contribute through merging. Neither
low E/high I nor lack of direct retention alone proves information was lost.

These three selected MMStar cases are consistent with the GQA pilot's lack of a
reliable positive E–I ordering under this measurement. They do not establish a
population-wide result, rule out nonlinear relationships, or show that ApET is
ineffective. Do not treat the 1,728 spatially dependent patches as independent
question samples or pool them into an unsupported significance claim.

## Runtime and saved analysis

Wall time: **1,165.53 seconds = 19.43 minutes**. The three measurement loops took
381.30, 387.49 and 378.50 seconds. Peak recorded allocated memory in baseline
generation was 7.493 / 7.307 GiB. This is an instrumented scoring run, not a
compression-speed benchmark.

The separate [review gallery](../results/rq1-mmstar-threecases-v2-20261016T170443901294Z-review/index.html)
retains question tables, native patch roles and low-E/positive-I candidates.
`calibration-comparison.json` records the exact cross-run comparison. Original
ZIP, log and extracted experiment records were preserved. The CPU summarizer's
generic description was corrected from “six-image” to “input-stage” so it also
describes this three-image review accurately; no inference or native code changed.

## Next step

No rerun or larger MMStar expansion is needed now. Prepare a small TextVQA
input-stage pilot using the already saved 100-question baseline: include the
three full-credit losses and predetermined baseline-correct preserved controls.
Confirm exact reference-token formatting and the same parity/no-op/repeat checks
before full patch sweeps. Keep all baseline records, including gains and partial
credit; any mechanism subset and its selection rationale should be explicit.
Decoder-layer work remains a subsequent step after this three-dataset input-stage
measurement has been checked.
