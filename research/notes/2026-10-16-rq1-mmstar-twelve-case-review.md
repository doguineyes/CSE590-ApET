# MMStar twelve-case RQ1 input-stage review — 2026-10-16

The twelve-case run passed. Both supplied Downloads files were readable and
copied unchanged into `research/results/`; the Downloads originals remain.
This completes the planned broader MMStar input-stage confirmation. No inference
code, native ApET file, dependency lock or frozen input was changed in this review.

## Evidence and validation

Source stem: `rq1-mmstar-twelvecases-v2-20261016T215926569196Z`.

- ZIP: 73,382,595 bytes, 111 entries, full CRC check passed.
  SHA-256: `6263266c03cff9830f5d601c1d85ce0b3a4dfb323cab25a259fdb0b06372c553`.
- Log: 8,278 bytes.
  SHA-256: `9c15cf0adf09de605c16d736c338a01b0fe93b8ee1ea85d49f2050286716cb6c`.
- Runtime commit: `2f66364513a7e50b59a90d383d56faa815c28b35`, clean worktree.
  Protocol: `rq1-input-zero-v2`; accepted native source hashes and original lock match.
- All 12 questions completed all 576 distinct patches: **6,912 interventions**.
  Reference generation was correct and format compliant in all 12 cases; fixed
  option-token targets aligned with generated prefixes. Full-forward reference,
  no-op, native-parity and final-reference controls have zero recorded drift.
- Evidence hashes, saved feature hashes, raw E/I arithmetic, exported analysis
  rows and native basis/selection roles were checked by the CPU review and
  archive checks. Wall time: **73.96 minutes**; peak allocated memory about
  **7.46 / 7.29 GiB** on the two T4s. Instrumented timing is not a speed benchmark.
- The earlier full-grid cases 313, 617 and 1014 repeat **all 1,728 E/I pairs
  exactly**, with matching feature hashes, prompts, targets, seeds and reference
  scores. They are reproduction checks, not additional independent examples.

Derived outputs are separate from the original extraction:
[role-aware gallery](../results/rq1-mmstar-twelvecases-v2-20261016T215926569196Z-review/index.html),
[summary](../results/rq1-mmstar-twelvecases-v2-20261016T215926569196Z-review/summary.json),
[repeat comparison](../results/rq1-mmstar-twelvecases-v2-20261016T215926569196Z-review/threecase-comparison.json).
These result files are ignored by Git; keep an additional backup.

## Findings

Median per-question signed Spearman rho is **+0.00912**; excluding the ten FPS
basis sources gives **+0.00873**. Six questions have positive rho and six negative.
The range is **−0.08087 to +0.17725**. The expansion adds nine new cases and shows
variation beyond the earlier three; it does not show a consistently strong
positive ordering of answer sensitivity by reconstruction error.

| MMStar ID | Task category | Archived input-only role | Signed rho |
|---|---|---|---:|
| 313 | Fine-grained perception | loss | −0.01266 |
| 617 | Instance reasoning | loss | +0.02358 |
| 1014 | Math | preserved | −0.04203 |
| 604 | Instance reasoning | preserved | −0.00535 |
| 895 | Logical reasoning | loss | −0.07005 |
| 825 | Logical reasoning | preserved | +0.15836 |
| 1188 | Math | preserved | +0.11445 |
| 1030 | Math | loss | +0.13144 |
| 626 | Instance reasoning | preserved | −0.08087 |
| 136 | Coarse perception | preserved | +0.17725 |
| 776 | Logical reasoning | preserved | +0.04007 |
| 1372 | Science & technology | preserved | −0.07589 |

All six MMStar categories are represented, but category counts are only 1–3.
These are curated, baseline-correct cases: four archived compression losses and
eight preserved answers. Their median rhos are +0.00546 and +0.01736 respectively;
this small set provides no useful separation of those roles by rho alone.

Visual inspection of the nine new maps suggests useful follow-ups:

- **895, bedroom:** strong positive sensitivity around the bed area despite an
  overall negative rho. Patches 344 and 347 have I = +0.29694 / +0.28952 and are
  outside direct-original retention. They may still contribute through merging;
  these measurements alone do not establish why the compressed answer failed.
- **136, soccer:** the strongest rho is still only +0.177. Patch 201 has
  I = +0.30513 and is directly retained. Patch 21, in the padded image border,
  also has high I (+0.23529). CLIP features carry context, and zeroing is an
  artificial perturbation; a padding coordinate is not proof of meaningless
  information or a failed implementation.
- **825, carpet; 1030, geometry; 1188, chart:** weak positive alignment occurs in
  preserved and loss roles. It should not be discarded because it differs from
  the near-zero early results.
- **626 and 1372:** most singleton zeroings increase the fixed correct-option
  score (negative I). That is a confidence effect, not evidence that most image
  tokens are useless, nor a measurement of answer flips.
- **604:** the frozen prompt contains literal `nan` alternatives for C/D. Keep
  this recorded limitation rather than silently editing or removing the case
  after observing E/I. Its baseline A and full measurement remain reproducible.

E is native reconstruction error before the projector. I is the signed change
in reference-answer log probability when one projected feature is zeroed in the
full 576-token model. It is neither native token deletion nor merging. See the
[cross-dataset summary and next steps](2026-10-16-rq1-input-stage-summary-and-next-steps.md)
for the current scientific conclusion and bounded next experiment.
