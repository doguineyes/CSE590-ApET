# TextVQA calibration passed; next larger RQ1 runs — 2026-10-16

The two-question × 32-patch calibration passed. Proceed to the six-case TextVQA
full-patch pilot, followed optionally by a larger, already supported MMStar
confirmation run. No inference-code change, new input upload or environment
update is needed in the current live notebook.

## Saved evidence and validation

Archived original ZIP/log:
`research/results/rq1-textvqa-calibration-v1-20261016T182930625290Z.{zip,log}`.
ZIP: 12,923,378 bytes / 49 entries, CRC passed. SHA-256:
`0813cf635cb1dc3770107d2f898b9d0dfea33d52a23305f6d053c8180731af13`.
Log SHA-256:
`1680d86201bd8638bdb050853e83374f435a21ab6d3e3d7c40559425fe967a55`.
Downloads originals and all prior artifacts were preserved.

Runtime commit `7bd25dc3309953376ebf7bb0a7511c60c509f405`, clean worktree;
original lock unchanged. All evidence hashes, cached tensor-file hashes, native
E values and I arithmetic were checked. All 1,152 exported patch rows are
consistent with raw records: 64 measured, 1,088 explicitly unmeasured/null.
The embedded prerequisite matches the passed six-case audit exactly. Both image
feature hashes and reference scores exactly repeat the audit's corresponding
values. The separate role-aware CPU review is saved alongside the source folder.

| Case | Archived role | Target | Interventions | Sampled rho | Measurement seconds |
|---|---|---|---:|---:|---:|
| 37533 | loss | Brolio | 32 | −0.19575 | 16.29 |
| 35670 | preserved | Dominion | 32 | +0.04655 | 16.92 |

Both fresh answers have full consensus credit and exact fixed-target prefix
alignment. Reference-repeat, no-op, native/cached-feature parity, first-token
generation comparison and final-reference drift are zero. Later cached-decoding
gaps are retained under the documented TextVQA validation policy. No tolerance
was increased. These 32-patch correlations are calibration observations; the
MMStar calibration/full-grid comparison already showed why they should not be
treated as stable full-image results.

Wall time was **48.37 seconds**; peak allocated memory **7.43 / 7.27 GiB**.
Token-loop time across both questions was 33.21 seconds. Extrapolating to six
576-patch questions gives roughly 30 minutes of token work, but four cases have
different prompts/target lengths. Allow approximately **30–45 minutes** and
inspect actual progress rather than treating this as a guarantee.

## Run 1: TextVQA six cases × 576 patches

```python
textvqa_pilot_dir = run_experiment(
    "rq1-textvqa-input-pilot-v1", "textvqa/manifest.json",
    ["--questions", "6", "--tokens", "576",
     "--target-audit-report", str(textvqa_audit_dir / "report.json")],
)
```

This measures **3,456 effects** on the frozen three losses and three preserved
controls. Its first two cases repeat the calibration's 64 measured patch effects.
Use the passed audit variable already in the live notebook. Do not update
measurement sources between audit and this run. Download the ZIP/log.

## Run 2: MMStar twelve cases × 576 patches

For a second useful run before the user's stated Friday quota reset, the current
CLI already supports twelve questions from the accepted 18-case primary manifest:

```python
mmstar_extended_dir = run_experiment(
    "rq1-mmstar-twelvecases-v2", "mmstar/manifest.json",
    ["--questions", "12", "--tokens", "576"],
)
```

This is **6,912 interventions**, estimated around **75–100 minutes** from the
earlier three-case full run; prompt lengths can change runtime. Run it after the
TextVQA subprocess completes. Each subprocess loads once and reuses cached
weights/environment; no simultaneous GPU jobs are intended. If putting both
calls in one cell, a failed first call raises and prevents the second starting.

Selection uses the pre-existing deterministic hash order, without changing it
after observing E/I. The twelve IDs are 313, 617, 1014, 604, 895, 825, 1188, 1030,
626, 136, 776, 1372. They span all six MMStar categories and contain four archived
input-only losses and eight preserved-correct cases. The first three repeat the
completed full-patch cases; nine are new. Repeats provide an additional cross-run
check. This remains a curated baseline-correct mechanism subset, not a population
accuracy benchmark. New cases receive all existing alignment/numerical controls.

## Research after these runs

The completed GQA 12-question pilot plus these TextVQA six and MMStar twelve
would provide full patch maps for **30 unique questions on 24 images** across
all three proposal datasets. Summarize per-question E–I relationships, inspect
low-E/high-positive-I regions with their native retained/merged roles, and use
same-image GQA contrasts. Keep question/image dependence and selection conditions
explicit; do not pool 576 patches per question into independent population tests.
Then design a small decoder-layer measurement pilot. That stage needs additional
instrumentation and cannot be claimed from these input-stage measurements.
