# MMStar RQ1 calibration v2 passed — 2026-10-16

All three selected MMStar questions passed the input-stage measurement check:
three images × 32 uniformly sampled patches = **96 interventions**. Proceed to
the retained TextVQA baseline. After that, measure all 576 patches on these same
three MMStar cases before expanding the case set or comparing datasets broadly.

## Evidence and execution

Original ZIP/log:
`research/results/rq1-mmstar-calibration-v2-20261016T160231333218Z.zip` and adjacent
`.log`. Original ZIP is 17.2 MiB / 48 entries; CRC validation passed. Extracted
files preserve native source snapshots, lock, manifest, three image-feature
tensors, figures, reference controls and raw token scores. All recorded evidence
and tensor-file hashes were verified. All 1,728 token-table rows were checked:
96 measured I values, 1,632 unmeasured/null I values. Arithmetic matches the raw
reference-minus-perturbed scores.

Protocol `rq1-input-zero-v2`, runtime commit
`a3842946ccd91c35805c161dca1f23729000d325`, clean working tree. The original native
source reference and dependency lock remain unchanged. The logged model-type and
TypedStorage warnings were non-fatal; all requested cases completed successfully.

Fresh baseline responses were exactly `D`, `A`, `A`, all correct and format
compliant. All fixed option-token prefixes matched native generation. Reference
repeat, no-op, cached/native parity, generated-token comparison and final-reference
score differences were **zero for all three cases**.

Wall time **75.84 seconds**, with 19.66/19.53/20.00 seconds per question's
measurement loop. Peak allocated GPU memory **7.49 / 7.31 GiB**. This is an
instrumented calibration, not a native generation-speed benchmark.

## Initial observations

| Case | Task | Target | Archived input-only role | E–I rho, sampled 32 | Excluding sampled FPS basis sources |
|---|---|---|---|---:|---:|
| 313 | Count buildings | D (four) | loss | +0.28446 | +0.34476 (31 patches) |
| 617 | Compare bird-beak curvature | A | loss | +0.42925 | +0.42925 (32 patches) |
| 1014 | Count bars strictly below six | A (zero) | preserved | -0.02163 | +0.11991 (30 patches) |

The first two samples exhibit a positive reconstruction-error/contribution rank
relationship; the chart case is near zero for the primary signed metric. The
building and beak examples have 28/32 and 24/32 positive I values; the chart has
25/32. Signed-I ranges are [-0.00888,+0.06290], [-0.05543,+0.18370] and
[-0.00953,+0.02719], respectively. All three exported plots were inspected.

This is not yet evidence that MMStar differs systematically from GQA: it contains
only three selected cases and 32 sampled patches each, versus 576 patches per
GQA question. The prior GQA calibration/full-run comparison already showed that
individual rho estimates can shift after adding the other patches. Keep all
three cases, including the near-zero chart case, rather than expanding only the
positive examples.

Building-count baseline option-prefix probability was 0.30854; runner-up B was
close (a 0.046875-nat first-token margin). Beak-case A had probability 0.45159,
and chart A 0.66963. These are exact option-token probabilities under the full
vocabulary, not calibrated task-confidence measures. Baseline correctness alone
does not demonstrate visual grounding or rule out answer/option priors.

FPS basis handling matters: the chart calibration sampled two basis sources,
and its correlation changes sign when they are separated. This supports keeping
the primary all-token metric and the role-aware auxiliary analysis distinct.
Individual zeroing effects do not directly predict native merging/compression
outcomes, even though two cases were losses in archived input-only evaluation.

## Saved CPU review

The [role-aware gallery](../results/rq1-mmstar-calibration-v2-20261016T160231333218Z-review/index.html)
and adjacent CSV/JSON files were generated separately with
`scripts/summarize_rq1_pilot.py`. Original ZIP, predictions and measurements were
not rewritten. The partial tables retain every original token index and explicit
unmeasured entries; analysis uses only measured interventions.

## Next run: TextVQA baseline

In the existing live notebook, add this cell and run it directly:

```python
textvqa_dir = run_experiment(
    "rq1-textvqa-baseline", "textvqa/manifest.json",
    ["--mode", "textvqa", "--configurations", "baseline", "input-only"],
)
```

This runs all **100 retained questions × two configurations = 200 generations**.
Input-only ApET retains 288 visual tokens; baseline uses 576. Native ten-annotator
TextVQA consensus scoring is used, without OCR transcripts or reference answers
in generation prompts. Keep all cases, including partial-credit and baseline-wrong
ones. This establishes baseline competence and preservation; TextVQA token
interventions still require a defined reference-format policy afterward.

Reuse the current input bundle, code, environment and checkpoint cache. Download
the ZIP/log. No MMStar code/environment repair or repeat calibration is needed.
The bounded full-patch MMStar follow-up, after TextVQA screening, can use:

```python
mmstar_fullpatch_dir = run_experiment(
    "rq1-mmstar-threecases-v2", "mmstar/manifest.json",
    ["--questions", "3", "--tokens", "576"],
)
```

That call preserves the same seed-ordered three cases, targets and token-order
prefix. It is a later separate run, not an expansion to all reviewed MMStar cases.
