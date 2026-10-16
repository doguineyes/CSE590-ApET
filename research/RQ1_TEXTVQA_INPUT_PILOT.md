# RQ1 TextVQA input-stage pilot

Use the existing locked original-ApET environment and **unchanged
`rq1-inputs-v1.zip`**. This adds project measurement support, not a compression
variant. Native ApET/LLaVA files and environment definitions are unchanged.

**Current step:** [audit v2 passed all six cases](notes/2026-10-16-rq1-textvqa-audit-v2-review.md).
[The two-question calibration also passed](notes/2026-10-16-rq1-textvqa-calibration-review.md).
The [six-question full sweep also passed](notes/2026-10-16-rq1-textvqa-full-pilot-review.md):
3,456 interventions, zero full-forward drift and exact repeats of calibration.
No TextVQA rerun is needed. The twelve-case MMStar follow-up remains the next
planned input-stage confirmation; decoder-layer instrumentation follows review.
No additional inference-code update is needed in the current live notebook.
The [first failed audit](notes/2026-10-16-rq1-textvqa-audit-fix.md) remains archived.

## Frozen diagnostic cases

The tracked [selection](protocols/rq1_textvqa_input_v1.json) records identities,
screening predictions, source ZIP/prediction hashes, roles and reasons. It does
not replace or modify the accepted 100-question manifest. All three baseline-
full-credit losses are included, along with three visually inspected preserved
literal-text controls. No E or I was used to select these cases.

| Order | ID | Reference | Archived role |
|---|---|---|---|
| 1 | 37533 | Brolio | loss |
| 2 | 35670 | Dominion | preserved control |
| 3 | 35962 | Cookery school | loss |
| 4 | 39323 | Sky mall | preserved control |
| 5 | 36001 | Give up | loss |
| 6 | 38025 | Laughter | preserved control |

These are six different images, not matched same-image causal pairs. Controls
provide explicit text-reading tasks with single/multiword references. The sample
is outcome-enriched and qualitative; it cannot estimate population accuracy.
All 100 screening cases, including gains and partial credit, remain archived.

## Existing live notebook: paste these cells

First **commit and push the new files** locally. Wait for any current model
subprocess to finish before updating code. Keep the previously defined `repo`,
`original_python`, `inputs` and `run_experiment` variables from the working RQ1
notebook. Run this update cell once:

```python
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True, timeout=120)
subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True)
subprocess.run(
    [str(original_python), "-I", "-c", "import torch; print(torch.__version__)"],
    cwd=repo, check=True,
)
for test in [
    "research/tests/test_rq1_input.py",
    "research/tests/test_rq1_textvqa_runner.py",
    "research/tests/test_rq1_native.py",
]:
    subprocess.run([str(original_python), "-I", test], cwd=repo, check=True)
```

No package installation, kernel restart, new input upload or previous baseline
rerun is needed in that live session. The following cells give the audit and
later calibration sequence; the current live session has completed the audit.
They intentionally load/release the model once per subprocess; weights and the
environment remain cached. Do not change code between audit and interventions.

**1. Six-question reference audit; no interventions:**

```python
textvqa_audit_dir = run_experiment(
    "rq1-textvqa-target-audit-v2", "textvqa/manifest.json",
    ["--mode", "target-audit", "--questions", "6"],
)
```

The fixed reference is the unchanged canonical annotation with its first
character uppercased; the rest is preserved (`cookery school` → `Cookery school`).
This annotation-only rule is fixed before new generation. The audit compares
literal, fixed-capitalized and diagnostic generated forms. It never adopts the
generated form as a target. Every fixed reference must have:

- fresh full consensus credit using all ten annotator answers;
- token IDs exactly matching the native generated prefix;
- full teacher-forcing reference-repeat, clone-only no-op and native multimodal
  parity differences no greater than 0.001 nats for every target token;
- first-target-token generation/teacher-forcing agreement within 0.001 nats.

Audit v2 records the **all-token cached-generation gap as a diagnostic**, rather
than requiring cached decoding and full teacher forcing to agree within that
tolerance for every later token. This separates execution paths: all intervention
S/I values use full teacher forcing. The
[failed first audit](notes/2026-10-16-rq1-textvqa-audit-fix.md) had exact token
alignment and repeatability but a 0.00550-nat cached/full gap on Cookery school.
The revised audit adds no-op/native-forward checks without increasing tolerance
or changing targets. It must be rerun under the final code; the old ZIP is not
rescored as passed. The audit saves model-scale canvases for inspection too.

A mismatch saves its evidence and stops. Download the failed ZIP/log for review;
do not increase tolerance, silently swap a target or replace the question.

**2. Two-question calibration: loss + preserved control, 32 patches each:**

```python
textvqa_calibration_dir = run_experiment(
    "rq1-textvqa-calibration-v1", "textvqa/manifest.json",
    ["--questions", "2", "--tokens", "32",
     "--target-audit-report", str(textvqa_audit_dir / "report.json")],
)
```

The runner rejects a failed/incomplete/stale six-case audit **before loading
weights**. It checks manifest, selection, seed, measurement sources and locked
environment identity. The calibration repeats fresh baseline competence and
exact prefix alignment, reference repeats, clone-only no-op, native/cached
forward parity and a final reference. A fresh baseline that loses full credit
stops the run with its raw answer saved, without retargeting or replacement.

**Download both ZIPs and both logs and send them for review. Stop here for the
first execution.** Rho need not be positive: this stage validates the measurement,
not a desired research outcome.

## Fresh notebook alternative

Import [kaggle_rq1_textvqa_input_pilot.ipynb](notebooks/kaggle_rq1_textvqa_input_pilot.ipynb)
into an editable Kaggle draft. Enable Internet/T4 ×2 and attach the existing v1
input bundle. For the current retry set `RUN_TARGET_AUDIT = True`, keep
`RUN_CALIBRATION = False` and `RUN_FULL_PILOT = False`, and run cells in order.
Enable calibration only after audit review. Default flags are all False to
prevent accidental inference during import. Setup installs the same locked
environment only when necessary. The notebook captures logs and preserves failed
run ZIPs. It also supports an existing extracted audit/calibration directory.

## Later full-patch pilot

After calibration review, run all six frozen questions × 576 patches:

```python
textvqa_pilot_dir = run_experiment(
    "rq1-textvqa-input-pilot-v1", "textvqa/manifest.json",
    ["--questions", "6", "--tokens", "576",
     "--target-audit-report", str(textvqa_audit_dir / "report.json")],
)
```

In the dedicated notebook, set `RUN_FULL_PILOT=True` in a separate cell and run
only its full-pilot cell after review; retain the audit/calibration variables.
Do not rerun the configuration/setup cell just to toggle that flag. The full
cell requires the passed two-question, 32-patch calibration. Runtime should be
estimated from the calibration; six full grids equal 54 times its number of
interventions, though prompts/answer token lengths can change forward time.

If the measurement sources, lock, selection or seed change after the audit, run
a new audit under the final code. Across sessions, restore the **whole extracted
audit ZIP**, not just `report.json`: its detailed six-case evidence is copied
into downstream result ZIPs too. A new environment/session may still require
checkpoint downloads. Reusing cached files does not preserve a live loaded model.

## Measurement and record keeping

The existing input-zero-v2 metric is retained: native pre-projector reconstruction
error E versus `S(full) − S(zero-one-projected-patch)`, where S is mean fixed
reference token log probability without EOS. Sequence shape/order/masks stay
fixed. TextVQA policy is separately labeled `textvqa-initial-capital-v1`.
Multiword targets are not reduced to their first token. Negative I is retained.

Result ZIPs include the original accepted manifest, separate frozen selection,
native/project source snapshots and lock, screening provenance, used raw and
336-pixel canvas images, cached features, generated/target token IDs, per-token
reference and perturbed scores, controls, tables and figures. Intervention ZIPs
also contain the prerequisite audit report and all six detailed audit records.
Keep the external `.log` beside each ZIP. The existing CPU role-aware summarizer
can review these saved intervention folders without loading model weights.

Singleton feature zeroing is a sensitivity proxy, not native selection/merging
or a direct causal estimate of local pixels. All cases are baseline-correct in
the earlier screen, but that alone does not demonstrate visual grounding.
Model-scale maps and preserved controls are needed before interpreting text loss.
