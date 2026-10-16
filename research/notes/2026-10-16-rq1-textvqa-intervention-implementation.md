# TextVQA input-stage extension prepared — 2026-10-16

The next implementation is ready for Kaggle validation: six-case reference audit,
two-question × 32-patch calibration, and a later six-question × 576-patch sweep.
Instructions: [RQ1_TEXTVQA_INPUT_PILOT.md](../RQ1_TEXTVQA_INPUT_PILOT.md).
Fresh-session notebook:
[kaggle_rq1_textvqa_input_pilot.ipynb](../notebooks/kaggle_rq1_textvqa_input_pilot.ipynb).
Existing live notebooks can paste the guide's update/tests and run cells instead.

## Selection and scope

The separate tracked selection contains all three full-credit losses from the
100-question screen (37533, 35962, 36001) and three assistant-inspected preserved
text-reading controls (35670, 39323, 38025). Order alternates loss/control;
Brolio and Dominion are first for calibration. The source ZIP and predictions
are fingerprinted. These are six different images, deliberately outcome-enriched,
not a representative sample or matched same-image contrast. None was selected
using reconstruction error or intervention results. Original 100-case manifests,
reference annotations, images, input ZIP and baseline exports were not changed.

## Project changes

- `rq1_protocol.py`: validates the six-case screening identity/reference scores,
  supplies a fixed annotation-only initial-capital TextVQA target and validates a
  compatible passed six-case audit before interventions.
- `run_rq1_input.py`: extends the existing intervention path to TextVQA, using
  full consensus credit for fresh-baseline competence. It stops on baseline or
  alignment changes, preserves failures, and snapshots selection/prerequisite
  audit evidence. It also saves 336-pixel canvases during the target audit.
- `rq1_native.py`: audit records now compare all target-token log probabilities
  to native generation, not just the first. TextVQA question records retain
  screening provenance. The existing zeroing/scoring math is reused.
- New notebook and guide provide explicit audit/calibration/full-pilot phases,
  default-disabled execution, reused bootstrap/logging, and prerequisite tests.

The audit gate checks selection/manifest/seed, requested checkpoint sources,
measurement scripts, native-reference manifest, scorer and lock. A source change
after audit requires another audit. All six detailed audit records and the audit
report are copied into intervention ZIP evidence and hashed. Targets are never
adapted to new predictions. Failed prefix checks retain finite JSON and halt.

Core metric remains `rq1-input-zero-v2`; the new dataset-specific rendering is
separately named `textvqa-initial-capital-v1`. GQA/MMStar rendering and zeroing
semantics are retained. The native ApET/LLaVA source reference (all eight hashes)
and original environment lock were verified unchanged. No model weights or new
dependencies were installed on the Intel Mac.

## Validation and pending work

CPU protocol/runner tests cover reference preservation, the actual six-case
selection, full-credit versus partial-credit semantics, nested calibration order,
missing/failed/stale audit rejection, pre-weight-load stopping, failure ZIPs,
and preservation of all six detailed audit records. Existing scoring/statistical
checks also pass. Notebook code parses and its default flags stop execution
before any subprocess/network call.

Torch-dependent native tests, including the added multiword TextVQA suffix check,
are intentionally skipped on this Mac. The notebook explicitly imports Torch
and runs those tests in Kaggle so absence cannot silently skip them there.
Pretrained target alignment and GPU calibration are **not yet validated**.
First push the changes, pull once in Kaggle, then run the six-case audit and
two-question calibration. Download both ZIPs and both logs for inspection before
enabling the full sweep. No baseline rerun, new data upload or environment upgrade
is required in the existing live session.
