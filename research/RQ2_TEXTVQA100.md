# RQ2: TextVQA100, native ApET and pure random

**Deferred optional ablation.** Run the [small staged calibration](RQ2_STAGED_CALIBRATION.md)
first. The next main system comparison uses native ApET and random retention with
matched two-stage schedules. This input-only runner remains available separately.

The [six-case pure-random calibration passed](notes/2026-10-20-rq2-pure-random-calibration-review.md).
The next comparison uses **all 100 existing TextVQA screening questions**, including
baseline-wrong and partial-credit cases. It tests whether the selected-case
random rescues and exact-text failures persist across a broader predefined pool.
This is a development/screening cohort, not independent held-out or full TextVQA.

## Conditions and outcomes

Each question has 11 primary predictions, for **1,100 total**:

| Condition | Visual tokens entering decoder | Replicates |
|---|---:|---:|
| Full uncompressed reference | 576 | 1 |
| Native ApET with merging | 288, 64 | 1 per budget |
| Native ApET with identical selection/basis, no merging | 288, 64 | 1 per budget |
| Pure random retention, without FPS/E/basis/merging | 288, 64 | seeds 101/202/303 per budget |

Decoder pruning is disabled. These are **input budgets**, not the paper's reported
staged-compression settings. Native uses ten FPS basis representations at both
budgets. Pure random retains original projected states; it uses the same
image-derived random permutations as the calibration, with 64 a subset of 288.
Native and pure random differ in basis replacement and merging as well as ordinary
selection; this alone does not isolate an E-only causal effect. Earlier basis-fixed
and merge/no-merge ablations remain separate evidence.

Prompts, all ten answers, image bytes, seed rule, FP16, checkpoint revisions,
greedy generation and 16-token maximum remain fixed. There are no OCR transcripts
in the prompt and no outcome-based exclusions. The new runner measures generated
answers; it adds no token interventions, reference-answer likelihood targets or
target-format audit. Zero/partial-credit baseline answers remain in the cohort.

Reports include native ten-annotator consensus mean, full-credit counts, APR
(fraction of freshly baseline-full-credit questions still receiving full credit),
paired score changes, full-credit gains/losses, empty answers and output caps.
Each random seed is reported separately, followed by its mean/range. Paired 95%
bootstrap intervals use questions/images as the unit; the seed-averaged difference
still has 100 observations, not 300. Intervals condition on these three seeds and
do not establish generalization outside the screening pool.

The full100 result is primary. The **94 outside the six previously studied TextVQA
cases** and the six diagnostic cases are separate secondary summaries. This does
not make the remaining 94 a held-out test: they were in the previous screening run.

## Execution and evidence

One pretrained model load runs all conditions sequentially. First unconditional
captured features are reused for generation and repeats; no answer-based retries
or replacement of an unfavorable realization are allowed. For every image, the
runner checks exact CLIP source identity, native merge/no-merge selection and basis,
consistent E/FPS across budgets, pure-random membership and original feature identity.
Generation traces must show the exact input count and no decoder pruning.

Full576 pure-random generation parity is checked on the six diagnostic questions.
All11 frozen conditions are repeated on the first screening question and Brolio.
Two additional fresh native merge captures at each budget on Brolio record numerical
and answer variation separately. Fresh merged features need not be bitwise equal;
source, selection, basis and routing structure must remain exact.

Historical full576/native288 responses for all100 are pinned in
`protocols/rq2_textvqa100_reference_v1.json`, with the original archive/prediction
hashes. Any new historical answer/score drift is reported, **not used to filter
cases or retarget measurements**. Original archives remain intact.

The ZIP preserves predictions, original questions/references/images, actual336×336
model canvases, per-image E/FPS/selection/merge graphs and source/output fingerprints,
per-question controls, all source/lock identities, notebook setup/runtime/test
evidence, CSV, paired summaries and an HTML gallery. Full tensor banks are retained
only for **Brolio and Dominion**, plus Brolio's four fresh captures. Other cases retain
encoding hashes and graph records, rather than automatically exporting100 tensor
banks. These instrumented timings are not a speed benchmark. Failure/interruption
also preserves partial evidence; a fresh output path is required for another run.

## Fresh Kaggle notebook

1. Commit and push the new files to `apet-analysis`.
2. Import [kaggle_rq2_textvqa100.ipynb](notebooks/kaggle_rq2_textvqa100.ipynb) into an editable draft.
3. Enable **Internet and T4 ×2**. Attach the same private **rq1-ih-inputs-v1**
   Dataset used for the recent RQ2 calibration. It already contains all100 images;
   no new ZIP upload is needed.
4. In the first code cell set:

   ```python
   RUN_TEXTVQA100 = True
   ```

5. Run cells in order. Setup reuses or creates the locked original runtime. All
   old and new prerequisites must pass with **zero skips** before pretrained
   loading. A saved proof binds that pass to the current sources, tests and runtime.
6. Download `rq2-textvqa100-random-v1-<timestamp>.zip` and the matching `.log`.
   Send both for verification and analysis before choosing the next dataset.

Allow roughly **30–60 minutes after setup**, as a conservative planning estimate
from the six-case run, not a measured full100 runtime. This runner omits the
calibration's teacher-forced scoring passes. Actual progress and final wall time
will establish a better estimate. Weight downloads/setup are additional if the
session is fresh. Results should be much smaller than the earlier multi-GB full
tensor exports, but their exact ZIP size is not yet measured.

## Existing live Kaggle notebook

Push first. If the original runtime, Dataset and helpers from the calibration
still exist in this session, add this cell. It fetches the new runner without
reinstalling the environment or redoing the six-case calibration:

```python
import json, re, shutil, subprocess, zipfile
from datetime import datetime, timezone
from pathlib import Path

subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True, timeout=120)
notebook = json.loads((repo / "research/notebooks/kaggle_rq2_textvqa100.ipynb").read_text())
helpers = ["".join(c["source"]) for c in notebook["cells"]
           if c["cell_type"] == "code"
           and "".join(c["source"]).startswith("def verify_rq2_textvqa_prerequisites(")]
if len(helpers) != 1:
    raise RuntimeError("Expected one TextVQA100 helper cell.")
exec(compile(helpers[0], "rq2-textvqa100-helpers", "exec"), globals())
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
previous_setup_log = setup_log
setup_log = Path(f"/kaggle/working/rq2-textvqa100-setup-{stamp}.log")
shutil.copy2(previous_setup_log, setup_log)
runtime_report = Path(f"/kaggle/working/rq2-textvqa100-runtime-{stamp}.json")
verify_rq2_textvqa_prerequisites(refresh_runtime=True)
textvqa100_dir = run_rq2_textvqa100()
```

This assumes `repo`, `inputs`, `original_python`, `setup_log`, `stream_command`
and `CHECKPOINT_ARGS` exist. If the session closed, use the new notebook instead.
Independent Kaggle notebook sessions do not share `.venv` or `/tmp` model caches.

## Local checks

```bash
environments/editor/.venv/bin/python -m unittest \
  research.tests.test_rq2_textvqa_protocol \
  research.tests.test_rq2_textvqa_runner -v
```

Actual tiny native tests run in Kaggle through the notebook; do not install Torch
on the Intel Mac. These new protocol/runner/notebook files are project-owned.
Native ApET model files, earlier frozen protocols and dependency locks are unchanged.
