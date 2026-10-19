# RQ2 v2: captured native features and separate repeatability controls

**Current result:** the seven-case calibration and full30 Kaggle panel have
passed. See the [full review](notes/2026-10-19-rq2-v2-full-review.md): native
merging gives 19/30 correct and no-merge 21/30 in this selected panel. The run
instructions below remain for reproduction; no repeat is needed to validate
the saved result. Selection/routing/protection controls remain future work.

The diagnostic established that independent original native merge executions can
produce different FP16 feature bytes. The old v1 pilot still requires those
independent outputs to be identical, so running it again cannot fix that check.
Its failed ZIPs and protocol remain preserved.

V2 captures the first native feature realization once per image and arm. It
replays exact clones for all same-image questions, generation and fixed-reference
scoring. Native source, FPS, E, basis placement, selection, merging math, weights
and dependency lock remain unchanged. The no-merge variant still omits only
merging after native selection/basis placement. No numerical tolerance is added.

Independent fresh native captures are separate uncertainty controls. Their
feature deltas, scores and generated-answer changes are recorded; they never
replace the primary first capture. Structural source/selection/merge-graph changes
still fail. Fixed-tuple replay repeats must remain exact.

## First run: seven-question calibration

1. Commit and push all new v2 files to `apet-analysis`.
2. Import [kaggle_rq2_input_merge_v2.ipynb](notebooks/kaggle_rq2_input_merge_v2.ipynb)
   into an editable Kaggle draft. Enable Internet and T4 ×2.
3. Attach the same private `rq1-ih-inputs-v1` Dataset. No new input ZIP is needed.
4. In the first code cell set:

   ```python
   RUN_V2_CALIBRATION = True
   RUN_V2_FULL = False
   ```

5. Run cells in order. Both flags default to False. The setup reuses an existing
   original virtual environment and checkpoint cache in the same live session;
   a new session rebuilds the same locked environment.
6. Download the new `rq2-input-merge-v2-calibration-<timestamp>.zip` and `.log`.
   Review them before the full panel. The old `run_rq2_pilot()` helper is v1;
   the new helper is **`run_rq2_v2()`**.

Calibration uses seven predeclared questions: the first GQA same-image pair,
MMStar313, and TextVQA37533/35670/35962/36001. It includes the known GQA numerical
case, Brolio and preserved controls. The five declared repeat-control questions
run all three arms twice on fixed tuples, plus two additional fresh native
realizations. Control repeats do not increase the sample denominator.

## Reuse the existing live v1 notebook

If the session is still live and `repo`, `inputs`, `original_python`, `setup_log`,
`stream_command` and `CHECKPOINT_ARGS` remain defined, add this cell after pushing.
It refreshes provenance and tests without reinstalling the environment. It copies
the old setup log to a new file rather than modifying the previous evidence.

```python
import json, re, shutil, subprocess, zipfile
from datetime import datetime, timezone
from pathlib import Path

subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True, timeout=120)
updated = json.loads((repo / "research/notebooks/kaggle_rq2_input_merge_v2.ipynb").read_text())
helpers = ["".join(c["source"]) for c in updated["cells"]
           if c["cell_type"] == "code"
           and "".join(c["source"]).startswith("def verify_rq2_v2_prerequisites(")]
if len(helpers) != 1:
    raise RuntimeError("Expected exactly one v2 prerequisite/helper cell.")
exec(compile(helpers[0], "rq2-v2-helpers", "exec"), globals())
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
previous_setup_log = setup_log
setup_log = Path(f"/kaggle/working/rq2-v2-setup-{stamp}.log")
shutil.copy2(previous_setup_log, setup_log)
runtime_report = Path(f"/kaggle/working/rq2-v2-runtime-{stamp}.json")
verify_rq2_v2_prerequisites(refresh_runtime=True)
rq2_v2_calibration_dir = run_rq2_v2(calibration=True)
```

If the session ended or variables were lost, use the new notebook's setup cells.
Do not invoke the old v1 pilot cell after this calibration.

## Full panel after calibration review

In the same live notebook, after the calibration is accepted:

```python
rq2_v2_full_dir = run_rq2_v2(
    calibration=False,
    calibration_report=rq2_v2_calibration_dir / "report.json",
)
```

The CLI verifies the calibration's actual question/evidence/tensor-file hashes,
all seven questions and controls, and matching measurement sources, native/runtime
identity and checkpoints. A diagnostic or v1 report cannot satisfy this gate.
The full run captures new first realizations; it does not choose them to reproduce
calibration answers. Both runs estimate outcomes conditional on recorded native
realizations. Calibration/full differences remain possible and are evidence.

In a fresh notebook, first extract the accepted calibration ZIP into a new
directory and set `CALIBRATION_REPORT` to its `report.json`, retaining all its
subdirectories; then enable only `RUN_V2_FULL`. Attaching the input Dataset alone
does not restore calibration results or share a virtual environment.

## Evidence and limits

Each run archives all first feature realizations before generated answers, JSON
capture/selection/replay controls, fresh repeat tensors and answers, raw primary
predictions, CSV/HTML review, original native/source/profile/runtime hashes and
setup logs. The full ZIP embeds the calibration prerequisite evidence. Preserve
the separate console `.log` as well. Failed runs retain partial evidence; avoid
overwriting or selecting a favorable rerun. Hard session termination can prevent
final ZIP creation.

This is the same curated 30-question/24-image panel, not a population accuracy or
speed benchmark. Exact capture/replay verifies the recorded execution; it does
not prove bitwise repeatability of separate unobserved GPU executions.
CPU tests cannot establish pretrained behavior; the new native Torch tests and
seven-question calibration must run on Kaggle before claiming success.
