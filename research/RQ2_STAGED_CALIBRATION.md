# RQ2: native staged ApET versus matched pure random

**Current next experiment.** Defer broad input-only TextVQA100 as an optional
ablation. First calibrate native ApET and random retention with identical input
and decoder compression boundaries/counts.

## Fixed panel and conditions

Six questions / five images from the existing `rq1-ih-inputs-v1` Dataset:

- GQA201636988 /201637316: cake/plate and stove, same image.
- MMStar136: clean soccer control. This new protocol uses136 instead of the
  earlier input-only calibration's flagged chart1014; old panels stay unchanged.
- TextVQA37533 /35670 /35962: Brolio, Dominion, Cookery school.

| Method | Input states | States after block16 | Replicates |
|---|---:|---:|---:|
| Uncompressed LLaVA | 576 | 576 | 1 |
| Original native ApET | 96 | 32 | 1 |
| Pure random | 96 | 32 | 101 /202 /303 |
| Original native ApET | 288 | 96 | 1 |
| Pure random | 288 | 96 | 101 /202 /303 |

Nine conditions ×6 questions = **54 primary predictions**. The released GQA
script uses96→32, TextVQA288→96, both after block16. Across32 equally counted
decoder layers their mean visual counts are64/192; these are not input-only
budgets. Equal schedules do not establish equal runtime. This selected cohort,
with our existing prompts/scorers, is not a complete paper reproduction.

Native arms call the original `encode_images` and `layer_prune`: FPS, E, basis
replacement and merging execute at each boundary. Decoder E is newly computed
from the surviving contextualized hidden states. Pure random uses the original
full576 projection and an input mask, then uniformly gathers surviving layer16
visual states, without FPS/E/basis/merge. Both stages restore sequence order.
Input96 selections are nested within288 for a replicate; layer16 has an independent
image/replicate/stage-derived RNG stream. Selection never uses question outcomes.

## Fidelity and controls

- Original source/lock and accepted input identities are pinned. First input
  realizations are shared per image/condition. Replaying **post-input RNG state**
  preserves the native FPS continuation at decoder16, as well as the feature tuple.
- Original forward, cache update and decode-position adjustment remain in charge.
  Random replaces only the prune gather. Layer outputs are captured per question.
- Full576 input + identity/no-op at layer16 on all six questions: generated IDs,
  processed generation scores and answers must equal the uncompressed reference.
- All nine conditions have fixed clone replays on cake/plate and Brolio, with exact
  pre-layer source/RNG and output-fingerprint checks.
- Four-token soccer probes run native and random101 for both schedules. Blocks1
  and17 must show the correct prefill/cache lengths and positions for three cached
  decode steps. Forced-length answers are controls, not primary predictions.
- Two fresh **whole native pipeline** runs per schedule on Brolio use no feature
  replay or replacement prune. Record answer/score variation without replacing
  the first primary. Do not demand byte equality of separate CUDA merge runs.
  Tiny deterministic CPU fixtures separately compare observed/direct native IDs
  and scores.

There is no compressed-accuracy pass threshold. Existing dataset scorers and all
ten TextVQA answers remain fixed. Seeds repeat questions, not independent samples.
No I interventions, target-format audit, new libraries or native source edits.

## Fresh Kaggle notebook

1. Commit and push changes to `apet-analysis`.
2. Import [kaggle_rq2_staged_calibration.ipynb](notebooks/kaggle_rq2_staged_calibration.ipynb)
   into an editable draft. Enable **Internet + T4 ×2**.
3. Attach the same private **rq1-ih-inputs-v1** Dataset; no new input upload.
4. In the first code cell set:

   ```python
   RUN_STAGED_CALIBRATION = True
   ```

5. Run cells in order. Actual tiny native prerequisites require **zero skips**
   before pretrained loading; a saved proof binds the pass to current code/runtime.
6. Download `rq2-staged-calibration-v1-<timestamp>.zip` and matching `.log` for review.
   Preserve failed/partial archives or a failed prerequisite's setup log too.

The model loads once; conditions run sequentially. Default False prevents accidental
background inference. Wall time and ZIP size remain to be measured. Review this
calibration before implementing broader staged TextVQA100/GQA/MMStar runs.

## Existing live session

If the previous original environment and helper variables still exist, push first
and add this cell to reuse the environment and checkpoint cache:

```python
import json, re, shutil, subprocess, zipfile
from datetime import datetime, timezone
from pathlib import Path

subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True, timeout=120)
notebook = json.loads((repo / "research/notebooks/kaggle_rq2_staged_calibration.ipynb").read_text())
helpers = ["".join(c["source"]) for c in notebook["cells"]
           if c["cell_type"] == "code"
           and "".join(c["source"]).startswith("def verify_rq2_staged_prerequisites(")]
if len(helpers) != 1:
    raise RuntimeError("Expected one staged prerequisite/helper cell.")
exec(compile(helpers[0], "rq2-staged-helpers", "exec"), globals())
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
previous_setup_log = setup_log
setup_log = Path(f"/kaggle/working/rq2-staged-setup-{stamp}.log")
shutil.copy2(previous_setup_log, setup_log)
runtime_report = Path(f"/kaggle/working/rq2-staged-runtime-{stamp}.json")
verify_rq2_staged_prerequisites(refresh_runtime=True)
staged_calibration_dir = run_rq2_staged_calibration()
```

Assumes `repo`, `inputs`, `original_python`, `setup_log`, `stream_command` and
`CHECKPOINT_ARGS`. If the session closed, use the new notebook from its beginning.
Independent sessions do not share virtual environments or `/tmp` checkpoints.

## Evidence and local checks

The ZIP contains raw answers/scores, CSV/HTML, images/model canvases, E/basis/merge
graphs, random memberships, source/output/RNG hashes, cache traces, source/lock
files, setup/runtime/test proof and an artifact hash inventory. Input tensor banks
are saved once per image; pure masks reference full576 bytes. Layer tensors are
saved only for soccer and Brolio. Other cases retain detailed JSON evidence.
Controls do not enter the54-prediction summary. Instrumented timings are not a
compression-overhead or speed benchmark.

```bash
environments/editor/.venv/bin/python -m unittest \
  research.tests.test_rq2_staged_protocol \
  research.tests.test_rq2_staged_runner -v
```

Actual tiny18-block native tests run in Kaggle. No Torch installation or model
execution on the Intel Mac. Protocol/runner/random adapter are project-owned;
native model files and the original runtime stay unchanged.
