# RQ2: pure random input retention at 288 and 64 tokens

This is the first execution calibration for the
[revised RQ2 plan](notes/2026-10-20-rq2-revised-evaluation-plan.md).
The [six-question Kaggle calibration passed](notes/2026-10-20-rq2-pure-random-calibration-review.md):
native input64 is verified, with 2/6 correct versus 2–4/6 for pure random seeds.
This is a selected execution panel, not broad benchmark evidence. Old native,
basis-fixed random, routing/protection and oracle protocols/results are unchanged.
The locked original environment and native model files stay unchanged.

## What runs

Six questions / five images from the existing accepted RQ1/RQ2 input bundle:

- GQA 201636988 / 201637316: same image, different questions.
- MMStar 1014: a bar-chart count with an existing grounding flag. An earlier
  description mistakenly called it soccer; the actual soccer control is 136.
  The frozen case and its flag remain unchanged. Its all-A answers do not establish
  preserved visual understanding.
- TextVQA 37533 / 35670 / 35962: Brolio, Dominion and Cookery school.

Each question has eleven primary conditions:

- Full 576-token reference, once.
- Native ApET input 288 and input 64, each with and without merging: four arms.
- Pure random input 288 and input 64, each at replicates 101/202/303: six arms.

Native retains ten FPS basis representations, so input 64 has 54 ordinary targets
and ten native basis slots. Decoder compression is off everywhere. K means tokens
entering the decoder, not an average across layers or a paper-reproduction budget.

Pure random samples K distinct original positions from 576, without replacement,
using an isolated image/replicate-derived RNG. One permutation supplies both
budgets, then the selected positions are spatially sorted. It invokes the native
uncompressed vision/projector branch and changes only the output selection mask:
no FPS, reconstruction solve/E ranking, basis replacement or merging. The projector
still processes all 576 states before masking. This preserves original vectors
and native insertion geometry; it is **not a compression-overhead benchmark**.
The old basis-fixed random control is a separate component ablation.

## Controls and saved evidence

- Native source and original lock identity, accepted images/prompts/targets/seeds.
- Exact native merge/no-merge selection and basis assignment at each budget.
- Exact archived CLIP, full projection, E/FPS and full reference scores.
- Nested random subsets with exact counts, unchanged original projected vectors.
- Pure random at 576: exact full features/mask, generated IDs and reference scores.
- Exact frozen feature replay for generation and fixed-gold scoring. Same-image
  questions share one first-captured feature bank, without answer-based retries.
- A second replay of every primary condition on the first GQA question and Brolio.
- Two additional fresh native captures per budget on Brolio. Their numerical and
  answer changes are recorded separately; fresh merging is not required to be
  bitwise identical. No compressed answer-accuracy threshold determines a pass.

The ZIP includes raw predictions, answers CSV, per-question controls, images and
actual model canvases, runtime/setup evidence, pinned sources, hashes and an HTML
summary. Full native tensors are stored once per image. Pure masks reference the
identical baseline projected tensor, avoiding six redundant full tensor copies.
Fresh native repeat tensors are also preserved with their numeric comparisons.
The separate console log is also required. Failure/interruption preserves partial
evidence, with its status explicit. These selected cases are not broad accuracy
estimates. There is deliberately no TextVQA100/full-benchmark switch yet.

## Fresh Kaggle notebook

1. Commit and push the new files to `apet-analysis`.
2. Import [kaggle_rq2_random_calibration.ipynb](notebooks/kaggle_rq2_random_calibration.ipynb)
   into an editable draft. Enable Internet and T4 ×2.
3. Attach the existing private **rq1-ih-inputs-v1** Dataset. It is the same bundle
   used for the recent RQ2 runs, not the older `rq1-inputs-v1` bundle. No new input
   upload is necessary if the existing Dataset is available under Add Input.
4. In the first code cell set:

   ```python
   RUN_RANDOM_CALIBRATION = True
   ```

5. Run cells in order. The notebook installs/reuses the original isolated runtime,
   runs CPU and actual tiny native prerequisites with zero allowed skips, then
   loads pretrained LLaVA once for calibration.
6. Download `rq2-pure-random-calibration-v1-<timestamp>.zip` and matching `.log`.
   Download the displayed setup log/runtime report as well. Review these before
   the whole TextVQA100 comparison. Stop on any prerequisite failure and retain
   its setup log; no pretrained run should start after failed prerequisites.

The notebook defaults to False to prevent an accidental imported/background run.
Its input locator handles either the extracted Kaggle Dataset or the original
ZIP. Optional `INPUT_ZIP` can name an exact path to resolve multiple matches.

## Existing live notebook

If the original runtime, attached input bundle, and helper variables from the
recent crossed/routing notebook are still present, this cell reuses them. Push
first. It neither reinstalls the runtime nor reuses old logs as new execution logs.

```python
import json, re, shutil, subprocess, zipfile
from datetime import datetime, timezone
from pathlib import Path

subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True, timeout=120)
nb = json.loads((repo / "research/notebooks/kaggle_rq2_random_calibration.ipynb").read_text())
helpers = ["".join(c["source"]) for c in nb["cells"]
           if c["cell_type"] == "code"
           and "".join(c["source"]).startswith("def verify_rq2_random_prerequisites(")]
if len(helpers) != 1:
    raise RuntimeError("Expected one pure-random prerequisite/helper cell.")
exec(compile(helpers[0], "rq2-random-helpers", "exec"), globals())
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
previous_setup_log = setup_log
setup_log = Path(f"/kaggle/working/rq2-random-setup-{stamp}.log")
shutil.copy2(previous_setup_log, setup_log)
runtime_report = Path(f"/kaggle/working/rq2-random-runtime-{stamp}.json")
verify_rq2_random_prerequisites(refresh_runtime=True)
random_calibration_dir = run_rq2_random_calibration()
```

The cell assumes `repo`, `inputs`, `original_python`, `setup_log`, `stream_command`
and `CHECKPOINT_ARGS` already exist. A closed/new session should use the new
notebook from its first cell. Environments and `/tmp` weight caches are not shared
between independent Kaggle notebook sessions.

## Local checks and implementation ownership

```bash
environments/editor/.venv/bin/python -m unittest research.tests.test_rq2_random_protocol research.tests.test_rq2_random_runner -v
```

Run native tests in the original Linux environment:

```bash
environments/original/.venv/bin/python -m unittest research.tests.test_rq2_random_native -v
```

The new protocol/native helper/runner are project-owned files, not an ApET release
or a replacement of the native model. CPU tests do not establish pretrained
correctness. The notebook runs the actual tiny native tests before the calibration;
they exercise 64-token preparation, cached decoding, replay/scoring, FPS/solve
guards and restoration after exceptions, without downloading extra checkpoints.
