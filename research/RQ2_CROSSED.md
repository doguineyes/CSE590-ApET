# RQ2: E/random selection × merge/no-merge

The [full30 run passed](notes/2026-10-19-rq2-crossed-full-review.md): E selection
preserves 19/30 with merging and 21/30 without; random seeds preserve 23–26/30
with merging and 24–27/30 without. Paired gains/losses depend on the seed and merge
rule. The full review records checks, numerical repeat limits and cohort bias.
The next separate experiment is [routing/protection](RQ2_ROUTING.md).

The [seven-question calibration passed](notes/2026-10-19-rq2-crossed-calibration-review.md):
all source/selection/replay checks and fresh answer repeats passed. Its saved
report gated the full30 panel reported above. Its literal-text
recoveries are informative, but the calibration is enriched for known E-merge
failures and does not estimate general random-versus-E performance.

This extends the passed [first merge/no-merge experiment](notes/2026-10-19-rq2-v2-full-review.md).
Use the same 30 questions / 24 images, original locked runtime, weights, prompts,
scorers and 288-token input budget. Decoder compression remains off. No native
model file or dependency changes are required.

## Main comparison

| Selected ordinary positions | Native nearest-target merging | No merging |
|---|---|---|
| Native E ranking | One native realization per image | Same native selection/basis |
| Random, seeds 101/202/303 | One realization per image and seed | Identical selected set for its paired merge condition |

The full576 baseline is also run: **nine conditions per question**, 63 primary
rows for the seven-question calibration and 270 for full30. Repeats are separate
controls. The three random seeds repeat conditions on the same questions, not
independent datasets. Summary JSON reports each seed, seed means, E-minus-random
effects under each merge rule and their interaction. TextVQA full-credit accuracy
and consensus mean score are kept separate.

All ten native FPS basis source vectors and their **original 576-coordinate
replacement slots** remain fixed. Randomize only the other 278 retained positions,
uniformly without replacement from the other 566 coordinates. A dedicated Python
RNG is seeded from a SHA256 of protocol version, image hash, archived seed and
selection-seed ID; it never advances Torch/FPS or generation RNG state.

Random subsets are ordered by descending native E with index tie-breaking before
the unchanged native basis-slot tail is appended. The selected set is uniform;
E determines this canonical routing order, not random set membership. Final model
inputs follow native boolean-mask spatial ordering. Basis **compact output
positions can change** under a different selected subset; source-to-spatial-slot
assignment stays fixed. Distinct ordinary coordinates may duplicate an FPS basis
source, as native selection permits. This tests the incremental benefit of E
selection conditional on native basis insertion, not an entirely E-free compressor.

The project-owned random hook changes only the single top-k index result called
directly by pinned native `encode_images`. Unrelated top-k calls are delegated
unchanged. It verifies the original E and native top-k first, restores on errors,
and then original FPS/reconstruction/basis replacement/merging/projection continue.
Both branches capture their first unconditional encoding. Exact source, pre-merge,
E/FPS, basis assignment and pair-selection controls must pass before interpreting
answers. Cached clone replay, same-image reuse, fixed-repeat controls and separate
fresh-native variability follow the passed v2 design. On the five declared GQA/
TextVQA controls, two additional fresh realizations are recorded for each of
native E-merge and random-101-merge. This checks fresh numerical variation under
both selectors without treating it as a new random selected set. The other two
random seeds are not given extra fresh repetitions.

## First Kaggle run

1. Commit and push all new files to `apet-analysis`.
2. Import [kaggle_rq2_crossed.ipynb](notebooks/kaggle_rq2_crossed.ipynb) into an
   editable Kaggle draft; enable Internet and T4 ×2.
3. Attach the same private **rq1-ih-inputs-v1** Dataset. No new input ZIP is needed.
4. Set only:

   ```python
   RUN_CROSSED_CALIBRATION = True
   RUN_CROSSED_FULL = False
   RUN_BROLIO_ORACLE = False
   ```

5. Run the cells and download `rq2-crossed-calibration-v1-<timestamp>.zip` and `.log`.
   Review that result before the full30 run. Original native Torch prerequisites
   and the new random/oracle tests run first, with zero allowed skips.

Execution and random-control correctness passed the Kaggle calibration and full30.
The previous v2 calibration cannot gate this experiment: it did not exercise
the random selector. Calibration uses the same seven predeclared cross-dataset
cases, including the GQA pair, Brolio and other TextVQA controls.

## Reuse an existing live notebook

After pushing, add the following cell when the original `repo`, `inputs`,
`original_python`, `setup_log`, `stream_command` and `CHECKPOINT_ARGS` variables
still exist. It loads only the new helper, refreshes runtime provenance and runs
new prerequisites without reinstalling the environment. Previous logs stay intact.

```python
import json, re, shutil, subprocess, zipfile
from datetime import datetime, timezone
from pathlib import Path

subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True, timeout=120)
updated = json.loads((repo / "research/notebooks/kaggle_rq2_crossed.ipynb").read_text())
helpers = ["".join(c["source"]) for c in updated["cells"]
           if c["cell_type"] == "code"
           and "".join(c["source"]).startswith("def verify_rq2_crossed_prerequisites(")]
if len(helpers) != 1:
    raise RuntimeError("Expected exactly one crossed prerequisite/helper cell.")
exec(compile(helpers[0], "rq2-crossed-helpers", "exec"), globals())
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
previous_setup_log = setup_log
setup_log = Path(f"/kaggle/working/rq2-crossed-setup-{stamp}.log")
shutil.copy2(previous_setup_log, setup_log)
runtime_report = Path(f"/kaggle/working/rq2-crossed-runtime-{stamp}.json")
verify_rq2_crossed_prerequisites(refresh_runtime=True)
crossed_calibration_dir = run_rq2_crossed(mode="calibration")
```

A new session must run notebook setup because virtual environments and `/tmp`
weight caches are not shared across notebook sessions. Neither path replaces the
default Kaggle kernel or requires a Transformers downgrade there.

## Full30 after calibration review

In the same live notebook:

```python
crossed_full_dir = run_rq2_crossed(
    mode="full",
    calibration_report=crossed_calibration_dir / "report.json",
)
```

The CLI verifies detailed calibration questions, all nine conditions and controls,
actual feature/evidence hashes, source/profile, runtime and checkpoint identity.
Full30 captures new first realizations; it does not choose seeds or realizations
to reproduce favorable calibration answers. In a new session, extract the accepted
calibration ZIP with all subdirectories and point `CALIBRATION_REPORT` at its
report, then enable only `RUN_CROSSED_FULL`.

Results include first captured feature banks, all chosen sets and seeds, basis
mapping, merge destinations/counts, original images and actual input canvases,
raw predictions, controls, answer CSV, HTML and source/runtime hashes. Full ZIPs
embed their prerequisite calibration evidence. Save the console `.log` as well.
Feature evidence makes archives larger; GPU feature caches are released after
each image's last question. Nothing is committed or uploaded automatically.

## Optional Brolio single-target restoration

The [one-question diagnostic passed](notes/2026-10-19-rq2-brolio-oracle-review.md):
restoring only317 recovers Brolio; sham and count-matched target141 remain Bottoli.
The recorded non-target features and mask stay exact. This is a selected
conditional rescue, not a deployable protection rule or full experiment result.

This is **separate, post-hoc oracle evidence**, not an adaptive compression method.
It does not affect the main crossed summary or satisfy its calibration gate.
Run in the same live notebook when convenient:

```python
brolio_oracle_dir = run_rq2_crossed(mode="oracle")
```

Alternatively import the same notebook with only `RUN_BROLIO_ORACLE = True`.
This one-question run captures full576, native E merge and E no-merge, then compares:

- **Sham:** clone the original native realization, with exact replay.
- **Restore317:** replace target317's merged visual feature with its pre-merge
  feature, project using the native MLP, and splice only the target's projected
  state into the native tuple. All other projected states and the mask must stay
  byte exact. A native-projector no-op must reproduce the original tuple first.
- **Matched control:** restore a different ordinary merge target with closest
  incoming count to317, breaking ties by original index. Its choice uses the
  recorded merge graph, not its new answer.

The target317 choice came from inspecting Brolio and is gold-informed. Recovery
would identify a feature change sufficient in this fixed context; failure is
also informative and must be retained. It would not establish letter-level
blurring, unique token importance, or a deployable protection rule. The matched
control may also recover the answer; do not assume it should fail.

This bounded selected panel retains earlier MMStar concerns and same-image
dependence. It supports mechanism comparisons, not population accuracy or speed
claims. Larger GQA/MMStar/TextVQA pools and routing/protection arms remain later
stages; no new dataset curation is needed now.
