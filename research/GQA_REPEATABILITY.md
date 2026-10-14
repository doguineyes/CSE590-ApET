# Native GQA repeatability check

Use [the starter notebook](notebooks/kaggle_original_gqa_repeatability.ipynb).
This is a small diagnostic experiment, not another dataset baseline. It adds a
project-owned runner and records native values; native ApET code, dependency
versions, scoring, token budgets and checkpoint revisions are unchanged.

## Cases and controls

| Question | Image | Role | Earlier pilot → development |
|---|---|---|---|
| `2058428`: Are there airplanes in this scene that are huge? | `n497658` | Answer drift | Input-only/combined No → Yes; baseline/decoder Yes → Yes |
| `201882614`: Is the lamp on the right or on the left of the photo? | `n125122` | Historically stable answer control | Baseline/decoder Left → Left; input-only/combined Right → Right |

The control is not correct under every configuration. Its usefulness here is
historical stability, including stable incorrect compressed answers. Both cases
were previously inspected and belong to development; reserved evaluation is not run.
The runner verifies exact questions, labels, raw/RGB image hashes, dimensions and
historical image seeds before loading weights.

One model session makes **72 recorded generations** plus four short warmups:

- Three consecutive counts-only repeats for each question/configuration.
- A counts-only → native-values → counts-only logging sandwich for each.
- Input-only → decoder-only → input-only, then combined → baseline → combined,
  each with counts-only and native-values logging.

Each call resets the historical Torch CPU/CUDA seed. Counts-only retains our
existing token-count hooks. Native-values additionally reads native FPS bases,
reconstruction errors, selected indices and merge assignments through the existing
return tracer. It changes recording overhead, not compression math.
Numerical/deterministic flags are recorded as found, not changed for this check.
Greedy generation uses the original prompt and unchanged 16-token maximum.

## Start on Kaggle

1. Commit and push the new runner, notebook and documentation.
2. Open the starter notebook in the Kaggle editor. Enable Internet and T4 ×2.
3. Set `RUN_REPEATABILITY_TEST = True` in the first code cell. Keep
   `MODEL_SESSIONS = 1` and `REPEATS = 3` for the first check.
4. Run the four code cells in order: update/path setup, environment check,
   repeated trials, ZIP/download. Download the ZIP before ending the session.

The flag defaults to False so automatic notebook execution stops before setup
or inference. In a fresh session, packages and weights need downloading again.
The 72 generations are a small job; downloads/loading may dominate elapsed time.

To reuse a current notebook, append the same four code cells from the starter.
Finish or stop other model processes first. Its setup command reuses the existing
original environment; model/data caches are reused when still present. The new
runner loads the model once in its own process. An earlier evaluator's finished
process cannot share its in-memory model. Different Kaggle sessions do not share
this notebook's installed environment or `/tmp` cache automatically.

`PREPARED_MANIFEST` may point to existing pilot/development data if both images
are available. Otherwise only the existing two pinned GQA parquets (~68 MB) are
downloaded and two cases are extracted. No whole GQA preparation/run is needed.
Optional `checkpoint_args` accepts existing original-format local model/CLIP paths,
as in previous notebooks; record their provenance separately.

## Read the outcome

`status = passed` means trials completed with valid native token-count/diagnostic
records. It does **not** mean the answers were repeatable or the historical
disagreement was explained. Look at:

- `findings.answer_variation_detected` and `findings.control_difference_detected`;
- `findings.native_value_difference_detected`, which can occur even when answers agree;
- per-question/configuration raw answer counts and counts by logging mode;
- logging-sandwich and configuration-return comparisons;
- native-value comparisons and fingerprints, with full arrays in the JSONL.

Answer changes are retained and the run continues. A true execution failure
preserves completed JSONL trials and available current-generation evidence. Run
the final download cell after stopping the subprocess to package partial results.
Results are never overwritten; rerunning the path-setup cell creates a new run.

If the first run is stable, it establishes stability only under this shorter
execution history. If variation appears, examine whether observed input errors,
FPS bases, selection or merging differ before changing the algorithm. Logging
parity failures need controls; they do not identify a particular faulty operation.

For an optional process-history check, set `MODEL_SESSIONS = 2` **before the first
cell**. The runner executes sequentially in two fresh model processes, reusing
packages/downloads. This makes 144 recorded generations with two loads; the
download cell compares their raw answer distributions. It does not reproduce the
full 40-/800-question historical execution sequences or every hardware condition.
Do not rerun the entire development baseline to investigate this one case.

## Evidence and local checks

Each ZIP includes JSON summaries, flushed JSONL trials, both original images and
full question metadata, source snapshots including `uv.lock`, runtime/package
identity, model device map and checkpoint revisions. Package environments and
model weights are excluded. Diagnostic timings are not speed benchmarks.

CPU protocol/orchestration checks (no local Torch required):

```bash
environments/editor/.venv/bin/python -I research/tests/test_gqa_repeatability.py
```

Optional data-only verification using previously prepared data:

```bash
environments/editor/.venv/bin/python -I scripts/repeat_original_gqa.py \
  --prepare-only \
  --data-manifest research/results/gqa-fixed-subsets-v1-s590/development/manifest.json \
  --output /tmp/apet-repeatability-data-check.json
```

Choose a new output name on later attempts. Inference verification remains on
Kaggle in the pinned original runtime.
