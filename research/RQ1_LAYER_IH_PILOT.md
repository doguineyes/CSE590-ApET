# RQ1: bounded zeroing / IH layer comparison

**The new six-question × 32-position calibration passed on Kaggle.** All IH
reference/no-op/retention repeats have zero drift, and all 384 zeroing effects
exactly reproduce the earlier calibration. See the
[results review](notes/2026-10-17-rq1-layer-ih-calibration-review.md).
The same-panel full comparison is ready; the new report is its prerequisite.
For future changes to the measurement, calibrate again before expanding.

## What this comparison answers

Keep the existing input and layer-16 singleton-zeroing measurements unchanged.
Add a second family only at the **same boundary after block 16, before block 17**:

- Zeroing: how does changing one state affect the answer with all the others present?
- IH retain-one: how much does one retained state change the answer relative to
  removing every visual state at that boundary?

Both are operational counterfactuals. Neither is intrinsic token importance.
The earlier full run might resemble the calibration, but the calibration cannot
establish the unseen positions. We do not assume agreement or disagreement of
the new IH ranking before testing it.

## Source checked, and adaptations

Official repository: [Information-Horizon](https://github.com/YahongWang1/Information-Horizon),
commit `75909b2936a13d7214e83514f0bc0cb9cc91139e`.
The [source-pinned profile](protocols/rq1_layer16_ih_v1.json) records four source
hashes and the differences from the authors' implementation. No IH package,
model fork, FlashAttention install or environment change is needed.

The [paper's section 3.2](https://arxiv.org/html/2512.07580v1#S3.SS2) expresses
retain-one using a binary visual mask and defines information as the change in
the first ground-truth tokenizer token's **probability**, relative to all visual
states being absent at that layer. The inspected local CVPR PDF gives the same
retain-one / all-absent probability definition.

The [released implementation](https://github.com/YahongWang1/Information-Horizon/blob/75909b2936a13d7214e83514f0bc0cb9cc91139e/llava/model/language_model/modeling_llama_self.py#L517)
physically shortens hidden states, preserves original retained position IDs, and
exports first-step logits. Its [example script](https://github.com/YahongWang1/Information-Horizon/blob/75909b2936a13d7214e83514f0bc0cb9cc91139e/scripts/v1_5/cal_info/textvqa.sh)
defaults to `SIZE=2` (2×2 regions). We use the supported **SIZE=1** setting to
match our singleton unit. This is a source-informed project implementation of
the released operator inside ApET, not an exact reproduction of the authors'
benchmarks, preprocessing, kernel, answer rendering or default region size.

The released zero-based pruning index `K=16` means before module 16 (block 17),
matching our after-block-16 boundary. All earlier native layers run with the full
image and text. We preserve all nonvisual states and every retained original
RoPE position. Temporary pre-hooks adapt the shortened causal mask and native
RoPE lookup extent for ApET's older SDPA implementation. They are removed even
on failure. Native model source files, weights, pruning configuration and lock
stay unchanged. No independently rewritten decoder suffix is used.

## Scores and controls

Use the same frozen six cases, initial-capital gold rendering, token IDs, random
order and 32 positions as the original layer calibration. No outcome-based
replacement, retargeting or new dataset selection.

Historical zeroing score remains:

`I_zero_whole = mean_log_p(full answer | full) − mean_log_p(full answer | zero one)`.

For the direct comparison, save both in **first-token probability units**:

`I_zero_first = p(y1 | full) − p(y1 | zero one)`.

`I_IH = p(y1 | retain one) − p(y1 | remove all visual states at layer 16)`.

The comparison uses the same teacher-forced full forwards, targets and captured
upstream states as zeroing, with causal label positions remapped after deletion.
The first-token prediction has no preceding gold-answer token; future gold
tokens cannot affect it causally. Later-token scores are retained as raw
diagnostics, but do not redefine IH's first-token probability metric. First
subwords such as `Cook` or `Domin` need not be a complete semantic answer.

Controls require fresh correct/target-aligned baselines and all existing zeroing
controls; exact equality of the uncompressed pre-retention boundary; keep-all
no-op parity; repeated remove-all scores; repeated first retained case; final
full-reference scores and boundary hashes. The frozen 0.001 per-answer-token
control tolerance is unchanged. Failure stops with partial evidence. Small
effects remain subject to numerical resolution even when reference repeats match.

The all-absent reference is **not a question-only model run**: nonvisual states
already contain image information from the first 16 blocks. Also, the two families
change both retained context and zeroing/deletion operator. Disagreement cannot
by itself identify which difference caused it. There is no claim that standalone
contribution, conditional necessity and native ApET compression are equivalent.

## Run in a new notebook

Push the changes to GitHub. Import the updated
[layer notebook](notebooks/kaggle_rq1_layer16_pilot.ipynb), attach the **same**
`rq1-inputs-v1` private dataset, enable Internet and T4 ×2, and set:

```python
RUN_LAYER_CALIBRATION = False
RUN_IH_CALIBRATION = True
RUN_FULL_LAYER_SWEEP = False
RUN_FULL_IH_SWEEP = False
```

Run cells in order, including tiny native prerequisites. A single model load
first completes the unchanged zeroing panel, then the IH measurements. The
original environment/cache is reused. The calibration has 384 old patch/stage
scores plus 192 layer retain-one scores = **576**, plus null/reference controls.
Allow roughly **8–15 minutes** in a warm session; this is a provisional estimate.

## Continue in the existing live notebook

After pushing, finish any active process, then run this cell. Existing `repo`,
`original_python`, `inputs`, `CHECKPOINT_ARGS`, `datetime` and `timezone` are reused.
Pulling GitHub does not replace a function already defined in notebook memory,
so explicitly reload just the tracked helper definition:

```python
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True, timeout=120)
subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True)
subprocess.run([str(original_python), "-X", "utf8", "-I", "-c", "import torch; print(torch.__version__)"], check=True)
for test in ["research/tests/test_rq1_layer_ih.py", "research/tests/test_rq1_layer_ih_native.py"]:
    subprocess.run([str(original_python), "-X", "utf8", "-I", test], cwd=repo, check=True)
notebook = json.loads((repo / "research/notebooks/kaggle_rq1_layer16_pilot.ipynb").read_text(encoding="utf-8"))
helper = next("".join(c["source"]) for c in notebook["cells"]
              if "".join(c["source"]).startswith("def run_layer("))
exec(compile(helper, "tracked layer notebook helper", "exec"), globals())
layer_ih_calibration_dir = run_layer(include_ih=True)
```

Download `rq1-layer16-ih-calibration-v1-<timestamp>.zip` **and** its `.log`.
The ZIP contains the unchanged raw zeroing records, new IH scores and null
references, retained original position indices, controls, native E/basis/merge
observations, feature snapshots and source/lock/protocol hashes. `questions.csv`
and `ih-comparison.csv` compare E–zero-first, E–IH-first and zero-first–IH-first
rankings, including basis-excluded results and padding labels.

## Full run after review

Review the **new** calibration controls, effect sizes and ranking agreement first.
Successful execution alone does not determine whether more positions are useful.
The full-run gate rejects the old zero-only report and changed measurement
sources/profile/lock/models.

```python
layer_ih_full_dir = run_layer(
    tokens=576, calibration=layer_ih_calibration_dir, include_ih=True,
)
```

That expands the same six questions to 6,912 zeroing scores plus 3,456 retain-one
scores = **10,368**, plus controls. IH is only at layer 16, not the input stage.
Budget provisionally **90–120 minutes**, then replace this estimate with timings
from the new calibration. No full run starts automatically. Keep datasets,
native algorithms and reference scoring fixed; a complete information-horizon
curve or general population claim is outside this small comparison.
