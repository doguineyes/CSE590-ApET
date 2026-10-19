# RQ2: native input merging pilot

This first experiment compares generated answers on the **same 30-question RQ1
panel**: GQA12, MMStar12 and TextVQA6. It is a diagnostic cohort, not a population
accuracy sample. Existing MMStar quality/grounding concerns remain in the panel.

| Arm | Visual inputs | What changes |
|---|---:|---|
| Baseline | 576 | Uncompressed original LLaVA |
| Native input ApET | 288 | Original E selection, 10 FPS basis replacements and similarity merging |
| Native selection+basis, no merge | 288 | The same selection and basis assignment; ordinary target averaging is omitted |

Decoder compression is disabled. The no-merge arm is a **project-owned ablation**;
it keeps native basis replacement and is not basis-free selection-only. The
original ApET files and original locked environment stay unchanged. Native parity,
selection/slot identity, repeated references and saved feature fingerprints must
pass before the comparison is interpreted.

## Run on Kaggle

1. Push these new files to `apet-analysis`.
2. Import [kaggle_rq2_input_merge_pilot.ipynb](notebooks/kaggle_rq2_input_merge_pilot.ipynb)
   into an **editable draft**. Enable **Internet and T4 ×2**.
3. Attach the **existing private `rq1-ih-inputs-v1` Dataset** used for the input-IH
   run. It contains [rq1-ih-inputs-v1.zip](results/rq1-ih-inputs-v1.zip), **34.1 MiB**,
   or its automatically extracted contents. No new bundle or public upload is
   required. The older `rq1-inputs-v1` bundle alone is insufficient.
4. In the first code cell set:

   ```python
   RUN_RQ2_PILOT = True
   ```

   Leave `QUESTIONS = 30`. `INPUT_ZIP = None` detects a ZIP or unpacked Dataset.
   If multiple copies are attached, set it to the exact ZIP path or the extracted
   folder containing `references/` and `protocol.json`.
5. Run the cells in order: code update, inputs/environment/native tests, helper,
   pilot, downloads. The setup reuses the existing original environment and
   checkpoint cache in a live session. It does not replace the notebook kernel,
   install packages into Kaggle's default Python, or require a restart.
6. Download the **result ZIP and matching `.log`**. The setup/native-test log and
   runtime JSON are also linked; both are copied into the runner's evidence ZIP.
   Send the result ZIP/log for review before expanding RQ2 variants or datasets.

The flag defaults to False and stops before cloning, setup or inference. Importing
the notebook must not start an experiment. If Kaggle imports it as a running
version, open **Edit** to make the change in a draft.

Optional `CHECKPOINT_ARGS` accepts existing original-format model snapshots:

```python
CHECKPOINT_ARGS = [
    "--model", "/kaggle/input/your-llava-snapshot",
    "--vision-model", "/kaggle/input/your-clip-snapshot",
]
```

Otherwise the runner reuses/downloads the same checkpoints as previous original
ApET runs. Do not pass the Hugging Face converted model used by the earlier
extracted-compression pipeline. The default original-model cache is outside
`/kaggle/working`; a new Kaggle session may still need to download it again.

## What to inspect

The pilot uses one loaded model for all three arms. Read actual generated answers
and dataset-appropriate correctness, then paired preservation, gains and losses.
Reference-answer scores are secondary diagnostics; an IH or zeroing score is not
the outcome metric for this experiment.

Check the three literal-text losses **Brolio, Cookery school and Give up**, together
with preserved controls **Dominion, Sky mall and Laughter**. Follow source IDs,
ordered selection, basis source-to-output slots and incoming merge assignments.
For GQA, compare the two questions on each shared image. Any rescue must be
reported alongside newly introduced failures across the whole fixed cohort.

The native arm must first reproduce its own controls and satisfy the reported
parity checks. A mismatch is evidence to review, not permission to silently change
answers, seeds, tolerance or the cohort. A previous correct answer is not a reason
to exclude a new wrong answer.

## Outputs and reuse

Each invocation creates a new `rq2-input-merge-pilot-v1-<UTC timestamp>` directory
and a sibling ZIP/log under `/kaggle/working`. Normal handled runner failures also
write evidence. If the process exits before creating its archive, the notebook
creates a clearly labeled fallback ZIP of whatever evidence exists; that fallback
does **not** establish that the run completed. A hard kernel/session termination
can prevent this cleanup, so download outputs before stopping Kaggle.

The wrapper leaves the runner's complete ZIP untouched. It separately streams the
console to a UTF-8 log. Setup and tiny native-test output are preserved and passed
to the runner with `--setup-log` and `--runtime-report` for archival provenance.
No model weights or environment directories are bundled.

For another run in the **same live notebook**, after the first process ends:

```python
rq2_repeat_dir = run_rq2_pilot()
```

This creates a fresh output directory and reloads the model, while reusing the
environment, accepted inputs and checkpoint cache. The pilot is not resumable;
do not overwrite an existing run. Do not pull Git or run another model subprocess
during inference. Separate Kaggle notebooks have separate sessions and should
each run setup; attaching the same input Dataset does not share a virtualenv.

## Validation and next scope

Kaggle setup explicitly imports Torch, then executes
`python -m unittest research.tests.test_rq2_native -v`. A zero-test run or any skipped
native test fails the notebook prerequisite. The native tests use tiny models and
do not download pretrained weights. Pretrained parity and answer checks still
belong to the actual GPU pilot; local CPU checks cannot establish those results.

After reviewing this pilot, the [RQ2 plan](notes/2026-10-19-rq2-plan.md) adds random
selection, random merge destinations and protected targets in stages. They are not
implemented as arms in this first notebook. No new dataset curation or RQ1 sweep
is needed to run this pilot.
