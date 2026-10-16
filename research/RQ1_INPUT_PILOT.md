# RQ1 input-stage pilot

This is the first contribution-measurement implementation, not a new compression
algorithm or a completed RQ1 result. Start with GQA, validate the measurement,
then expand to MMStar and TextVQA. Decoder-layer interventions come later.
All accepted datasets, native ApET files and dependency locks are preserved.

**2026-10-16 update:** the target audit confirmed a casing mismatch in our
measurement. Protocol `rq1-input-zero-v2` now renders GQA labels with an initial
capital (`no` → `No`, `stove` → `Stove`) using a fixed rule before generation.
Raw annotations stay unchanged. Native ApET, dependencies and the v1 input ZIP
remain unchanged. The corrected small calibration has now passed; proceed to
the 12-question expansion described below.

## Current next step: GQA input pilot v2

The [v2 calibration review](notes/2026-10-16-rq1-calibration-v2-review.md) confirms
target alignment and numeric controls. In the same live notebook, run this new
cell directly using the existing helper:

```python
gqa_pilot_dir = run_experiment(
    "rq1-gqa-input-pilot-v2", "gqa/manifest.json",
    ["--questions", "12", "--tokens", "576"],
)
```

This measures 12 questions on six images, all 576 patches, with checks per question.
Plan for about one hour based on calibration, with actual runtime depending on
the prompts and hardware. Download its ZIP and log before ending the session.
No new input upload, environment setup or inference-code update is needed in the
current live notebook. Keep other optional runs for after reviewing this pilot.

## Corrected calibration v2 (completed; retained for reproduction)

Push the updated scripts. In the same live Kaggle notebook, add and run this cell
after `run_experiment` has been defined (it was defined by the calibration cell):

```python
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True, timeout=120)
subprocess.run(
    [str(original_python), "-I", "research/tests/test_rq1_native.py"],
    cwd=repo, check=True,
)
calibration_v2_dir = run_experiment(
    "rq1-gqa-calibration-v2", "gqa/manifest.json",
    ["--questions", "2", "--tokens", "32"],
)
```

Keep all larger/optional runs off. The existing input bundle, environment and
checkpoint cache are reusable. The subprocess reloads the model from cache.
Download the v2 calibration ZIP/log; the old calibration/audit remain intact.
For a fresh session use `RUN_CALIBRATION=True`, `RUN_TARGET_AUDIT=False`, and keep
other run flags `False` in the updated notebook.

The fixed target is derived from the gold label only, not copied from the model's
answer or chosen per intervention. For baseline-correct questions, target token
IDs must match the generated answer prefix. On matching prefixes, teacher-forced
and native generation target log probabilities must agree within the existing
0.001-nat control tolerance. A mismatch stops before patch interventions. For
baseline-wrong cases the fixed gold target is retained; it is not replaced with
the wrong prediction to make the check pass.

V2 changes only the measured reference spelling and adds alignment controls;
E, full-context features, seed, patch order, zeroing intervention and no-op/repeat
controls are unchanged. Compare new and old results as different scoring
protocols, not as an algorithm improvement. The capitalization rule is a fixed
pilot representation, not full case/synonym marginalization or a semantic-answer
probability. Other answer-format mismatches need explicit diagnosis.

The optional `--mode target-audit` remains available to compare raw/capitalized
forms and actual native generated token IDs without interventions. Its third,
generated-response candidate is explicitly diagnostic and is never adopted as a
gold label. The completed two-question audit ran successfully in about 18 seconds.

## First Kaggle run

1. Push the new source files and
   [notebook](notebooks/kaggle_rq1_input_pilot.ipynb) to GitHub.
2. Download/copy the locally prepared
   [rq1-inputs-v1.zip](results/rq1-inputs-v1.zip) (32.6 MiB). Upload it using the
   Kaggle editor's **Upload** button, or attach it as a private Kaggle Dataset.
   Git ignores this artifact. To rebuild it on Mac, run:

   ```bash
   environments/editor/.venv/bin/python scripts/prepare_rq1_pilot.py
   ```

   The preparer refuses to overwrite an existing bundle; no rebuild is needed
   unless the accepted inputs change. The bundle holds the unchanged GQA 49,
   MMStar primary 18 and all TextVQA 100 manifests and images, with their hashes.
   The notebook also detects the extracted files if Kaggle unpacks the uploaded
   ZIP automatically. `INPUT_ZIP` can specify either a ZIP or its extracted folder.
3. Import the notebook into an editable draft; enable **Internet** and **T4 ×2**.
   Set `RUN_CALIBRATION = True`. Keep the other run flags `False`.
4. Run the setup/tests and calibration cells. The notebook leaves its kernel
   environment alone and executes our locked original runtime in subprocesses.
5. Download the `rq1-gqa-calibration-*.zip` and `.log`. Extract the ZIP and open
   `analysis/index.html`. Send the ZIP back for inspection before expanding.

Calibration uses **two GQA questions about one image × 32 sampled patches**.
It is a measurement and runtime check. The native Torch tests require Kaggle;
the Mac only runs protocol/scoring/input checks. Neither native GPU execution
nor research findings are claimed by the local implementation checks.

After calibration passes, target-format alignment is resolved, and its results make sense, set `RUN_GQA_PILOT = True`
in a new cell and run its optional GQA cell. It measures **12 questions on six
images × all 576 patches**. Set `RUN_CALIBRATION = False` if rerunning all cells.
The fixed pilot contains four loss/control image pairs and two preserved pairs;
it is deliberately outcome-enriched and does not estimate population behavior.
The excluded airplane image is absent. No further dataset curation is required.

The environment and checkpoint cache are reused in the same live session.
Each subprocess loads the model once, then releases it. This avoids another
environment install/download, while keeping model processes from accumulating
GPU memory. Fresh sessions recreate the locked environment and download missing
weights. Different notebooks do not share a live environment automatically.

## Measurement definition

For patch `i`, native ApET computes `E_i`, the L2 reconstruction residual of its
1024-dimensional CLIP feature using ten FPS basis features and epsilon `1e-5`.
The uncompressed branch skips this calculation. Our instrumentation therefore:

1. Encodes the image through native uncompressed LLaVA and captures its CLIP
   features and 576 projected visual embeddings.
2. Temporarily replays exactly those CLIP features into native input-only ApET.
   Native FPS, reconstruction, selection and merging run unchanged. We read their
   existing locals and discard the merged output. The replay and observer are
   restored after the call, including on exceptions.
3. Uses native multimodal preparation to align the full 576 visual embeddings,
   prompt, reference answer tokens, attention mask and positions.
4. Computes `S = mean log p(reference token | preceding prompt/image/reference
   tokens)` by teacher forcing, without EOS. For the singleton intervention,
   replaces one **post-projector visual embedding with a zero vector** and computes
   `I_i = S_full - S_zero(i)`. A positive value means the reference becomes less
   probable; a negative value means it becomes more probable.

Zeroing is a shape-preserving sensitivity proxy. It keeps 576 positions, token
order, positions and attention masks. It is not physical token deletion, native
ApET merging, or proof that a patch caused a generated answer. E and I are measured
at explicitly different representations. Later perturbation variants can check
whether conclusions depend on zeroing. Initial I values are singleton effects
and do not measure redundancy or interactions between multiple removed patches.

The raw canonical label stays unchanged in the input manifest. V2 applies a
fixed initial-capital rendering to GQA measurement targets and preserves MMStar
option letters. Rendered targets are materialized for every selected case before
generation; the model's prediction does not choose them. Both raw and measured
targets and the policy/version are saved. One separator space is appended after the Vicuna prompt. Tokenization
must preserve the entire prompt prefix; scoring uses the preceding logit for
each target token. Only teacher-forced measurement sees the reference. The fresh
greedy baseline generation prompt contains no answer and is also saved/scored.
A baseline-correct response alone does not establish visual grounding.

Same-image GQA questions share exact cached features, errors, FPS seed and token
indices. The 32 calibration indices are a seeded uniform prefix of a 576-index
permutation, independent of reconstruction scores. All 576 E values are saved;
unmeasured I values are `null`, not zero. The pilot defaults do not run TextVQA
interventions or decoder pruning.

## Controls and outputs

Before intervening, each question runs two unchanged reference forwards, a
clone-only no-op and a full native multimodal forward that reprojects the same
cached CLIP features. The last checks cached/native scoring agreement. The
maximum difference for any target token log probability must be at most `0.001`
nats; a failure stops the expensive loop. A final reference checks drift after
the interventions. This tolerance is a pilot acceptance threshold, not proof of
determinism. Do not silently increase it if controls fail. Effects comparable
with observed drift need caution even when a run passes.

Each result ZIP includes:

- `report.json`: runtime/packages, commit/status, source hashes, pinned checkpoint
  revisions, manifest identity, protocol, controls, runtime and peak memory.
- `questions/*.json`: raw baseline generation/scoring, exact prompt, fixed target
  token IDs, input/answer positions, reference/no-op/parity scores and perturbed
  token scores, raw/rendered target, actual generated IDs and alignment checks,
  measurement version, seed, metadata and archived outcomes. During the loop these are
  checkpointed every 16 tokens; interrupted runs can lose up to 15 token forwards.
- `features/*.json`: all native reconstruction errors, FPS basis, selection/merge
  diagnostics, feature hashes and processor configuration, grouped by image.
  Adjacent `.pt` files preserve the exact cached CLIP/projected features on CPU
  (roughly 6 MiB per image) for later audits and controlled replay.
- `analysis/tokens.jsonl`: `sample_id, image_id, token_index, reconstruction_error,
  intervention_score`, stage, reference score and measurement/control status.
- `analysis/summary.json` and `analysis/index.html`: per-question Spearman
  correlations, paired same-image sensitivity correlations, aligned patch maps,
  scatter plots and outer-quartile examples. PNG/PDF figures are included.
- Used original images, actual preprocessed 336×336 canvases, source snapshots,
  original native reference and matching `uv.lock`. Model weights are excluded.

Plots use the actual padded/preprocessed CLIP canvas, rather than stretching a
24×24 grid over the original image. Exemplars label token indices on that canvas.
High-I quadrants require positive I; low-I can include negative sensitivity.
Correlations are descriptive per question. Questions/tokens on the same image
are dependent; do not pool them as independent observations or treat this curated
subset as a population correlation/accuracy estimate.

For analysis without new inference (in the original Linux environment):

```bash
environments/original/.venv/bin/python -I scripts/analyze_rq1_input.py /path/to/extracted-run
```

Runs use new output directories and do not overwrite prior evidence. The small
pilot does not implement resume. Ordinary failures and notebook termination
attempt to save a ZIP; a killed VM/process cannot guarantee final packaging.
Download results before shutting down the session.

## Optional TextVQA baseline and MMStar check

After the GQA measurement check, the same notebook can set `RUN_TEXTVQA = True`.
This generates baseline and input-only ApET (288 visual tokens) predictions for
**all 100 retained questions**, using all ten original reference answers and the
native `TextVQAAccuracyEvaluator`. No OCR transcripts are supplied. This is
screening/model evaluation, not a TextVQA intervention run. Baseline-wrong and
partial-credit examples remain in the saved results.

TextVQA uses normalized leave-one-annotator-out consensus scores in `[0,1]`.
Reports include mean score, full-credit count and paired full-credit preservation;
the last uses baseline full-credit cases as its denominator. Paired mean-score
ratios are labeled separately. Partial/positive credit is preserved per record;
this does not reuse GQA exact matching or call partial credit fully correct.
The default is two configurations; the CLI can request all four native settings.

Once GQA measurement is validated, `RUN_MMSTAR_CHECK = True` measures three
seed-ordered examples × 32 patches from the accepted 18 baseline-correct cases.
The same CLI supports up to 12 questions and all 576 patches for this pilot.
TextVQA token interventions and a wider MMStar run follow a successful first
measurement, rather than running every dataset immediately.

Direct CLI equivalents (fresh output directory names required):

```bash
environments/original/.venv/bin/python -I scripts/run_rq1_input.py \
  --manifest /kaggle/working/rq1-inputs-v1/gqa/manifest.json \
  --output /kaggle/working/rq1-gqa-calibration \
  --questions 2 --tokens 32 --download-budget-gib 20

environments/original/.venv/bin/python -I scripts/run_rq1_input.py \
  --mode textvqa \
  --manifest /kaggle/working/rq1-inputs-v1/textvqa/manifest.json \
  --output /kaggle/working/rq1-textvqa-baseline \
  --configurations baseline input-only --download-budget-gib 20
```
