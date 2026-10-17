# RQ1 validity diagnostics

This bounded run checks the interpretation of our input-stage E–I measurements.
It tests image/option reliance on selected MMStar cases and compares three
feature interventions. It is project-owned instrumentation around unchanged
original ApET/LLaVA. No baseline, native source, dependency lock, accepted input
manifest or previous result is modified.

**Implementation status:** CPU protocol/export tests pass. Torch replacement and
tiny native text-only tests are included as Kaggle prerequisites. The new
pretrained diagnostics still need their first Kaggle run.

## Run in a fresh Kaggle notebook

1. Commit and push the new code, protocol, tests and notebook to `apet-analysis`.
2. Import [kaggle_rq1_validity_checks.ipynb](notebooks/kaggle_rq1_validity_checks.ipynb)
   into an **editable draft**. Enable Internet and **T4 ×2**.
3. Attach the same `rq1-inputs-v1.zip` used previously, or its extracted Kaggle
   Dataset. This ZIP is ignored by Git; pulling the repository does not download
   it. No new input ZIP, prior result ZIP or separate TextVQA audit upload is needed.
4. In the first code cell set **`RUN_VALIDATION = True`**. Keep **`MODE = "all"`**.
   Run the code cells in order. The model is loaded once for both parts. All flags
   start disabled to prevent accidental execution when importing a notebook.
5. Download **both** `rq1-validity-all-<timestamp>.zip` and its adjacent `.log`.
   Send them for review before expanding the experiment or changing the protocol.

The setup reuses the original environment and checkpoint cache in a live session;
a fresh session recreates the same locked environment. The notebook kernel stays
unchanged. The runner starts with the 78 short generations, then prints progress
for each question's 32 zeroing and 32 alternative-patch records. Each alternative
record contains both half-strength and neighbor replacement, giving **576
patch/method scores** overall. Numerical controls add a small number of forwards.
These instrumented timings are not generation-speed benchmarks.

## Existing live notebook

Finish any running subprocess first. After pushing the code, update the checkout:

```python
subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True, timeout=120)
subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True)
MODE = "all"
subprocess.run([str(original_python), "-I", "-c", "import torch; print(torch.__version__)"], check=True)
for test in [
    "research/tests/test_rq1_validation.py",
    "research/tests/test_rq1_validation_native.py",
    "research/tests/test_rq1_native.py",
]:
    subprocess.run([str(original_python), "-I", test], cwd=repo, check=True)
```

Use the **new notebook's `run_validation` helper cell**, keeping existing `repo`,
`original_python`, `inputs` and `CHECKPOINT_ARGS` variables if already defined.
That helper launches `scripts/run_rq1_validation.py`. The older `run_experiment`
helper launches a different script and should not be used for this diagnostic.
The equivalent command, from the repository root, is:

```bash
environments/original/.venv/bin/python -I scripts/run_rq1_validation.py \
  --inputs /kaggle/working/rq1-inputs-v1 \
  --mode all \
  --output /kaggle/working/rq1-validity-new-run \
  --download-budget-gib 20
```

Use a new output name. `--mode shortcuts` or `--mode perturbations` can run one
part separately if needed; `all` saves a model reload. Existing results are
never resumed or overwritten. The notebook helper captures a separate console
log; the CLI always exports a ZIP, including failed/interrupted records when
the process can exit normally. Download failure evidence before restarting.

## A. Image and option reliance: 72 trials + 6 repeats

Frozen MMStar IDs: **1014, 1030, 1188, 776, 1372, 136**. These are five previously
reviewed chart/diagram questions plus the soccer control. None is chosen using
the new diagnostic outcomes.

For each, run four cyclic option-content orders under three conditions:

| Condition | Visual input | Interpretation |
|---|---|---|
| `real` | Original image, 576 visual tokens | Normal reference behavior |
| `question_only` | No image placeholder or image tensor | Checks answering without visual input; sequence is shorter |
| `blank` | Neutral 336×336 RGB image through the normal vision encoder | Retains the visual-token/prompt structure; features are not zeroed |

Only option content moves. Original question stem, option labels, punctuation,
whitespace and answer instruction are preserved; shift zero is exactly the old
question. The correct answer **meaning** stays fixed while its displayed letter
changes. Every prediction is mapped back to its original answer content. The
gold is used only for scoring, not parsing or generation.

After the twelve trials for a case, repeat its original-image/original-order
reference once. Repeats are excluded from the 72-trial accuracy denominators.
Greedy generation, seed, original checkpoint, 16-token output budget and
uncompressed settings stay fixed. Wrong, empty or unparsed predictions remain
in the records; no cases are automatically removed or replaced.

A passed run means the trials completed. It does not mean all answers are
correct or visually grounded. Question-only success can reflect guessing,
question cues, memorization or legitimate option reasoning; it does not prove a
specific shortcut. Option sensitivity alone also does not establish its cause.

## B. Alternative interventions: six questions × 32 patches × three methods

| Dataset | Fixed case | Archived role / purpose |
|---|---|---|
| GQA | 202218829 — Window | Same-image question dependence |
| GQA | 202218593 — Bottom | Same image; both input answers preserved |
| MMStar | 895 — Bedroom | Input-compression loss |
| MMStar | 136 — Soccer | Preserved control |
| TextVQA | 35962 — Cookery school | Input-compression loss |
| TextVQA | 35670 — Dominion | Preserved control |

Each case has 32 **already frozen, unique** patch IDs:

- 16 from the previous seeded random order, including any basis/padding positions
  encountered. This is the random stratum within selected questions.
- 8 high positive archived-I, non-basis photo/mixed patches not already selected.
- 8 highest signed archived-I patches in the low-E non-basis third, photo/mixed,
  excluding earlier picks. This last stratum can include negative values; three
  paper-towel patches do. It is not filtered to make an interpretation favorable.

The two diagnostic strata use previously observed outcomes and are post-hoc.
They and the random stratum are summarized separately. Combined correlations
are conditional descriptions of this selected panel, not population estimates.
No significance tests treat patches as independent images.

For each patch, on the same uncompressed full-context representation:

| Method | Replacement |
|---|---|
| `zero` | Zero vector, reproducing the existing intervention |
| `half` | 0.5 × the original projected feature |
| `neighbor_mean` | FP32 mean of the original immediate up/down/left/right features, cast back to FP16 |

Neighbors never wrap across rows or use previously modified features. At a
photo boundary, a neighborhood can mix photo and padding; patch location and
neighbor IDs remain recorded. Sequence length/order, positions, mask, answer
targets and teacher-forced scoring stay fixed. The answer is not supplied to the
free-generation prompt. I remains signed mean-reference-log-probability change,
without EOS. E is read from native ApET on the same cached full CLIP features.

Half-strength is a dose check along the zeroing direction, not independent
confirmation. Neighbor replacement changes the intervention direction but can
preserve needed information. Their perturbation L2 norms are recorded; equal I
magnitudes are not expected. Rankings, signs and E–I correlations are useful
descriptive comparisons, not a definition of ground-truth importance.

Before alternative scoring, the runner checks a fresh correct baseline, fixed
target IDs, repeated reference, clone no-op, cached/native parity and a
strength-zero sham for every method. It repeats the reference after all methods.
TextVQA retains its full-teacher-forced/first-token control policy; later cached
generation gaps remain diagnostic. Failed controls stop the run and preserve
evidence; there is no target substitution or automatic tolerance increase.
Historical feature hashes and selected E/I differences are recorded separately
from within-run controls, without silently treating cross-run differences as
new scientific effects. Original TextVQA screening roles are overlaid from their
existing frozen profile; the v1 input manifest itself stays unchanged.

## Results and provenance

The ZIP retains the profile, original accepted manifests, source/lock hashes,
native implementation reference, model/runtime metadata, raw trials and prompts,
semantic option mappings, free answers, target tokens, controls, per-patch scores,
neighbor IDs and perturbation norms. Small current feature snapshots are kept
for intervention audit, not model weights. A local `index.html` gives a compact
summary; `perturbations.csv` contains every completed patch/method observation.

The tracked [profile](protocols/rq1_validation_v1.json) embeds the necessary prior
E/I values and source hashes, so prior result ZIPs need not be uploaded to Kaggle.
[prepare_rq1_validation.py](../scripts/prepare_rq1_validation.py) documents how
it was constructed from archived results and refuses to overwrite it. This
profile supports validation only; it does not replace the accepted RQ1 datasets.

Review these results before deciding on one more bounded check or proceeding to
layer 16. No large baseline rerun, new dataset curation, compression refactor or
resumption of the deferred airplane numerical study is required here.
