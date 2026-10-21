# RQ2: staged native ApET versus pure random on TextVQA100

The [six-question staged calibration passed](notes/2026-10-21-rq2-staged-calibration-review.md).
This is its broader comparison on **all 100 existing TextVQA screening questions**.
Use this notebook next; the earlier input-only100 experiment remains optional.

## Conditions and scoring

| Method | Input states | States after block16 | Replicates |
|---|---:|---:|---:|
| Uncompressed LLaVA | 576 | 576 | 1 |
| Original native ApET | 96 | 32 | 1 |
| Pure random | 96 | 32 | 101 /202 /303 |
| Original native ApET | 288 | 96 | 1 |
| Pure random | 288 | 96 | 101 /202 /303 |

**900 primary predictions**, one model load, sequential conditions. The input and
layer16 adapter, original model forward/cache, native source files and runtime
lock are unchanged from calibration. Native uses FPS/E/basis replacement/merging
at both stages; pure random uses no such operations. It gathers the original
states at each boundary, preserving sequence order. Random membership uses the
same image/seed/stage streams as calibration. Native post-input RNG continuation
is replayed exactly before decoder FPS. Second-stage states are contextualized.

Keep all questions, including baseline-wrong cases. A fresh baseline runs for
every question. Existing prompts, all ten reference answers, image preprocessing,
greedy cached generation and maximum16 new tokens remain fixed. Consensus mean
is the main metric; partial credit, full-credit preservation (APR), gains/losses,
per-seed values and paired differences are retained. Summary groups are all100,
the previous six diagnostic cases, and their **94-question complement**.

Paired percentile bootstrap intervals resample question/image units (100 distinct
images), averaging random seeds within each question before estimating the
seed-mean difference. They condition on the three prespecified seeds. Seeds do
not increase the question count. The 94-question complement remains development
data, not an independent held-out set.

Whole native ApET versus pure random compares complete procedures. It cannot
attribute differences solely to E selection, FPS or merging. Matching schedules
does not establish equal runtime. This is not full-TextVQA or a complete paper
reproduction. No minimum accuracy determines whether execution passes.

## Kaggle steps

1. Commit and push the new files to `apet-analysis`.
2. Import [kaggle_rq2_staged_textvqa100.ipynb](notebooks/kaggle_rq2_staged_textvqa100.ipynb)
   into an editable Kaggle draft. Enable **Internet and T4 ×2**.
3. Attach the same private **rq1-ih-inputs-v1** Dataset. It already contains all100
   images; no new Dataset upload or calibration-results ZIP is required.
4. Set the first cell's flag:

   ```python
   RUN_STAGED_TEXTVQA100 = True
   ```

5. Run cells in order: fetch code → locate inputs/setup locked environment →
   prerequisites → full100 → download. There is no need to rerun the pretrained
   six-question calibration. The notebook's prerequisite tests are tiny native
   fixtures and require **zero skips** before loading model weights.
6. Download `rq2-staged-textvqa100-v1-<timestamp>.zip` **and** its matching `.log`.
   Then stop the GPU. Keep failed/partial exports too if a run stops early.

The default False flag prevents automatic inference on notebook import. The
runner validates the complete cohort and current source/runtime proof before
loading pretrained weights. A compact, hash-pinned reference to the reviewed
calibration travels with Git; original calibration ZIP/log remain separately
archived. If calibrated operator sources change, that reference gate fails.

The full100 duration has not been measured. The earlier 6.6-minute calibration
included controls, loading and evidence work, so its timing is not a reliable
linear prediction. Progress prints one completed question at a time. Independent
Kaggle sessions install their own environment and download their own checkpoints;
reuse is within a live session. The original runtime leaves Kaggle's notebook
kernel unchanged.

For an existing live session, import this notebook as a draft and run its setup
cells; the setup script reuses the existing original virtual environment and
checkpoint cache in that same session. A fresh session needs the full setup.

## Controls and compact evidence

In addition to900 primary predictions:

- Six full576 identity controls on the previously studied six questions.
- Eighteen exact clone replays: all nine arms on37109 (a historically wrong
  baseline) and37533 (Brolio). Each compressed replay uses the same per-question
  decoder snapshot and source/RNG checks as the calibration.
- Four forced four-token cache probes on35670 (Dominion), native/random101 at
  both schedules. Each tests three cached decoding steps.
- Four fresh whole-native generations on Brolio, two per schedule, without input
  replay or replacement prune. Record variation; retain first primary answers.

Total: **932 generations**, with controls excluded from all primary denominators.
Every primary generation records cache geometry. Input E/FPS and sources are
checked across conditions; random masks match the calibrated policy exactly.
There is no accuracy gate or requirement that fresh independent CUDA merges be
bitwise identical.

The ZIP includes raw JSONL, CSV, a local image/answer gallery, all question
records,100 input-encoding records, native E/basis/merge graphs, random memberships,
input/decoder/RNG fingerprints, cache traces, historical baseline drift, source
snapshots, environment lock, setup/runtime/prerequisite proof, and file hashes.
Input diagnostics live once per image and are referenced by answer rows. Full
input/decoder tensors are saved only for Brolio and Dominion (18 tensor files),
instead of every image/condition. Packaging size still depends on actual outputs.

On interruption, complete matched questions form a descriptive partial summary;
unfinished arms stay in raw records, are excluded from that summary, and execution
remains interrupted/failed. Choose a new output path when rerunning. This version
does not resume a partial run automatically.

## Local checks

```bash
environments/editor/.venv/bin/python -m unittest \
  research.tests.test_rq2_staged_textvqa_protocol \
  research.tests.test_rq2_staged_textvqa_runner -v
```

No Torch installation or model execution is needed on the Intel Mac. The actual
native adapter tests execute inside the original Kaggle environment. Broader
pretrained results remain pending until this notebook runs there. Previous
input-only protocols, native sources and saved results are preserved.
