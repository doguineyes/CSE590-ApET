# TextVQA random-seed experiment v1

## Question and scope

Does the chance retention of answer-region positions predict generated-answer credit
within a question? This is a first mechanism test, not an improvement proposal.
Keep the original frozen TextVQA100 (including baseline-wrong cases), original runtime,
prompt, scorer, 96→32 schedule and native layer16 boundary. Do not extend the cohort yet.

The November 5 note calls the old seeds 1–3; their actual values are 101, 202, 303.
Use twenty seeds 101×i, i=1…20. The original mask hash namespace and permutation
algorithm are unchanged. All 300 archived random masks match exactly in CPU tests.
The new policy is scoped to the existing staged adapter, without changing any archived
protocol, native `llava/` code, or environment lock.

## Kaggle

Upload `research/notebooks/kaggle_textvqa_seed_coverage.ipynb`, attach the existing
`rq1-ih-inputs-v1` Dataset, enable Internet and T4 ×2.

1. Set `RUN_SEED_CALIBRATION = True`; run setup and prerequisites.
2. Run calibration (first three manifest questions × six conditions = 18 primary
   predictions). Controls: 576 identity, seed404 exact frozen replay and forced
   four-token cache probe. Native tiny-model tests must run with zero skips.
3. Download calibration ZIP + log for review. Check execution controls, counts,
   historical mask equality and prediction drift. Drift is recorded, never repaired
   by picking a preferred repeated answer.
4. Once reviewed, set `RUN_FULL_SEED_TEST = True`; run that cell in the same session.
   Full run: 100 ×22 conditions =2,200 predictions plus three controls. No resumption
   into an old directory: fresh paths preserve incomplete runs. Failed runs export
   their partial records too.
5. Download the full ZIP + log. Historical seed predictions are reference repeats;
   use the new run once per question/seed, not old+new as extra observations.

The old 900-prediction TextVQA run took 965 seconds. Linear extrapolation for 2,200
predictions is about 39 minutes, excluding setup; allow roughly 45–70 minutes until
calibration gives a better estimate. This is planning guidance, not a speed benchmark.

## Saved evidence

- Per-question/per-seed generated answers, token IDs, consensus/full-credit scoring,
  prompts, exact stage1 indices, compact stage2 indices and original-position mapping.
- Fresh baseline/native reference and comparisons with the archived October21 run.
- For each native image: actual normalized pixels, **pre-merge** CLIP patch features,
  native 96-state layer16 features, E/FPS indices and native merge records at both stages.
- Original images, model canvases, source hashes/source copies, manifest, lock,
  setup/runtime/prerequisite evidence, artifact checksums. No model weights in ZIP.

This implements the first TextVQA experiment plus its TextVQA feature preparation;
GQA800 feature dumping and the selector×compensation grid are deferred.

## What remains before coverage analysis

The current manifest has answers and images but **no OCR word boxes**. Obtain and
verify matching OCR annotations before calculating answer regions; audit mappings
visually. Do not manufacture boxes from the answer string. Record unmatched answers,
multiple matches and partial matches separately. Geometry should follow the actual
pad-to-square/resize operation, not a threshold on normalized pixel intensity.

Report stage1 and final-position coverage separately. CLIP and layer16 states are
contextualized: a retained position can carry information from other patches. Native
basis replacement also moves content to other slots; merge membership means a path
for information, not proof that the original evidence remains readable.

Within-question coverage differences reduce question-difficulty confounding but
random pruning changes other retained content too. A coverage–correctness association
is a mechanism candidate, not isolated causal mediation. Start with paired descriptive
comparisons (correct vs wrong seeds), image-clustered uncertainty, and report how many
questions vary across seeds. All-correct/all-wrong questions provide no within-question
binary contrast. Do not treat 2,000 rows as 2,000 independent questions.
