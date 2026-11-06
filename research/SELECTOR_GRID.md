# No-basis selector grid v1 — November 6

## Purpose and interpretation

Run GPU experiments while answer boxes are reviewed independently. No arm uses OCR,
reference answers, predicted correctness, or an answer-sensitivity map for selection.
This is a new **no-basis research family**, not a faithful reimplementation of native
ApET. The original native method and previous fixed-basis experiments remain unchanged.

Same 96→32 schedule, after block16, checkpoint/prompt/scorer and fixed development
cohorts. The core compares random, nonpadding random, pixel detail, and ApET E scores,
with/without averaging into all kept targets. A fresh native arm remains the anchor.

Do not interpret these comparisons too narrowly:
- Nonpad vs random tests the operational padding-aware selection policy, with changes
  at both cuts. CLIP padding states may carry contextual information.
- Detail vs nonpad is not a perfect removal of padding confounding: partial patches
  and the two policies' eligibility/ranking differ. Check saved padding shares.
- E-drop vs native changes both basis insertion and merging (and later hidden states).
  It does not isolate either component.
- Merge vs drop shares input masks. For E, stage2 E/FPS and selected masks can change
  after stage1 merging. We explicitly record decoder overlap. This is a total
  pipeline merge effect, not a fixed-mask intervention at both stages.
- E/norm is a residual ratio under ridge regularization, not exactly sin(theta).
  The existing negative E–norm association does not establish directional novelty
  as the causal mechanism. Final-position coverage association likewise does not
  establish that late access is necessary; force-in/out waits for reviewed boxes.

## Family A: TextVQA100

| Selector | Drop seeds | Merge seeds |
|---|---|---|
| Pure random |101,202,303,404,505 (fresh references) |same5 |
| Nonpadding random |101 through1010, increments101 (10 seeds) |first5 |
| Detail |deterministic |deterministic |
| E |one fixed image-conditioned FPS seed |same FPS RNG policy |

Plus fresh full576 and native96→32 = **31 conditions**. Calibration uses the first5
frozen manifest questions,155 primary predictions. Full uses all100,3,100 primaries.
Match first5 seeds when calculating paired merge effects; don't compare a10-seed
average against a5-seed average as though they were paired. Existing20 random seeds
remain historical evidence, not additional independent questions in this run.

## Exact implementation choices

- Input features: raw CLIP patches before nonlinear projector. Drop preserves the
  original projected vectors. Merge changes raw CLIP vectors, then calls the projector.
- No-basis E: input E read from the actual native scorer on identical raw features;
  top96 chosen without overwriting slots. At stage2, native FPS10 and ridge solve in
  fp32 score current96 hidden states; top32 survive without basis insertion.
- Detail v2: shared CPU/GPU helper `detail_without_padding`. Compute within-patch
  mean absolute normalized-pixel gradients only where **both pixel footprints** lie
  fully within geometric image bounds; each direction averaged over valid edges;
  zero when none. This differs from the earlier exploratory detail map. The portable
  CPU reference was regenerated with this exact metric from the saved seed-run pixels.
- Nonpadding: padding fraction≤1e−9 first. Seeded random tie ordering; if insufficient,
  fill ascending padding fraction. Square images exactly reproduce random masks.
- Detail/E ties: ascending original/compact position. Random uses the original hash
  namespace and permutation algorithm. Stage2 padding/detail use inherited target
  original positions, including merged arms; this is not a group-content score.
- Merge: every selected token is a target, sorted by position. All unselected tokens
  route to maximum cosine similarity; ties choose first target. Average own+sources.
  Stage1 uses fp16 arithmetic, stage2 fp32 then cast back, following native stage
  precisions. No basis special cases, no proportional attention; never pass k=0 to
  native merging. Save assignment, count and own-share for each target.
- Native/reference numerical variation remains possible. Preserve the first result,
  record historical drift; do not rerun to select a favorable answer.

## Notebook and gates

`research/notebooks/kaggle_selector_grid.ipynb`

1. Push code/reference/protocol/test files; import notebook into Kaggle.
2. Attach existing **rq1-ih-inputs-v1**, Internet, T4×2.
3. Set `RUN_SELECTOR_CALIBRATION = True`. Run setup, prerequisites, calibration.
4. Prerequisites require zero skips: actual tiny original-LLaVA tests cover all8
   selector/compensation types, frozen replay, pure-random score equivalence and
   all-target merge averaging. CPU tests cover padding transitions/fallback, ties,
   historical masks, square-image identity and portable reference membership.
5. Download calibration ZIP/log for review before enabling `RUN_FULL_TEXTVQA`.
   Pretrained gates check exact saved-feature identity, CPU/GPU stage1 top96 detail/E
   equality, original random mask equality,96/32 geometry, no-basis membership,
   identity576, fixed replay and forced cache probes. Pixel detail values allow1e−12
   floating summation tolerance; selected indices must match exactly.
6. Set `RUN_FULL_TEXTVQA = True` and run its cell after review. Keep the same session.

Don't rerun model setup/install for each arm. Each invocation loads the model once.
Prior rate suggests roughly40 minutes for3,100 predictions, but merge instrumentation
and checks add overhead: use calibration timing instead of promising completion
before the quota reset. Do not lower validation standards to consume credits.

## Family C: optional GQA replication

The same notebook includes deferred GQA setup. Set `PREPARE_GQA = True` to reuse or
recreate the unchanged development800 study. After TextVQA review, set `RUN_GQA = True`.
Default `GQA_SHARDS_TO_RUN=[0]`:25 complete images ×8 questions =200 questions. Later
run disjoint[1],[2],[3]. All14 conditions per shard =2,800 predictions (11,200 overall).
Includes native/full576, E-drop, detail-drop,5 nonpad-drop and5 random-drop seeds.
Use the passed TextVQA grid calibration report; source hashes must match. GQA tests
validate original cohort/scorer; summaries need image-clustered uncertainty because
questions share images. Never interpret one200-question shard as full800.

GQA currently repeats input capture per question (same image-conditioned seeds),
so do not assume the old cached-image timing. All same-image masks and results remain
available to check later. Family B localization and force-in/out are not implemented
here; don't add answer-targeted arms until box quality is reviewed.

## Archives

ZIP contains source and protocol snapshots, lock/runtime/prerequisite evidence,
original images/model canvases, per-image padding/detail/E maps, per-question full
predictions/scoring, input+decoder indices, input fingerprints, merge routes/counts,
controls, historical drift and initial arm summaries. Large feature banks are not
repeated for every arm. Decode positions through saved stage1 order. Random/drop
positions and merged target positions must not be interpreted as identical evidence.

Download ZIP **and** matching log before closing. Failure exports are kept as partial
records; fresh output paths prevent overwrite. Aggregate only disjoint complete GQA
shards and keep paired seeds/question identities.
