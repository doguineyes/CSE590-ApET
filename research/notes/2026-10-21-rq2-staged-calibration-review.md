# RQ2 staged calibration review — 2026-10-21

The six-question calibration passed. Both native ApET schedules and their matched
pure-random comparators execute correctly in the locked original Kaggle runtime.
The GPU can be stopped: the supplied ZIP and console log contain the evidence
needed for this review and preparation of the next runner. No calibration rerun
or further GPU experiment is needed before that implementation.

## Preserved evidence and checks

Copied these Downloads files unchanged into `research/results/`, leaving the
Downloads originals in place:

- `rq2-staged-calibration-v1-20261021T130721550505Z.zip`
  (147,265,126 bytes), SHA-256
  `8ec5c16997e80cf6051436e9c511f5f77b592441cfede9238ddd428843c3943a`.
- Matching `.log` (11,051 bytes), SHA-256
  `ef1c2ee48d5b2e8ef81afb8d029e83f1a83309a967dc27dc59ce26e629c832c4`.

Both companions are present. The archive has 163 unique members, all read fully
for CRC verification; all 162 artifact hashes, 116 evidence hashes and 21 tensor
file byte hashes verify. The eight pinned native sources and the original
environment lock match the current repository and archived evidence. Measurement
sources and the signed prerequisite evidence match too. Kaggle ran all 26
prerequisite tests with zero skips. Runtime: Python 3.10, Torch 2.1.2+cu121,
Transformers 4.37.2, FP16 LLaVA-1.5-7B across two T4s.

All six questions / five images and 54 unique primary predictions are present.
Independent CPU rescoring reproduces every recorded score and dataset summary.
The complete runner took 395.7 seconds (about 6.6 minutes), including model
loading, evidence capture, controls and export; this is not a throughput benchmark.

Verified recorded counts and selection metadata confirm:

- Input576→96 followed by 96→32 **after decoder block16**, or input576→288
  followed by 288→96 at the same boundary. Native stages use ten FPS basis
  sources, E selection, basis replacement and native merging at each stage.
  The second stage recomputes E on its current 96/288 contextualized states.
- Pure random uses exactly the same two counts and boundary, with no FPS,
  reconstruction solve, basis replacement or merging at either stage. Membership
  is reproducible from image identity and prespecified seed101/202/303.
- Native selected sets match recorded E rankings; merge assignments cover all
  dropped states. Random memberships/mask hashes reproduce exactly. Input sources,
  full projected features and native input E/FPS match the archived RQ1 references.
- All six full576 no-op controls and 18 fixed replay controls match generated
  token IDs, processed generation scores and answers exactly. Replayed decoder
  source/output/RNG fingerprints match. Both same-image GQA questions reuse the
  same input bank while computing their own decoder states.
- Cache traces have the correct stage-dependent sequence lengths and positions
  for every recorded generation. Four forced four-token probes each exercise
  three cached decoding steps. They are controls, excluded from primary scores.
- Four fresh, unmodified whole-native runs on Brolio (two per schedule) reproduce
  the corresponding primary answer and score. This is bounded evidence of
  stability, not a general guarantee against CUDA numerical variation.

These are checks of archived bytes, JSON, scoring and structural arithmetic.
No Torch installation, tensor deserialization, numerical tensor recomputation or
model inference ran on the Mac. GPU tensor assertions are retained from the
actual Kaggle execution.

## What the selected examples show

The uncompressed model gets all six correct. Correct-answer counts below are
descriptive; random seeds repeat the same questions rather than supplying new
independent observations. All TextVQA scores in this calibration are either zero
or full consensus credit.

| Schedule | Native ApET | Random s101 | Random s202 | Random s303 |
|---|---:|---:|---:|---:|
| 96→32 | 1/6 | 2/6 | 2/6 | 3/6 |
| 288→96 | 4/6 | 5/6 | 5/6 | 4/6 |

At 96→32, all three TextVQA questions fail for both methods and all random seeds.
Native also loses both GQA questions; random preserves one or two. The MMStar
soccer control remains correct in every arm. Thus aggressive staged compression
is damaging on these particular OCR examples, without an observed native
advantage over random on this selected panel.

At 288→96, both GQA questions, soccer and Dominion are correct in every arm.
Brolio and Cookery school account for the remaining differences:

| Reference | Native 96→32 | Native 288→96 | Random 288→96: s101 / s202 / s303 |
|---|---|---|---|
| Brolio | Travel | Castello di rivoli | Brolio / Brolio / Bodega |
| Dominion | Will | Dominion | Dominion / Dominion / Dominion |
| Cookery school | Cooking school | Culinary | Culinary school / Cooking school / Cooker school |

The near-name substitutions continue to motivate exact-text preservation
analysis. “Travel” and other wrong responses may also involve evidence binding
or language priors. These answers alone do not identify the responsible token,
stage or mechanism; no new causal explanation is established here.

This cohort was chosen for calibration and includes known compression failures;
it cannot establish a general ranking or reproduce the paper's accuracy table.
The staged protocol also deliberately uses clean MMStar soccer136 in place of
the earlier input-only calibration's flagged chart1014. Five questions overlap,
but six-case aggregate totals are not matched across those protocols. Random
streams are versioned separately. Keep the earlier input-only results intact
and scoped to their input boundary.

## Next implementation

The calibration removes the immediate execution blocker. Follow the
[revised evaluation plan](2026-10-20-rq2-revised-evaluation-plan.md): extend the
staged runner to the existing, fixed **TextVQA100** screening cohort, retaining
all questions and the same two schedules, baseline, three random seeds, prompts,
scoring and native sources. That is nine arms / 900 primary predictions; the
three studied OCR cases must not determine the wider selection.

The broad runner is **not yet implemented**. It needs a separate versioned
protocol/input check and calibration prerequisite, compact evidence storage,
bounded repeat/cache controls, per-seed consensus scores and baseline-correct
preservation/gains/losses. Keep partial TextVQA credit and report the complement
of the six previously studied TextVQA cases separately. Save large tensor dumps
only for declared diagnostic cases. Rerun the broad cohort's baseline within the
new run; do not substitute selected calibration or older baseline scores.

After TextVQA100, expand to GQA development and confirm on MMStar if runtime
permits. Keep reserved GQA evaluation untouched. Input-only100 remains an
optional ablation; no compressor redesign is required before the broad staged
comparison. The existing six-question notebook cannot be expanded by merely
increasing `--questions`, because its accepted identities and summaries are fixed.

The derived audit script, verification JSON, answers/budgets/fresh-repeat CSVs
and a local gallery are in
`research/results/rq2-staged-calibration-review-v1-2026-10-21/`.
Open its `gallery/index.html` to inspect the unchanged exported images and answers.
Original result files were not modified. Results remain ignored by Git.
