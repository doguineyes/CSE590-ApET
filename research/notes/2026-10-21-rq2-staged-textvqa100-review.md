# Staged TextVQA100 results — 2026-10-21

The complete run passed. On this fixed 100-question screening cohort, native ApET
outperforms every pure-random seed at both matched schedules. Its advantage is
larger under 96→32. This is stronger evidence for the native procedure than the
earlier selected diagnostic examples supplied. Exact-text failures still occur.
No execution fix or TextVQA rerun is required before starting the prepared GQA
development experiment.

## Archived and verified

Copied both supplied Downloads files unchanged to `research/results/`; Downloads
originals remain in place. Both companions are present:

- `rq2-staged-textvqa100-v1-20261021T134751714241Z.zip`, 137,412,185 bytes
  (131.0 MiB), SHA-256
  `52e593236faac3f22bd20da85a1dcf7e8f0508d7dd1ff0a036e176fca8089f76`.
- Matching `.log`, 150,831 bytes, SHA-256
  `1ebf5e7d10486eb49d43fee605435d390155d24107f9cf2187d5283d62634909`.

All 488 unique ZIP members pass full CRC reading; 487 artifact hashes, 64 evidence
hashes and 18 tensor-file byte hashes verify. All 100 encoding and 100 question
records verify. The 52 measurement sources match archived snapshots and the
current repository. The eight native source pins and original uv lock match the
reference. The run used clean commit
`0f02634d6e79df5def3d24bcd19d49c0708b1308`, Python 3.10.18, Torch 2.1.2+cu121,
Transformers 4.37.2, native FP16 LLaVA-1.5-7B across two T4s.

All 900 primary predictions rescore exactly using the fixed ten-annotator
TextVQA consensus scorer. All three summary groups and paired intervals reproduce.
The 100 questions/images/labels/seeds match the frozen manifest. The uncompressed
baseline reproduces **all 100 historical raw answers and scores**, not just the
aggregate mean. All 27 overlapping predictions from the earlier three-TextVQA
staged calibration reproduce generated IDs, answers and consensus scores; their
CLIP sources and native input E/FPS also match.

Kaggle passed 40 current prerequisites with zero skips. Six full 576 no-op controls
and 18 exact frozen replays match generated IDs, processed generation scores and
answers. Replayed decoder source/output/RNG fingerprints match. Four forced
four-token probes each verify three cached decoding steps. Four fresh whole-native
Brolio runs reproduce the corresponding primary IDs and scores. These bounded
fresh controls do not guarantee universal CUDA bitwise determinism.

Actual 96→32/288→96 counts after block 16, mask geometry, native E-selected sets,
ten-source FPS bases, complete merge assignments, random memberships, original
state preservation flags and cache positions all verify. Pure random runs no
FPS/E/basis replacement/merge at either boundary. There are no empty answers or
outputs reaching the 16-token cap in any primary arm. No control is counted in
primary denominators: total execution is 932 generations.

Recorded runner wall time is 965.2 seconds (**16.1 minutes**), including checkpoint
work, model loading, inference, controls and export; environment setup is separate.
Model loading reports 24.0 seconds. Peak allocated memory is 7.822 GiB on GPU 0 and
7.317 GiB on GPU 1. These observations are not controlled latency/speedup estimates.

Review was CPU-only: bytes, JSON, scoring, selection/count arithmetic and recorded
controls. No Torch, tensor deserialization, independent tensor numerical
recomputation or local inference was used. The original ZIP/log were not edited.

## Main scores

TextVQA mean consensus score includes partial credit. “Full credit” and APR are
separate: the new baseline has 44 fully correct answers and 49.7% consensus mean.
Schedules denote input states followed by states after block 16, not input-only
64/192 budgets.

| Condition | Consensus mean | Full-credit answers | Baseline 44 preserved | APR | Full-credit losses / gains |
|---|---:|---:|---:|---:|---:|
| Baseline 576 | 49.7% | 44 | 44 | reference | — |
| Native 96→32 | 41.5% | 37 | 34 | 77.3% | 10 / 3 |
| Random 96→32 s101 | 17.1% | 15 | 13 | 29.5% | 31 / 2 |
| Random 96→32 s202 | 23.0% | 20 | 15 | 34.1% | 29 / 5 |
| Random 96→32 s303 | 21.8% | 20 | 18 | 40.9% | 26 / 2 |
| Native 288→96 | 44.2% | 40 | 37 | 84.1% | 7 / 3 |
| Random 288→96 s101 | 35.6% | 32 | 28 | 63.6% | 16 / 4 |
| Random 288→96 s202 | 39.1% | 34 | 32 | 72.7% | 12 / 2 |
| Random 288→96 s303 | 29.6% | 26 | 24 | 54.5% | 20 / 2 |

| Schedule | Native | Random seed mean (range) | Native advantage, consensus percentage points | Paired 95% interval |
|---|---:|---:|---:|---:|
| 96→32 | 41.5% | 20.63% (17.1–23.0%) | +20.87 | +13.93 to +29.03 |
| 288→96 | 44.2% | 34.77% (29.6–39.1%) | +9.43 | +3.20 to +15.43 |

Intervals resample 100 question/image pairs after averaging the three random
seeds within each question. They condition on these prespecified seeds and the
screening cohort. Three seeds do not provide 300 independent questions. No
population/full-TextVQA ranking or unqualified claim about statistical significance
is made.

The stronger-compression advantage is 11.43 points larger. A separate derived
exploratory paired comparison gives a 95% interval of +3.87 to +19.57 points for
that gap difference. Both boundaries/counts change together, so this does not
identify the responsible stage or component.

Native remains below the uncompressed reference: −8.2 points at 96→32 and −5.5 at
288→96. Corresponding paired intervals versus baseline are [−15.3,−1.5] and
[−11.6,+0.7] points. The moderate interval including zero does not erase its
seven observed full-credit losses or establish equivalence to baseline.

## Beyond the six diagnostic questions

| Cohort | Baseline | Native 96→32 | Random 96→32 mean | Native 288→96 | Random 288→96 mean |
|---|---:|---:|---:|---:|---:|
| All 100 | 49.70% | 41.50% | 20.63% | 44.20% | 34.77% |
| Other 94 | 46.49% | 43.09% | 21.95% | 43.83% | 33.79% |
| Previous six | 100.0% | 16.7% | 0.0% | 50.0% | 50.0% |

On the 94-case complement, native advantages remain +21.13 and+10.04 points,
with paired intervals [+13.40,+29.22] and[+4.11,+16.10]. The wider advantage is
not supplied solely by the original selected examples. The complement remains
development/screening data, not reserved held-out evaluation.

At 288→96 the old six give a tie with the average random seed, while the other 94
show a clear descriptive native advantage. The earlier calibration also mixed
GQA/MMStar/TextVQA selected cases, so its 6-question aggregate cannot be compared
to this 100-question score as if the populations were identical. Diagnostic cases
were useful for exposing mechanisms and validating execution, not ranking methods.

## Failures, gains and nonmonotonic answers

Native exact-text failures continue. Recorded examples include:

- `35962`: Cookery school → Cooking school (96→32) or Culinary (288→96).
- `36001`: Give up → Gonna or Gone.
- `37533`: Brolio → Travel or Castello di rivoli. Random 288→96 seeds 101/202
  retain Brolio; 303 gives Bodega. Individual random rescues still occur.
- `39308`: Orcid.org → Wwwidcom at 288→96, while96→32 retains Orcid.org.
- `39323`: Sky mall → Sky at 96→32; native 288→96 preserves Sky mall while all
  three random 288→96 responses fail.

Compression also gains full credit in both schedules. Some are lexical/scoring
changes: Um → Umd, Monday night → Monday, and at 288→96 Steigl → Stiegl.
These are valid score changes under the frozen scorer; they do not all prove
new visual reasoning competence or removal of a specific distracting token.
All raw answers and references are retained for separate qualitative inspection.

Across the two native schedules, the larger schedule recovers seven full-credit
answers but loses four that the smaller schedule gets right: First, Orcid.org,
Mall and Tea become Summers first, Wwwidcom, Building and Hello. Larger retained
counts improve the aggregate without guaranteeing monotonic per-question behavior.
Selected sets, basis placement and merging also change with the schedule; this is
not an intervention that adds unchanged evidence to an otherwise identical state.

## Interpretation and next step

For this TextVQA cohort the **whole native staged procedure** preserves answers
substantially better than pure random, especially under stronger compression.
This supports proceeding with the proposed broader comparison rather than using
selected random rescues as a general method ranking. It does not isolate the
benefit of E alone: FPS bases, basis replacement, merging and two compression
boundaries differ between the methods. Earlier fixed-basis/merge ablations retain
their narrower scopes and are not overwritten by this result.

This also coexists with RQ1: weak or operator-dependent individual E–sensitivity
alignment does not by itself imply that the entire compression procedure is no
better than random. The present experiment measures generated-answer preservation
of complete procedures, not intrinsic token importance or a causal explanation
of a particular spelling change.

Proceed with [GQA development800](../RQ2_STAGED_GQA800.md) under the unchanged
prespecified schedules and seeds. It tests whether this task-dependent pattern
persists on same-image questions and more general visual QA. No TextVQA rerun,
environment update, new compression algorithm or accuracy-based selection is
needed. Keep reserved GQA evaluation outside development. MMStar confirmation
remains the next planned cross-dataset step after GQA review.

Derived audit, full condition/answer CSVs, additional comparisons, native budget
switches and the unchanged exported gallery are in
`research/results/rq2-staged-textvqa100-review-v1-2026-10-21/`.
Open `gallery/index.html` for all 100 images/questions/answers. The review script
can reproduce the verification and derived summaries without GPU inference.
