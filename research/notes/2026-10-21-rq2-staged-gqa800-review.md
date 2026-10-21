# Staged GQA development800 results — 2026-10-21

The complete run passed: all 800 questions on 100 development images, nine
conditions, 7,200 primary predictions and 32 separate controls. Native ApET
outperforms each random seed descriptively at both schedules. Its advantage is
much smaller here than on TextVQA100; the moderate-schedule paired interval
includes zero. Proceed with the independent MMStar notebook under the frozen
settings. No GQA execution fix or inference rerun is required.

## Archived evidence and verification

Both supplied Downloads files were copied unchanged into `research/results/`;
the originals remain in Downloads:

- `rq2-staged-gqa800-v1-20261021T142303483991Z.zip`: 157,939,720 bytes
  (150.6 MiB), SHA-256
  `58ef6ea2228fdcb38d3cb6bdee2f67dccf5e9ee9d8cda3fab9c67a1b69fe3c73`.
- Matching `.log`: 519,003 bytes, SHA-256
  `c857529b1f1c573f1a36d1ef08bc48e6f1a737e4ea99b01eb4b03d42519f9e44`.

All 1,201 unique ZIP members pass full CRC reading; 1,200 artifact hashes,
78 evidence hashes and 18 tensor-file byte hashes verify. The 62 measurement
sources match the archived snapshots and current repository. All eight native
source pins, the reviewed calibration contract and original uv lock are unchanged.
The run used clean commit `78e8f0d979c35f7f42155c084f8bf56050d65ae5`, original
Python 3.10.18 / Torch 2.1.2+cu121 / Transformers 4.37.2, FP16 native LLaVA-1.5-7B
on T4 ×2. Kaggle passed all 39 prerequisites with zero skips; the 73-file source
signature, setup and runtime evidence match.

The unchanged development/study manifests, all 800 question/image identities,
prompts, labels, image seeds and task types verify. Reserved evaluation receives
no inference. All 7,200 unique primary rows rescore exactly and match their
question records. All summary tables and paired image-bootstrap intervals
reproduce from the raw predictions. There are no empty primary answers or
outputs reaching the 16-token cap.

All 100 image encoding banks verify; each is shared by exactly its eight
questions. The audit checks actual 96→32/288→96 counts after block 16, masks,
top-E retention, basis counts, merge assignments/counts, random memberships,
source-preservation flags, compact geometry and all cached decoding traces.
Six full576 no-ops and 18 fixed replays reproduce answers, generated IDs and
processed scores exactly; decoder source/output/RNG fingerprints agree on
replay. Four forced four-token probes each verify three cached decoding steps.
The four fresh whole-native controls are discussed separately below.

Runner wall time was 3,795.4 seconds (**63.3 minutes**), including checkpoint
work, model loading, inference, controls and export; setup is separate. Model
loading reports 24.2 seconds. Peak allocated GPU memory is 7.826 / 7.323 GiB.
These are observed run costs, not controlled speedup measurements.

Review is CPU-only: bytes, JSON, scoring and recorded structural/control evidence.
Tensor bytes were stream-hashed without deserialization or independent numerical
recomputation. No Torch installation or local inference was used. No supplied
companion is missing.

## Accuracy and preservation

The new uncompressed baseline is **552/800 = 69.0%**. APR uses these 552 correct
answers as its denominator; baseline-wrong questions remain in accuracy/gains.

| Condition | Correct / 800 | Accuracy | Baseline correct preserved | APR | Losses / gains |
|---|---:|---:|---:|---:|---:|
| Baseline576 | 552 | 69.00% | 552 | reference | — |
| Native96→32 | 510 | 63.75% | 474 | 85.87% | 78 / 36 |
| Random96→32 s101 | 483 | 60.38% | 448 | 81.16% | 104 / 35 |
| Random96→32 s202 | 494 | 61.75% | 460 | 83.33% | 92 / 34 |
| Random96→32 s303 | 494 | 61.75% | 456 | 82.61% | 96 / 38 |
| Native288→96 | 544 | 68.00% | 518 | 93.84% | 34 / 26 |
| Random288→96 s101 | 529 | 66.13% | 501 | 90.76% | 51 / 28 |
| Random288→96 s202 | 537 | 67.13% | 511 | 92.57% | 41 / 26 |
| Random288→96 s303 | 538 | 67.25% | 510 | 92.39% | 42 / 28 |

| Schedule | Native | Random seed mean (range) | Native advantage, percentage points | Paired 95% interval |
|---|---:|---:|---:|---:|
| 96→32 | 63.75% | 61.29% (60.38–61.75%) | +2.46 | +0.25 to +4.71 |
| 288→96 | 68.00% | 66.83% (66.13–67.25%) | +1.17 | −0.58 to +2.83 |

Intervals resample 100 whole eight-question image groups, averaging random seeds
within each question first (2,000 draws, seed 59022). They condition on the fixed
development cohort, seeds and first native realizations. They do not incorporate
whole-run numerical variation or turn 800 questions/three seeds into 2,400
independent observations. The moderate interval includes zero; an overall
native advantage at that schedule remains uncertain.

Native versus baseline changes are −5.25 points at 96→32 (interval −8.13 to
−2.63) and −1.00 at 288→96 (−2.88 to +0.75). Similar overall accuracy does not
mean preservation is complete: the moderate schedule still loses 34 baseline
correct answers and gains 26 others.

## Task patterns and same-image controls

The following are descriptive slices, without separate subgroup inference:

| Structural task | Questions | Baseline | Native96→32 | Random96→32 mean | Native288→96 | Random288→96 mean |
|---|---:|---:|---:|---:|---:|---:|
| Choose | 105 | 83.81% | 74.29% | 71.43% | 80.95% | 79.05% |
| Compare | 78 | 66.67% | 62.82% | 67.95% | 67.95% | 68.80% |
| Logical | 147 | 78.91% | 77.55% | 73.92% | 80.27% | 78.00% |
| Query | 318 | 52.52% | 44.97% | 41.61% | 50.63% | 49.37% |
| Verify | 152 | 84.87% | 82.89% | 79.82% | 83.55% | 83.11% |

The native advantage is not uniform across task slices: random's mean is higher
for structural comparison questions at both schedules. This is an observation
for later synthesis, not proof of a task-specific mechanism or a reason to
filter the benchmark. GQA's exact-match/annotation limitations remain relevant;
the scorer stays frozen and raw strings are retained for qualitative review.

All 78 severe-schedule losses and all 34 moderate losses have at least one
baseline-and-native-correct question about the same image. The saved loss/control
CSV retains those IDs for targeted inspection. They span multiple images, not a
single shared scene. As before, preserving one question does not prove the
representation is sufficient for every question about that image.

Individual random rescues coexist with the aggregate native advantage. Native
is correct while all three random seeds fail on 20 severe / 13 moderate cases;
native is wrong while at least one random seed succeeds on 82 / 47. These are
asymmetric comparisons: choosing the best of three random answers after seeing
gold is an oracle, not a deployable random method or a valid aggregate ranking.

## Repeatability and historical agreement

All **800 uncompressed raw answers and scores** reproduce the earlier original
GQA development baseline exactly, with matching images, prompts, labels and seeds.
The native288→96 arm has the same nominal schedule as the older combined arm.
Its aggregate accuracy remains 544/800, but four raw answers change:

| Question ID | Previous combined | Current native288→96 | Correctness |
|---|---|---|---|
| 201896041 | Left | Right | wrong → correct |
| 2058428 | Yes | No | correct → wrong |
| 201641244 | White | Tan | both wrong |
| 201935028 | Sidewalk | Street | both wrong |

This is historical answer agreement, not an exact frozen-feature replay check.
The flagged 2058428 remains included and flagged; no numerical debugging or
selective rerun was performed.

The predeclared fresh-native case 201636988 asks “Is the color of the cake
different than the plate?” (reference **no**). Severe native primary and both
fresh repeats say **Yes** (wrong). Moderate native primary says **No** (correct);
one fresh repeat says No and the other says **Yes** (wrong). Thus three of four
fresh repeats match their primary generated IDs and one changes correctness.
Exact fixed-feature replay on this same case remains stable.

The protocol deliberately records fresh variation without replacing the first
primary answer, so this does not invalidate execution or trigger a full rerun.
It does limit fine claims about small accuracy differences. Four targeted
repeats do not estimate a dataset-wide instability rate. The underlying cause
is not established by these records; do not label it proof of CUDA scatter or
one particular compression stage. Historical drift is likewise not a direct
measurement of within-session stochasticity alone.

## What this adds and what follows

The broad GQA evidence favors the native procedure descriptively over every
random seed, with a modest severe-schedule advantage and an unresolved moderate
gap. On the fixed TextVQA cohort, the corresponding native advantages were
20.87 / 9.43 consensus points; here they are 2.46 / 1.17 accuracy points. These
are different dataset metrics and cohorts, but they reinforce the need to study
task dependence rather than impose a universal random-versus-ApET conclusion.

The complete procedure comparison still combines E selection, FPS/basis
replacement and merging. It does not identify E alone or contradict RQ1's
operator-dependent sensitivity findings. It also does not yet explain exact
text failures. Finish the [prepared MMStar comparison](../RQ2_STAGED_MMSTAR1500.md)
with unchanged schedules/seeds, then consolidate all three datasets before
choosing the focused staged merge ablation. Reserved GQA evaluation stays unused.

Derived verification, reproduced summary, condition CSV, historical changes,
fresh repeats, losses/same-image controls and an unchanged exported gallery are
saved separately in `research/results/rq2-staged-gqa800-review-v1-2026-10-21/`.
Open `gallery/index.html` for the original run's complete question/answer review.
