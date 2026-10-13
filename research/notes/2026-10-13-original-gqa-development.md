# Original ApET: frozen GQA development baseline — 2026-10-13

Evidence: `research/results/original-gqa-development-20261013T221805402373Z.zip`.
Audit: `research/results/original-gqa-development-audit-20261013.json`.
Complete bundle with the omitted matching lock restored:
`research/results/original-gqa-development-20261013T221805402373Z-complete.zip`.
Original ZIP and all original member bytes are preserved.

## Completeness and provenance

The run passed: 800 development questions on 100 images, all four configurations,
3,200 unique predictions in the expected order. No failed or empty generation
records, duplicate pairs, or missing requested compression stages were found.
Maximum output length is six tokens, below the unchanged 16-token cap.

Kaggle commit `ecf4662ba3d4e353a99cca947341e14eb93d9c53`, clean checkout;
one native model load across both T4s. The frozen study ID is
`3153bd275dbefd346e132c18ac16d7cbe46e5534c4f9e84fed2786abe038f512`.
The archived manifest matches the locally prepared development manifest exactly.
All 100 images validate, including raw bytes, dimensions and decoded pixel hashes.
The copied study assignment confirms no reserved-evaluation images are used.

All recorded scores, summaries, APR/type/image comparisons, seeds, prompts,
settings and token-count traces were independently checked. Source snapshots
and recorded hashes match the unchanged original native reference and project
code. The prediction JSONL hash agrees with the report.

One archive packaging omission was found: the notebook skipped every `.lock`
file, which also excluded `environments/original/uv.lock`. The subset and full GQA
notebook ZIP filters now exclude only the specific runtime process lock. The local
dependency lock matches both recorded hashes; it was added to a separate complete
ZIP. No prediction/report bytes were changed and no GPU rerun is needed.
Use the complete ZIP when restoring this baseline for future comparisons.

## Main results

APR uses the 552 baseline-correct questions. Lost/gained counts compare exactly
paired correctness against baseline.

| Configuration | Correct / 800 | Accuracy | APR | Lost / gained | Mean generation seconds |
|---|---:|---:|---:|---:|---:|
| Baseline | 552 | 69.000% | Reference | — | 0.730 |
| Input-only | 549 | 68.625% | 94.38% | 31 / 28 | 0.497 |
| Decoder-only | 549 | 68.625% | 97.46% | 14 / 11 | 0.562 |
| Combined | 544 | 68.000% | 93.84% | 34 / 26 | 0.438 |

The initial reference settings are unchanged: baseline input 576; input-only 288;
decoder-only input 576 with layer 16 → 96; combined input 288 with layer 16 → 96.
There is no new compression algorithm in this run.

Nearly unchanged overall accuracy hides which answers change. Combined loses
34/552 previously correct answers (6.16%) and gains 26 previously wrong answers,
leaving a net loss of eight questions / 1.00 percentage point. Input-only and
decoder-only each lose a net three questions, but their losses and preserved
answers differ. Selected-token budgets/layer placement differ across these
configurations, so this is not an isolated controlled comparison of layers alone.

Parsed prediction differences:

- Baseline → input-only: 78 changed answers, 59 correctness changes.
- Baseline → decoder-only: 47 changed answers, 25 correctness changes.
- Baseline → combined: 88 changed answers, 60 correctness changes.
- Input-only → combined: 51 changed answers, 29 correctness changes.

Scoring remains project `gqa-short-answer-v1`, with case/whitespace/trailing-period
normalization and exact label matching. Literal case-sensitive correctness is
zero because answers are capitalized. This is expected, not zero task accuracy.
Answer vocabulary/specificity changes (for example, `Woman` → `Girl`) need separate
manual review from clear relation/detection changes. Do not automatically turn
synonyms or related labels into correct predictions after seeing the outcomes.

## Task patterns and interpretation

| Structural question type | Questions | Baseline correct | Combined correct | Combined APR | Lost / gained |
|---|---:|---:|---:|---:|---:|
| choose | 105 | 88 | 84 | 94.32% | 5 / 1 |
| compare | 78 | 52 | 53 | 92.31% | 4 / 5 |
| logical | 147 | 116 | 118 | 98.28% | 2 / 4 |
| query | 318 | 167 | 161 | 88.62% | 19 / 13 |
| verify | 152 | 129 | 128 | 96.90% | 4 / 3 |

Open query questions have the lowest preservation among these structural groups,
and account for 19 of 34 combined losses. Baseline query accuracy is only 52.52%,
so this set is not saturated for the current model/scoring. Logical and verify
questions preserve more baseline-correct answers in this run. Semantic object
questions preserve all 85 baseline-correct answers, whereas global questions
preserve 21/25 (84%); the latter has a small denominator.

These are patterns in a deliberately image-balanced/type-diverse development
subset. They are not full-GQA population results or proof that one task type
causally needs a particular layer/token budget. For descriptive uncertainty,
10,000 bootstrap resamples of whole eight-question image groups (seed 590) give
accuracy-difference intervals versus baseline:

- Input-only: −0.375 pp, interval [−2.125, +1.375] pp.
- Decoder-only: −0.375 pp, interval [−1.750, +1.125] pp.
- Combined: −1.000 pp, interval [−2.875, +0.750] pp.

All intervals include zero. They describe this chosen sample's image-cluster
variability, not random-sample inference about all GQA. The evidence is stronger
for finding diagnostic cases than claiming a small overall accuracy difference.

## Runtime and memory

One scored session: **1,842.22 seconds / 30.70 minutes**, consistent with the
planning estimate. Model loading took 22.358 seconds and warmups 4.894 seconds,
separate from scored wall time. The recorded native mean generation time drops
from 0.730 to 0.438 seconds under combined compression (about 40% lower, observed
1.67× ratio). This includes vision/compression/prefill/decode and count hooks,
with variable short answers; it is not a controlled repeated latency benchmark.

Peak allocated memory stays within both T4s. GPU 0 maxima are 8.134 GiB for
baseline/input-only/combined and 9.047 GiB for decoder-only. GPU 1 maxima range
from 7.212 to 7.274 GiB. Decoder pruning does not guarantee a lower peak allocation.

## Repeatability and next experiments

On the 40 earlier pilot questions, all baseline and decoder-only raw answers
match. Input-only and combined each differ on one question, `2058428`, image
`n497658`: `Are there airplanes in this scene that are huge?` Gold is `yes`.
The earlier observed pilot answered `No` under both compressed settings; this
development run answers `Yes`. Prompt, seed, image hash, stage settings and
token-count trace match. The cause is not established. Possible numerical,
logging or execution-state differences require controls; a fixed seed alone
has not established exact repeated-answer identity for every compressed case.

Next perform a tiny repeatability check on this question: repeated identical
generation, configuration switching, and logging off/on on the same hardware,
recording actual FPS/selection/merge values. This can be done without rerunning
all 800 questions. Repeat promising failure cases before attributing a single
answer change to particular visual tokens.

The current audit identifies 34 combined losses across 27 images. Every loss has
at least one baseline-correct, compression-preserved question on the same image,
which provides useful RQ1 controls. Input-only has 31 losses across 27 images;
decoder-only has 14 across 14 images. The initial metadata-selected 80-question
diagnostic set covers 8 combined, 5 input-only and 2 decoder-only losses.

Examples include `201428959` (image `n64959`, refrigerator right → left) and
`20118949` (image `n540852`, necktie/shoe color comparison no → yes). The audit JSON
retains every lost case, raw answer, source question type and same-image preserved
control ID. Select a small, frozen development-only case set spanning relation,
attribute and scene queries, alongside gained/preserved controls. Then collect
native reconstruction/selection/merge diagnostics and implement limited
answer-contribution interventions. Reserve evaluation for confirmation after
tuning; no further large baseline run is needed now.
