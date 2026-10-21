# Staged MMStar1500 results and three-dataset summary — 2026-10-21

The complete MMStar run passed. All 1,500 questions, six parts and 13,500 primary
predictions are present, and the matching console log ends with `EXIT 0`.
The screenshot showing 1,272/1,500 therefore displays older progress than the
downloaded artifacts. Delayed front-end output is consistent with this evidence;
the files establish completion of this run, not the state of every live process
in Kaggle. No MMStar rerun or extra inference is needed. Both downloaded files
are safely archived, and the GPU can be stopped when convenient.

ApET and pure random have similar aggregate MMStar accuracy at both schedules.
The native-minus-random paired intervals include zero. This does not establish
equivalence, random superiority or a universal absence of compression effects.
It completes the planned broad staged comparison across TextVQA100, GQA800 and
MMStar1500 under the existing fixed methods. The staged merge ablation remains
a possible follow-up, not a completed experiment or a prerequisite to archiving
these valid results.

## Archived and verified

Copied both supplied files unchanged to `research/results/`. The originals
remain in Downloads; neither supplied companion is missing:

- `rq2-staged-mmstar1500-v1-20261021T153730115642Z.zip`: 353,174,628 bytes
  (336.8 MiB), SHA-256
  `55896bafdb5d9d7ab51c3024b2465c2b0e73933adf08a86bc51aa41168ebe641`.
- Matching `.log`: 2,068,063 bytes, SHA-256
  `824b89377f4fdcad70752a3560235fda58a8d8b954353a5ee3fb648edc1187b2`.

All 5,958 unique ZIP members pass full CRC reading; 5,957 artifact hashes,
74 evidence hashes and 18 tensor-file byte hashes verify. All 59 measurement
sources match their archived snapshots and current repository. The 70-file
prerequisite signature, setup and runtime proof match. Kaggle passed all 40
prerequisites with zero skips. The eight native source pins, original uv lock
and reviewed staged calibration remain unchanged.

Execution used clean commit `b62a014393b4acc5c772f853cab00ee59f38be3f`, original
Python 3.10.18 / Torch 2.1.2+cu121 / Transformers 4.37.2, native FP16 LLaVA-1.5-7B
and T4 ×2. The pinned full1500 input manifest matches its frozen reference;
all question/label/category/image identities and original `590 + index` seeds
verify. The 1,500 questions reuse 1,430 distinct image-byte hashes. All six parts
are included without filtering baseline-wrong cases.

All 13,500 unique primary rows rescore exactly with the unchanged project
`mmstar-explicit-choice-v2` scorer and match their detailed question records.
The summary tables and paired intervals reproduce. All 1,500 per-question
encoding records verify without overwriting repeated-image evidence. The audit
checks actual stage counts, masks, top-E retention, basis counts, merge
assignments/counts, pure random memberships, source-preservation flags, compact
geometry and every cache trace. Maximum expanded primary prompt length is
1,268 tokens.

Six full576 identity controls and 18 fixed replays reproduce generated IDs,
processed scores and answers exactly. Replayed decoder source/output/RNG
fingerprints match. Four forced four-token probes each verify three cached
decoding steps. All four fresh whole-native repeats on question136 match their
primary IDs and answers (B, Soccer) at both schedules. This bounded result does
not remove the fresh-native variation observed previously in GQA or establish
dataset-wide numerical determinism. Controls stay outside primary denominators:
total execution is 13,532 generations.

Recorded runner wall time is 10,093.3 seconds (**168.2 minutes / 2 h 48 min**),
including checkpoint work, model loading, inference, controls and intermediate
record writes. It is captured before final CSV/gallery/ZIP packaging;
environment/input setup is also separate. Model loading reports 25.5 seconds. Peak allocated GPU memory
is 8.072 / 7.652 GiB. These are observed run costs, not controlled speedup estimates.

Review is CPU-only: byte inventories, JSON, scoring, selection/count arithmetic
and recorded controls. Tensors were hashed without deserialization, numerical
recomputation or local model inference. Original result files remain unchanged.

## Accuracy, preservation and uncertainty

The uncompressed baseline scores **521/1,500 = 34.73%**. APR uses those 521
baseline-correct questions. Accuracy and gains retain baseline-wrong questions.

| Condition | Correct / 1,500 | Accuracy | Baseline correct preserved | APR | Losses / gains |
|---|---:|---:|---:|---:|---:|
| Baseline576 | 521 | 34.73% | 521 | reference | — |
| Native96→32 | 501 | 33.40% | 442 | 84.84% | 79 / 59 |
| Random96→32 s101 | 503 | 33.53% | 442 | 84.84% | 79 / 61 |
| Random96→32 s202 | 511 | 34.07% | 448 | 85.99% | 73 / 63 |
| Random96→32 s303 | 495 | 33.00% | 445 | 85.41% | 76 / 50 |
| Native288→96 | 525 | 35.00% | 485 | 93.09% | 36 / 40 |
| Random288→96 s101 | 526 | 35.07% | 480 | 92.13% | 41 / 46 |
| Random288→96 s202 | 533 | 35.53% | 484 | 92.90% | 37 / 49 |
| Random288→96 s303 | 525 | 35.00% | 482 | 92.51% | 39 / 43 |

| Schedule | Native | Random seed mean (range) | Native minus random, percentage points | Paired 95% interval |
|---|---:|---:|---:|---:|
| 96→32 | 33.40% | 33.53% (33.00–34.07%) | −0.13 | −1.35 to +1.07 |
| 288→96 | 35.00% | 35.20% (35.00–35.53%) | −0.20 | −1.25 to +0.85 |

Intervals resample the 1,430 image-byte-hash groups, calculating a question-weighted
ratio of sampled score sums to sampled question counts because image groups have
unequal sizes. Random seeds are averaged within each question first. The 2,000
bootstrap draws use seed59023. Seeds do not triple the question count; intervals
condition on those three seeds and the first captured native realizations.

Both intervals include zero. The random mean's small numerical lead does not
establish superiority, nor do these intervals establish a chosen equivalence
margin. Native versus baseline is −1.33 points at 96→32 (interval −2.79 to
+0.13) and +0.27 at 288→96 (−0.93 to +1.39). The latter is four net correct
answers, while losing 36 and gaining 40; it is not evidence that compression
improves the model generally or preserves every previously correct answer.

At 288→96 native preserves slightly more baseline answers than the random mean
(485 versus 482), while random gains more baseline-wrong answers (mean 46 versus
40). Accuracy alone hides that distinction. Severe native preservation is
442/521 versus random mean 445/521. This is why both APR and paired losses/gains
remain necessary even when total accuracy is similar.

## Categories and output parsing

All categories contain 250 questions; slices below are descriptive.

| Category | Baseline | Native96→32 | Random96→32 mean | Native288→96 | Random288→96 mean |
|---|---:|---:|---:|---:|---:|
| Coarse perception | 56.00% | 54.00% | 55.07% | 58.40% | 59.47% |
| Fine-grained perception | 20.80% | 22.80% | 21.33% | 22.80% | 22.53% |
| Instance reasoning | 44.40% | 42.00% | 41.33% | 45.20% | 43.87% |
| Logical reasoning | 32.40% | 30.80% | 30.93% | 31.20% | 30.80% |
| Math | 26.00% | 25.20% | 25.73% | 24.80% | 26.40% |
| Science & technology | 28.80% | 25.60% | 26.80% | 27.60% | 28.13% |

The native procedure has no uniform category advantage. Its baseline competence
is also limited on several task types, so similar generated accuracy cannot
establish that the model retained all task-critical visual evidence. Question-only
shortcuts or particular numerical/visual causes are hypotheses, not established
by this aggregate experiment. Earlier bounded RQ1 validity findings retain their
scope; they do not classify every MMStar question.

There are **zero unparsed or empty primary responses**. Format compliance is
recorded separately: the baseline has seven noncompliant outputs, severe native
one, and moderate native eight. Primary scoring accepts explicit choices followed
by exactly matching option text, while strict scoring remains separate.
Baseline strict accuracy is 34.47%, severe native 33.40%, moderate native 34.67%;
random strict seed means are 33.53% and 34.91%. The overall interpretation persists.

Question62 reaches the 16-token cap in four arms: baseline, native288→96 and
random288→96 seeds101/303. The output is the complete explicit option text
`A: A sailboat on its side in the ocean at sunset`, which the frozen scorer
parses unambiguously. It remains flagged at the cap; no output is selectively
regenerated and no parser change is made after seeing these results.

Individual random rescues remain: native fails while at least one random seed
succeeds on 91 severe / 72 moderate questions. Native succeeds while all three
random seeds fail on 24 / 23. These are asymmetric comparisons; selecting the
best seed with gold afterward is an oracle rather than the evaluated random
procedure. The per-seed table is the relevant aggregate comparison.

## Historical agreement

All **1,500 uncompressed raw answers and scores** reproduce the October13 native
baseline exactly, with matching images, prompts, labels and seeds. Do not confuse
this 34.73% native baseline with the earlier HF LLaVA result from a different path.

Compared with the old combined288→96 arm, two raw native answers change:

- Question544, cage material (reference A, Plastic): C → B, both incorrect.
- Question648, instrument (reference B, Drums): C → B, wrong → correct.

The new moderate native score is 525/1500 versus 524/1500 previously. These are
historical fresh-realization comparisons, not frozen-feature replay failures or
evidence of a particular numerical cause. The current primary remains the first
recorded realization; nothing was replaced to improve the score.

## Three-dataset synthesis and next step

The same calibrated operator sources, conditions, dependency lock, package
versions and pinned model/vision checkpoints verify across all three runs.
The planned broad comparisons now cover 2,400 question evaluations and 21,600
primary predictions. Cohorts and metrics remain separate; there is no pooled
accuracy or full-TextVQA/full-GQA claim.

| Dataset / metric | 96→32 native advantage over random mean | 288→96 native advantage over random mean |
|---|---:|---:|
| TextVQA100 consensus | +20.87 points [13.93, 29.03] | +9.43 [3.20, 15.43] |
| GQA800 accuracy | +2.46 [0.25, 4.71] | +1.17 [−0.58, 2.83] |
| MMStar1500 accuracy | −0.13 [−1.35, 1.07] | −0.20 [−1.25, 0.85] |

The current defensible result is heterogeneous performance: a substantial native
advantage on the fixed TextVQA cohort, a smaller GQA effect, and no clear aggregate
native advantage on MMStar. The text cases remain useful for explaining failures
despite better overall preservation than random. MMStar shows that a native
advantage does not automatically generalize to every task distribution. Different
cohorts, metrics and baseline abilities preclude a simple pooled effect estimate
or a causal attribution of the difference between datasets to OCR alone.

This compares complete procedures: selection, FPS/basis replacement and merging
all differ. It cannot attribute native gains solely to E or overturn RQ1's
intervention-specific sensitivity observations. A focused staged merge on/off
study can address the remaining component question, after saved-data synthesis
and with its selection policy explicitly specified. Avoid another broad sweep
or a new compressor based solely on these means. Reserved GQA evaluation stays
untouched.

The original archive is intact. Derived verification, condition/category/format
CSVs, historical changes, fresh controls, cross-dataset CSV/JSON and the unchanged
exported gallery are saved in
`research/results/rq2-staged-mmstar1500-review-v1-2026-10-21/`.
The audit and cross-dataset scripts reproduce this analysis without Torch.
