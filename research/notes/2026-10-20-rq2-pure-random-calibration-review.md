# Pure random / native ApET calibration review — 2026-10-20

The calibration passed, and **native ApET really compresses the decoder input to
64 visual states**. The six-case results do **not** show an ApET advantage over
pure random under stronger compression. Native64 gets 2/6 correct; the three pure
random seeds get 4/6, 3/6, 2/6. Both native64 variants fail all three selected text
questions. The sample is diagnostic and partly failure-enriched, so these are
case observations, not a benchmark ranking.

The UTC run timestamp is 2026-10-21T00:19:51Z, which is the evening of October 20
in the user's America/New_York timezone. This review uses the local date.

## Saved evidence and verification

Copied both supplied Downloads files into `research/results/`, leaving originals
intact and verifying matching SHA256 digests:

- `rq2-pure-random-calibration-v1-20261021T001951260192Z.zip`:
  183,759,898 bytes, SHA256
  `acfcb256e078f12c2349e8d6e83973286860de0b04425ea3e3e030e0233b2f9b`.
- Matching `.log`: 9,596 bytes, SHA256
  `905cc8611154b716c096d3140f2e7aba02ba5640dbef4469cc86d5322f0be660`.

All 141 unique members pass streamed CRC/SHA checks: 140 artifact hashes, 111 evidence
hashes, 9 feature hashes and 6 question hashes. All 40 measurement source copies
match their digests and current files. All eight pinned native model sources and
the original environment lock remain unchanged. Runtime commit
`b4d4c2c175ef81c933ba02171962c58e36c9ea4e` has a clean checkout. The original Python
3.10.18 / Torch 2.1.2+cu121 / Transformers 4.37.2 runtime and two CUDA GPUs are recorded.

The setup log shows 26 prerequisites passing without skips, including the new
actual tiny native64 and pure-random tests. All 66 primary predictions independently
rescore with unchanged dataset scorers; JSONL rows and reproduced summaries match
the reports. Each question has exact pure576 feature/mask/generated-ID/reference-
score parity. All 22 fixed replays match; four fresh native captures on Brolio keep
generated IDs unchanged while recording small feature/score differences. The
largest fresh reference-token log-probability change is 0.00801021. All 18 shared
baseline/native288/no-merge generated-ID comparisons match the earlier crossed
full30 archive. There is no observed reference drift requiring a rerun.

The reproducible CPU review, verification JSON, answers CSV, budget audit,
fresh-repeat/history tables and unchanged model canvases are in
`results/rq2-pure-random-calibration-v1-review-2026-10-20/`. Its
[gallery](../results/rq2-pure-random-calibration-v1-review-2026-10-20/index.html)
shows every case, answer and retained budget. No Torch, model or tensor pickle was
loaded on the Mac. Tensor arithmetic is checked through archived runtime assertions,
source inspection and artifact bytes, not independently recomputed locally.

## Does ApET really compress to 64 input states?

The records agree at multiple boundaries for every native64 run:

1. Native reconstruction selection has 64 unique indices and a returned mask with
   exactly 64 true positions. The CPU review independently regenerates its boolean
   mask fingerprint from the selected indices.
2. Ten native FPS representations occupy ten replacement slots; the other 54
   ordinary retained targets receive all 512 unselected source states in the
   merge arm. The merge graph covers every source exactly once, and recipient
   counts sum to 512. No-merge keeps the identical selected positions and basis
   mapping but bypasses averaging.
3. Generation observes `input: {before:576, after:64}` and no decoder-pruning stages.
4. Native multimodal preparation physically indexes `image_feature[index_mask]`
   before inserting image states into the decoder. Fixed-reference scoring checks
   the actual expanded embedding length and native compact positions.

For GQA201636988, the generation prompt expands to 641 tokens at full576 and 129
at64; the scored prompt+answer expands to 642 versus 130. Both shrink by 512,
consistent with 576→64. These lengths include text, so a 129-token prompt still
contains only 64 visual states.

The intermediate stored projected tensor has 576 positions, accompanied by the
selection mask. This storage shape does not mean that 576 image states reach the
decoder. Also, K=64 is input-only retention here; it is not an average-per-layer
or combined input/decoder paper-reproduction budget. Retention is 11.1%, or 88.9%
fewer visual decoder inputs. CLIP still processes the original image.

## Results on the six selected questions

Full576 is 6/6. TextVQA correctness in this table means full consensus credit.

| Method | Correct at 288 /6 | Correct at 64 /6 |
|---|---:|---:|
| Native ApET with merging | 3 | 2 |
| Same native selection/basis, no merging | 5 | 2 |
| Pure random 101, no merging | 5 | 4 |
| Pure random 202, no merging | 5 | 3 |
| Pure random 303, no merging | 4 | 2 |

Stronger compression hurts several cases, but it does not make E-based ApET beat
pure random in this panel. At 64, random 101 rescues cake/plate and Dominion versus
native, random 202 rescues cake/plate, and random 303 rescues cake/plate while losing
the native-correct stove question. Thus the third seed ties native in total while
preserving different answers.

This compares complete native ApET with pure random dropping. Selection, basis
handling and merging differ, so it does not isolate E's individual contribution.
The older basis-fixed crossed result remains the narrower ordinary-selection
ablation at288. A general E-versus-random conclusion requires broader predefined
data; six questions on five images, with three seeds, are not 18 independent cases.

## Text failures: stronger compression changes the kind of wrong answer

| Case | Full576 | Native288 merge | Native64 merge | Native64 no merge | Pure random64 seeds 101 / 202 / 303 |
|---|---|---|---|---|---|
| Wine name 37533 | Brolio | Bottoli | Chianti | Ccf | Chateau / D'travel / D'angelo |
| Theatre name 35670 | Dominion | Dominion | Will | Will you rock you | Dominion / Will / Rock |
| School name 35962 | Cookery school | Culinary school | Cooking school | Cooking school | Culinary / Culinary / Culinary |

All three native64 answers are wrong with and without merging. Removing merging
therefore does not repair these64-token text cases. In particular, Brolio's288
no-merge rescue no longer survives the tighter budget; it would be premature to
claim that no-merge is a general OCR fix.

The actual 336×336 Dominion and Brolio model canvases were inspected. The theatre
name DOMINION is above a large WE WILL ROCK YOU poster; Will/Rock answers overlap
the poster text rather than the requested venue name. The wine image also has a
bottom watermark containing CCF/DTRAVEL-like text, compatible with the Ccf and
D'travel outputs. These support a qualitative **wrong text source / requested-name
preservation** hypothesis, beyond visually similar misspellings alone. We have not
isolated the causal visual states or proved which text region generated an answer.
Do not automatically describe these outcomes as hallucinated or blurred letters.

The school case remains an exact-name failure: Cooking/Culinary school is related
language but does not preserve the printed Cookery school name. The existing
three-case screen remains too small to establish a general semantic-versus-
literal preservation law.

## Correction: MMStar1014 is not the soccer control

The run guide and implementation note incorrectly described 1014 as soccer. The
actual frozen question is a **bar-chart count**, “How many bars have values smaller
than 6?” It already carries `mmstar_grounding: true` in the accepted parent profile.
Earlier option/image checks found that this case always answered displayed A.
The actual soccer control is 136.

This was a case-description/selection-intent mistake in our documentation, not a
new inference error or a changed execution identity. The submitted protocol and
archive consistently use 1014, and the existing flag is preserved. All methods
return A here, which must not be promoted as evidence that64 tokens preserve
grounded MMStar reasoning. Correct the documentation without replacing the sample
or rewriting old reports. No rerun is needed to establish64-token execution.

A labeled sensitivity view excluding this already-flagged question gives:

| Method | Correct at288 /5 | Correct at64 /5 |
|---|---:|---:|
| Native merge | 2 | 1 |
| Native no merge | 4 | 1 |
| Pure random 101 | 4 | 3 |
| Pure random 202 | 4 | 2 |
| Pure random 303 | 3 | 1 |

The comparative pattern is unchanged, but this is a secondary view, not a new
frozen cohort. The two GQA questions also share one image.

## Next step

Calibration is sufficient to validate the implementation and document these
failures. There is no required fix or repeat of this six-question run. It does not
support the proposed conclusion that ApET's selection becomes better than random
at64. The next substantive comparison remains **all TextVQA100 at both budgets**,
including all baseline failures, gains and unchanged cases, with all three fixed
seeds and the same locked runtime. That broad-pool runner is not yet implemented.

Keep the current calibration as the execution reference, preserve its flags and
archives, and use the larger screening pool to test the budget-dependent
hypothesis without choosing only favorable examples. Further improvements to
ApET or another token-intervention sweep are unnecessary for this next question.
