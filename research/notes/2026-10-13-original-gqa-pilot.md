# Original ApET GQA diagnostic pilot — 2026-10-13

Evidence: `research/results/original-gqa-pilot-20261013T194041878789Z.zip`.
Independent CPU audit: `research/results/original-gqa-pilot-audit-20261013.json`.
Kaggle commit `d6a87285418626101cce3e09753aaafdd0087e1e`, clean checkout;
original environment lock unchanged. No native code, dependency, scoring or
pilot inference changes were made during this audit. The ZIP is preserved.

## Execution and evidence checks

All 160 predictions completed: 10 images, four questions per image, all four
native configurations. No duplicate question/configuration pairs, empty answers,
failed generations, missing stages or invalid token indices were found.

Logging on/off controls passed on the first question under each configuration,
checking responses, generated-token counts and token-count traces. This is a
limited observation check, not parity testing for all questions or all inputs.
The pilot notebook also runs native synthetic-tensor observer tests before
inference; their console output is not bundled as a separate test log in this ZIP.

Archive CRC, source snapshot hashes, original native reference hashes and lock
hash passed validation. All 10 raw images match byte hashes, dimensions and
locally decoded RGB hashes. All 40 original question rows, image records and
image-based FPS seeds match the already-prepared full Test-Dev manifest.

All short-answer scores, type/category summaries, APR comparisons, grouped
same-image comparisons and diagnostic fingerprints were independently
recomputed and agree with the saved report.

## Scores and answer review

| Configuration | Correct / 40 | Accuracy | APR on 29 baseline-correct | Lost / gained |
|---|---:|---:|---:|---:|
| Baseline | 29 | 72.5% | Reference | — |
| Input-only | 26 | 65.0% | 86.21% | 4 / 1 |
| Decoder-only | 28 | 70.0% | 96.55% | 1 / 0 |
| Combined | 23 | 57.5% | 79.31% | 6 / 0 |

This is a selected same-image diagnostic subset, not an estimate of full GQA
accuracy. All answers are short, with 2–4 generated tokens including termination;
none reaches the 16-token cap. The full generation limit can remain unchanged.

Literal case-sensitive correctness is zero in all configurations because the
model capitalizes its short answers (for example, `Left` versus gold `left`).
The declared `gqa-short-answer-v1` normalization correctly handles this. Zero
literal matching is not zero answer accuracy or a scoring bug.

Reviewing all 40 question groups found no need for an inference/scoring fix.
There are exact-label mismatches such as `Field` versus `pasture`, `Sunny` versus
`cloudless`, and `Brown` versus `dark brown`. These include answer vocabulary or
specificity issues, not necessarily pure visual failures. Retain strict declared
scoring and raw answers; do not introduce gold-dependent synonyms or manually
change scores after seeing the results. Future independent rescoring remains
possible without rerunning inference.

Examples to revisit for RQ1:

- `201882614`, image `n125122`: lamp left/right. Baseline and decoder-only answer
  `Left` correctly; input-only and combined answer `Right`. Three other questions
  about the image retain their earlier correctness outcomes.
- `20162446`, image `n202379`: fence in front of what? Baseline/decoder-only answer
  `Trees`; input-only/combined answer `Field`.
- `202257086`, image `n92308`: place identification. Baseline/input-only answer
  `Beach`; decoder-only/combined answer `Park`.
- `2075475`, image `n469525`: animal standing on grass. Only combined changes the
  correct `Deer` to `Giraffe`.

These are candidate compression-sensitive questions. Observations alone do not
measure token answer contribution or establish which discarded/merged feature
caused an error; controlled interventions are still needed.

## Same-image observations

There are 30 first-question-versus-other-question comparisons per configuration.
For input-only and combined input compression, all 30 have identical errors,
FPS basis indices, selected sets, basis assignment slots and merge records.

At decoder layer 16, selected sets and FPS basis indices also match on all 30
pairs in both decoder configurations. Other quantities vary:

| Decoder comparison | Decoder-only | Combined |
|---|---:|---:|
| Error vectors outside declared closeness tolerance | 14 / 30 | 30 / 30 |
| Largest absolute error-vector difference | 0.0235901 | 0.0464973 |
| Basis slot-to-source mapping differs | 3 / 30 | 17 / 30 |
| Dropped-to-target assignment list differs | 6 / 30 | 3 / 30 |
| Entire merge record differs | 14 / 30 | 29 / 30 |

The report field `merge_assignments_equal` compares the entire merge dictionary,
including target order, counts and basis slots. A false value does not by itself
prove a different dropped-to-target map. The raw records allow the more specific
checks above. Keep this distinction in later analyses.

Visual tokens precede question tokens in the causal prompt, so visual states
cannot attend to a later question at these decoder layers. Different question
lengths can change numerical/kernel behavior; the observed variations should
not automatically be called question conditioning. Numerical explanations need
controlled testing. Identical selected sets also do not guarantee identical
merged features or answers.

## Full-run readiness and timing

No blocking inference, environment or scoring problem was found. The separate
full runner uses the same native loading/generation helper, checkpoints, prompt,
normalization, seed rule and budgets. Its seven CPU tests passed again, including
guarded resume after a partial question. No compression code fixes are needed.

Pilot scored wall time: **91.323 seconds**. Model loading: 5.519 seconds; warmup:
3.041 seconds; logging parity controls: 4.242 seconds. Summed scored generation
times across the four configurations: 84.152 seconds.

Scaling 40 questions to 12,578 yields approximately **7.35 hours** from generation
time or **7.98 hours** from scored wall time. These are rough forecasts, not a
guarantee: the selected question mix and detailed logging differ from the full
run, which records token counts only and refreshes summaries periodically.
The full runner prints an evolving ETA and supports interrupted-run recovery.

Next: commit/push the prepared full-run files, update Kaggle after this pilot has
finished, and use `research/notebooks/kaggle_original_gqa_full.ipynb` with
`RUN_FULL_TEST = True`. The same live session can reuse environment and weights.
There are 50,312 scored predictions; allow roughly eight hours. Download the
result ZIP before stopping the session, including partial results if interrupted.
Keep this pilot ZIP alongside full GQA and MMStar evidence for later RQ1 analysis.
