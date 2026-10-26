# Midterm v2 supporting evidence

Collected October 25, 2026. These files support revision of the report; the report itself and frozen experimental protocols were not changed. All computations use saved records, with no model/GPU runs.

## 1. How the 30-case panel was chosen

The RQ1 panel and the later RQ2 component panel use the same 30 questions across 24 images. They were selected from **our own full576 versus native input-only288 results**, without decoder compression. The selection was enriched for compression failures and preserved controls: **11 losses and 19 preserved answers**, all baseline-correct. These are historical selection labels, not the outcomes of later component/staged captures. E/I values and later random/merge results were not used to choose these cases.

| Dataset | Candidate history | Reviewed primary pool | Final panel |
|---|---|---:|---|
| GQA | 60 first-round cases; 39 new-image candidates plus 10 same-image replacement reviews in round 2; 109 unique cases across rounds | 49 | 12: 4 losses + 8 preserved; 6 images |
| MMStar | 30 first-round cases; 12 candidates plus 6 replacement reviews in round 2; 48 unique cases across rounds | 18 | 12: 4 losses + 8 preserved; 12 images |
| TextVQA | Seeded 100-question/100-image screening cohort; the earlier 30-image gallery is nested within it | 100 screened | 6: all 3 full-credit input288 losses + 3 reviewed literal-text preserved controls |

GQA first-round review had 30 keeps, 13 exclusions, 11 ambiguous, 3 uncertain and 3 possible annotation issues. A stricter mechanism rubric adopted 22 of the keeps; 27 round-2 additions produced the 49-question pool. MMStar first-round review had 17 keeps, 12 exclusions and 1 ambiguous: seven baseline-correct keeps plus 11 additions formed the primary 18; ten other keeps remained qualitative. Round 2 was assistant-reviewed and unblinded, not independently human-validated. GQA final IDs were fixed as four loss/control pairs and two preserved pairs; MMStar used a stable SHA ordering of its 18 accepted cases. This is a diagnostic panel, not a population accuracy sample.

See [selection funnel](panel_selection_funnel.json) and [all 30 cases with historical responses](panel_cases.csv).

## 2. Preservation and transitions for all six Table 2 rows

APR = preserved / baseline-correct. Loss means baseline-correct → native-wrong; gain means baseline-wrong → native-correct. For TextVQA these counts use **full consensus credit**, while the report's main score remains partial-credit consensus. Counts therefore should not be substituted for its 41.50/44.20 scores.

| Cohort | Schedule | Baseline correct | Preserved | Losses | Gains | Native correct | APR |
|---|---|---:|---:|---:|---:|---:|---:|
| textvqa 100 | 96→32 | 44 | 34 | 10 | 3 | 37 | 77.27% |
| textvqa 100 | 288→96 | 44 | 37 | 7 | 3 | 40 | 84.09% |
| gqa 800 | 96→32 | 552 | 474 | 78 | 36 | 510 | 85.87% |
| gqa 800 | 288→96 | 552 | 518 | 34 | 26 | 544 | 93.84% |
| mmstar 1500 | 96→32 | 521 | 442 | 79 | 59 | 501 | 84.84% |
| mmstar 1500 | 288→96 | 521 | 485 | 36 | 40 | 525 | 93.09% |

The exact identity is native correct = baseline correct − losses + gains. Aggregate changes conceal paired transitions in every cohort, not only MMStar. [Six-row export](table2_preservation.csv) also contains random-seed means and sample SDs; [all 27 conditions](preservation_by_condition.csv) preserves each random seed separately. Fractional random mean counts are averages over three runs, not additional independent questions.

## 3. Thirty per-question correlations

[Wide CSV: 30 questions](rq1_correlations_30.csv); [long CSV: 90 comparison values](rq1_correlations_long.csv); [strip plot PNG](figures/rq1_correlation_strip.png) and [vector PDF](figures/rq1_correlation_strip.pdf). The primary columns use all 576 positions and signed first-reference-token probability differences. Nonbasis/photo filters and existing MMStar caveat flags remain in the wide export; no cases were dropped for this plot.

## 4. Same-image example

The pan/window image **n578564** has identical input E arrays for questions gqa:202218829 (‘Window’) and gqa:202218593 (‘Bottom’). [Combined image](same_image_pair/same_image_comparison.png), [vector PDF](same_image_pair/same_image_comparison.pdf), [individual maps and canvas](same_image_pair/metadata.json), and [576-position data](same_image_pair/grid.csv) are included. The two signed retention maps use the same symmetric scale. Raw retained probabilities are also exported as individual maps with a shared 0–1 scale.

The image is the actual 336×336 model canvas including padding. Every heatmap cell is one of its 24×24 positions (14×14 pixels). These are contextual CLIP features, not independent raw pixel patches. The ‘Bottom’ question has a much larger all-removed probability than ‘Window’; comparing signed maps must account for their different targets and null priors. This illustration supports question dependence, not a causal explanation of an ApET failure.

For this pair, ‘Window’ is scored through tokenizer piece `▁Window`, while ‘Bottom’ is scored through **`▁B`**, not the full word. Their all-visual-removed probabilities are 0.01390 and 0.71827 respectively. Use the figure to discuss the defined first-piece response, rather than confidence in the complete words.

## 5. Exact TextVQA prompting and comparison limits

[Verbatim Brolio prompt](textvqa_prompt.txt) and [protocol details](textvqa_prompt_and_protocol.json) are included. Project prompts use the Vicuna-v1 wrapper, image + question, then `Answer the question using a single word or phrase.` No extra OCR transcript or gold answers are supplied. All 900 primary generation arms in the screening100 archive were checked against that template. The runner uses greedy generation, at most 16 new tokens, batch size 1, FP16.

The local upstream-reference launcher points to `llava_textvqa_val_v051_ocr.jsonl`, and its loader defaults to 128 new tokens. The actual question JSONL is absent locally; its exact prompt contents and parity with the paper are **not verified**. The paper's 58.2 full576 score and our screening100 score of 49.7 use different evaluation cohorts. Do not attribute that 8.5-point gap to prompting/OCR alone without a matched experiment.

## 6. Saved E availability and exploratory loss prediction

All three large-cohort ZIPs contain complete 576-position input E arrays, ten FPS basis source indices, and per-question decoder-stage E arrays. The exports include [input encoding statistics](large_cohort_input_E.csv), [question-linked outcomes](large_cohort_question_E_outcomes.csv), [decoder E statistics](decoder_E_statistics.csv), and [availability/integrity inventory](E_availability.json). GQA shares one encoding over eight questions; MMStar contains repeated image bytes and sometimes different per-case encoding seeds. Validation therefore groups by actual image hash, not question ID.

The [analysis plan](analysis_plan.json) was written before calculating these associations, after the cohorts/outcomes had already been reviewed; it is exploratory, not a confirmatory preregistration. The outcome is loss **conditional on baseline correctness**. Primary E summaries omit the ten FPS source positions. A fixed three-feature ridge logistic model uses log(1 + mean E), E coefficient of variation and the fraction of E mass in the top 10% of positions. Five outcome-independent image-group folds, training-only standardization and no tuning are used. No decoder E is used for the predictor because it is measured after input compression.

| Cohort | Schedule | Eligible / losses | E mean AUROC | Three-feature OOF AUROC [95% interval] | Brier improvement vs prevalence |
|---|---|---:|---:|---:|---:|
| textvqa | 96→32 | 44 / 10 | 0.682 | 0.568 [0.371, 0.776] | +0.0044 |
| textvqa | 288→96 | 44 / 7 | 0.614 | 0.498 [0.263, 0.749] | -0.0002 |
| gqa | 96→32 | 552 / 78 | 0.523 | 0.477 [0.410, 0.538] | -0.0013 |
| gqa | 288→96 | 552 / 34 | 0.496 | 0.410 [0.294, 0.539] | -0.0002 |
| mmstar | 96→32 | 521 / 79 | 0.608 | 0.568 [0.501, 0.634] | +0.0002 |
| mmstar | 288→96 | 521 / 36 | 0.638 | 0.562 [0.457, 0.661] | +0.0001 |

AUROC 0.5 is chance ranking; descriptive feature AUROCs retain their original direction (higher statistic predicts more losses). Brier improvement is prevalence Brier minus model Brier, so positive is better. Intervals resample image groups 1,000 times over fixed scores/predictions and do not include model refitting, split choice, fresh native numerical variation or external confirmation. The analysis is question-weighted. TextVQA has only 44 baseline-full-credit cases and 10/7 losses, so its precision is especially limited.

**Interpretation:** mean E has a modest descriptive association with loss on MMStar (AUROC 0.608/0.638), little association on GQA (0.523/0.496), and uncertain estimates on TextVQA. The fixed three-feature model gives weak and inconsistent out-of-fold rankings, with no clear Brier improvement over the prevalence predictor (all six improvement intervals include zero). This provides no robust cross-dataset adaptive-budget rule. It also does not rule out other features or a useful future policy. Image complexity, question type and model competence may confound these associations; they have not been controlled here.

Input E is question-agnostic **conditional on the image, FPS basis and seed**; it need not be identical across different seeded encodings of the same image. All 1,700 encoding records do have exactly matching input E and FPS basis indices across the two staged schedules. In GQA, 56 images at 96→32 and 27 at 288→96 contain both lost and preserved baseline-correct questions under the shared encoding. An image-only statistic cannot distinguish those outcomes within the image.

[All feature/model metrics](e_loss_feasibility_metrics.csv), [OOF predictions](e_loss_oof_predictions.csv), [folds/coefficients](e_loss_model_folds.json), and [machine-readable summary](e_loss_feasibility_summary.json) are retained. These numbers test feasibility of an image-level failure-risk signal. They do **not** test an adaptive budget policy or demonstrate improved accuracy at matched mean tokens. A useful policy would still require a fixed rule and an independent, matched-token evaluation.

## Sources and reproducibility

The three raw staged archives, the frozen protocols, reviewed manifests and the existing RQ1 synthesis are authoritative sources. [Provenance](provenance.json) records source hashes and the code commit. Raw archives and the existing report are unchanged. Rebuild with `research/reports/midterm/collect_supporting_data.py` in a CPU environment containing NumPy, Pillow and Matplotlib. No large tensors or model weights are opened.
