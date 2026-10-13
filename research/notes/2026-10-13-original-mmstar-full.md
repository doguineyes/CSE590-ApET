# Original ApET full MMStar baseline — 2026-10-13

Evidence: `research/results/original-mmstar-1500-20261013T182319548616Z.zip`.
Independent CPU audit: `research/results/original-mmstar-1500-audit-20261013.json`.
The source ZIP was read without extraction or modification. No inference or
rescoring policy changes were needed; saved scores were recomputed for validation.

## Run identity and completeness

Kaggle commit `b7d10355d923d0cc03f7222ee2b37859af947ff2`, clean checkout.
Python 3.10.18; original dependency lock SHA256
`6458cb0ab526857c483b41c19cb87e63179507ebaf6e48f4e9020d18d261f3a2`.
Original-format LLaVA-1.5-7B and CLIP revisions match the pilot. One native
`LlavaLlamaForCausalLM_X` instance was sharded across both T4s, with SDPA and FP16.

All 6,000 predictions completed: all 1,500 IDs under each of four configurations,
with 250 examples in each of six categories. No duplicate question/configuration
pairs or failed generation records were found. Input and decoder token-count
traces match requested budgets on every record. Prompts, questions, gold labels,
images and seeds match across configurations. The unscored baseline repeat passed.

Archive CRC validation passed. All eight monitored native source snapshots match
the passed reference, including the existing mask-device compatibility patch.
Scorer and dependency lock hashes agree with their archived snapshots. All 1,500
image files match byte hashes, dimensions and locally decoded RGB pixel hashes.
There are 1,430 distinct image byte hashes; multiple MMStar questions can reuse
an image, so image files per sample are not necessarily unique photographs.

All v2/strict scores, aggregate/category summaries and paired comparisons match
independent recomputation. Every raw answer and parsed choice on the previous
24-sample pilot subset matches this full run, under all four configurations.

## Results

APR is the fraction of the baseline's 521 correct answers that remain correct.
Losses/gains below compare correctness against baseline on exactly paired IDs.

| Configuration | Token settings | Correct / 1,500 | Accuracy | APR | Lost / gained | Mean generation seconds |
|---|---|---:|---:|---:|---:|---:|
| Baseline | Input 576; no decoder pruning | 521 | 34.73% | Reference | — | 0.801 |
| Input-only | Input 288; no decoder pruning | 524 | 34.93% | 93.09% | 36 / 39 | 0.545 |
| Decoder-only | Input 576; layer 16 → 96 | 520 | 34.67% | 98.08% | 10 / 9 | 0.592 |
| Combined | Input 288; layer 16 → 96 | 524 | 34.93% | 93.09% | 36 / 39 | 0.467 |

Scoring is project policy `mmstar-explicit-choice-v2`, not the official evaluator.
Strict correct counts are respectively 517, 519, 515 and 519. All responses parse
under v2; 7–9 per configuration do not comply with the exact single-letter output
format, but explicitly name an option and match its text. Only sample 62 reaches
the 16-token output maximum, under every configuration; its answer still parses.

The three extra correct answers under input compression are an observed
0.20-percentage-point difference, not evidence of a general accuracy improvement.
Nearly equal aggregate accuracy hides lost baseline-correct examples. These
paired losses are useful candidates for RQ1 diagnostics and later interventions.

| Category (250 questions each) | Baseline accuracy | Input-only | Decoder-only | Combined | Combined APR |
|---|---:|---:|---:|---:|---:|
| Coarse perception | 56.0% | 58.4% | 57.2% | 58.4% | 96.43% |
| Fine-grained perception | 20.8% | 23.2% | 21.6% | 22.8% | 86.54% |
| Instance reasoning | 44.4% | 44.0% | 43.6% | 44.8% | 93.69% |
| Logical reasoning | 32.4% | 30.8% | 32.8% | 31.2% | 90.12% |
| Math | 26.0% | 25.2% | 25.2% | 24.8% | 95.38% |
| Science & technology | 28.8% | 28.0% | 27.6% | 27.6% | 91.67% |

Fine-grained perception has the lowest combined APR, despite a higher aggregate
category accuracy. Its 7 lost and 12 gained answers illustrate why preservation
and net accuracy must be reported separately. Category denominators differ in
baseline-correct counts; these observations do not establish significant
differences or the causal role of reconstruction error.

Decoder-only changes 27 parsed predictions relative to baseline, including 10
losses and 9 gains. Input-only and combined differ on 10 parsed predictions,
including 6 correctness changes. Their identical overall accuracy/APR therefore
does not mean decoder pruning has no effect. Exact differing IDs and full
lost/gained lists are retained in the audit JSON.

## Runtime and memory

Scored wall time across all four configurations: **3,986.235 seconds / 66.44 minutes**.
This excludes environment/data/weight setup, model loading (24.098 seconds) and
warmup (5.268 seconds), and includes recording plus one unscored baseline repeat.
Summed recorded generation time is 3,608.762 seconds / 60.15 minutes. Generation
includes vision encoding, compression, prefill, decoding and token-count hooks.

Combined mean generation time is about 42% lower than baseline (observed 1.71×
ratio). Mean output lengths are almost identical, roughly 2.05 tokens, but this
remains a single run with hooks and varying outputs, not a controlled repeated
benchmark or a guaranteed speedup for longer answers.

Peak allocated GPU memory, maxima across questions:

| Configuration | GPU 0 GiB | GPU 1 GiB |
|---|---:|---:|
| Baseline | 8.134 | 7.608 |
| Input-only | 8.134 | 7.451 |
| Decoder-only | 9.279 | 7.346 |
| Combined | 8.134 | 7.346 |

This run fits comfortably on the two T4s. Decoder-only has a higher GPU 0 peak
than baseline, so fewer retained tokens do not guarantee lower peak allocation.
These are PyTorch allocated-memory measurements, not total device or disk use.

## Answer-position bias and next research steps

Baseline predicts A on **1,059/1,500 = 70.6%** of questions, compared with
**429/1,500 = 28.6%** gold A answers. All configurations show a similar imbalance.
It is especially pronounced in math (230/250 predicted A, 68/250 gold A).
The audit confirms these are actual parsed responses, not a fallback for failed
parsing. This is a baseline observation; its cause is not established. Model,
prompt and option-order effects should be checked before attributing every error
pattern to visual compression. A later deterministic option-order control can
test this while preserving today's baseline and native code.

This is a complete, internally consistent baseline for the declared settings,
not a full reproduction of all ApET paper experiments or proof of the algorithm's
correctness on every model/dataset. Continue with the already-prepared GQA pilot
and review its answer scoring and native observations before committing to full
GQA. GQA short answers avoid this particular A/B/C/D output-position issue, while
introducing their own short-answer matching considerations. For RQ1, examine
lost/gained examples and reconstruction/selection/merge evidence, then run
controlled answer-contribution interventions. Preserve this ZIP unchanged.
