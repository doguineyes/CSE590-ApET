# Selector-grid TextVQA100 review — 2026-11-06

Run: selector-grid-textvqa-full-v1-20261106T165132846914Z

Saved ZIP and log with SHA-256 copy verification; ZIP CRC and all 470 internal artifact hashes passed. Status passed: 100 questions, 3100/3100 predictions, 1802.61 seconds. All 100 CPU-reference selection checks passed; identity and recorded replay checks passed. All paired input masks match.

| Condition | Mean consensus score (%) |
|---|---:|
| baseline576 | 49.70 |
| native-96-to-32 | 41.50 |
| grid-random-drop-s101 | 17.10 |
| grid-random-drop-s202 | 23.00 |
| grid-random-drop-s303 | 21.80 |
| grid-random-drop-s404 | 20.50 |
| grid-random-drop-s505 | 18.10 |
| grid-random-merge-s101 | 23.10 |
| grid-random-merge-s202 | 30.90 |
| grid-random-merge-s303 | 25.10 |
| grid-random-merge-s404 | 24.30 |
| grid-random-merge-s505 | 29.80 |
| grid-nonpad-drop-s101 | 29.90 |
| grid-nonpad-drop-s202 | 28.30 |
| grid-nonpad-drop-s303 | 26.50 |
| grid-nonpad-drop-s404 | 21.90 |
| grid-nonpad-drop-s505 | 18.70 |
| grid-nonpad-drop-s606 | 21.40 |
| grid-nonpad-drop-s707 | 20.90 |
| grid-nonpad-drop-s808 | 18.00 |
| grid-nonpad-drop-s909 | 21.70 |
| grid-nonpad-drop-s1010 | 22.60 |
| grid-nonpad-merge-s101 | 32.70 |
| grid-nonpad-merge-s202 | 31.30 |
| grid-nonpad-merge-s303 | 26.90 |
| grid-nonpad-merge-s404 | 27.30 |
| grid-nonpad-merge-s505 | 29.70 |
| grid-detail-drop-s0 | 38.10 |
| grid-detail-merge-s0 | 43.50 |
| grid-error-drop-s0 | 20.00 |
| grid-error-merge-s0 | 33.00 |

Random drop five-seed mean 20.10; random merge 26.64. Nonpad drop ten-seed mean 22.99 (matched first five 25.06); nonpad merge five-seed mean 29.58. Use matching seeds for paired contrasts.

Detail drop 38.10 / merge 43.50; no-basis E drop 20.00 / merge 33.00; native 41.50; full576 49.70. These are descriptive cohort results, not significance claims. Detail is promising; padding avoidance alone does not close the native gap. Native versus no-basis E changes basis handling and merge targets, and merging can change subsequent E masks: do not attribute the gap solely to basis insertion. Detail versus native +2.0 points is not established superiority.

Historical drift: native textvqa:35257 changed 2010 to 2015, score unchanged (both wrong). Baseline and five comparable random-drop arms match archived generated IDs. Preserve this drift rather than retrying to obtain an answer.

Proceed with planned GQA drop-arm replication in 200-question shards. Start shard0, inspect completion, then 1–3 without repeating 0. Existing GQA grid does not test merge benefits; it tests whether selector differences generalize beyond OCR. No measurement-code change is required before running.
