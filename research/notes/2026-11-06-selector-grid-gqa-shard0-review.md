# GQA selector-grid shard0 review — 2026-11-06

Archived ZIP/log: selector-grid-gqa-full-v1-20261106T174104944642Z in research/results. Copies verified by SHA-256, ZIP CRC passed, all345 internal artifact hashes passed. Source files in Downloads retained.

Despite `full` in the run name, this is exactly shard0:200 questions on25 images (eight questions/image), not the full800. All200 question records passed;2800/2800 predictions complete. Runtime1433.91 seconds (~23.9 minutes). The set of IDs exactly matches frozen protocol shard0.

| Condition | Accuracy % |
|---|---:|
| Full576 | 70.0 |
| Native96→32 | 62.0 |
| Random drop, five-seed mean | 61.6 |
| Nonpad random drop, matched five-seed mean | 65.2 |
| Detail drop | 59.5 |
| E drop, no basis | 62.0 |

Random range59.5–65.0; nonpad range63.0–70.0. Fixed10000 paired image-cluster bootstrap draws with seed590, averaging seeds inside each question, preserves eight-question grouping:

- Detail − random: −2.1pp,95% interval[−7.9,+3.4].
- Nonpad − random: +3.6pp,[+2.1,+5.3].
- E − random: +0.4pp,[−3.0,+4.0].
- Nonpad − native: +3.2pp,[−0.9,+7.4].

These intervals condition on the tested seeds and on a small25-image shard; no seed-population uncertainty or multiplicity adjustment. They are preliminary, not final GQA800 results.

Native loses24 of140 baseline-correct answers and gains8, APR82.86%. E-drop also scores124/200 but loses26 and gains10, APR81.43%. Detail loses33 and gains12, APR76.43%. Equal accuracy does not imply identical preservation.

Interpretation: The large TextVQA detail advantage is absent on this first GQA shard. Nonpad improves the random control; E-drop remains close to random. This suggests task dependence but cannot establish it conclusively before completing the frozen cohort. This grid has no GQA detail-merge condition, so it does not test whether detail+merge competes with native on GQA.

Next: continue shards[1,2,3] in the existing notebook without rerunning0 or changing the condition grid. Save each ZIP/log. Expected duration about72 minutes extrapolating this chunk; actual runtime can vary. Aggregate all four disjoint shards by question/image, not by treating seeds or questions from one image as independent samples.

Machine-readable review: research/results/selector-grid-gqa-shard0-review.json.

ZIP SHA-256 f9dee720dd06db5626e549e469e4fb191a2cdba844fdbd97872d133163ac3734
Log SHA-256 b2cd235b6cbb45d904ed5a7bd90a8fd5af146d2f33fcaf8f9a83f2d5523a7cae
