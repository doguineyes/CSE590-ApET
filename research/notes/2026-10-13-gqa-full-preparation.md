# GQA full Test-Dev preparation — 2026-10-13

Prepared locally while the user runs full MMStar and the GQA diagnostic pilot.
Push/run decision follows pilot review; no commit or push was performed here.

Scope: all 12,578 balanced Test-Dev questions on 398 images, four existing native
configurations, 50,312 scored predictions. This extends dataset coverage, not
ApET's algorithm. Native model files, dependency locks, pilot runner and scorer
remain unchanged. The implementation-scope table now lists the full runner.

Added CPU full-data preparation, a separate resumable evaluator, Kaggle notebook,
guide and seven CPU tests. Evaluation loads one shared model per invocation,
flushes every raw prediction, refreshes summaries every 100 questions, retains
full source question metadata/images and snapshots inference/scoring sources.
Resume validates dataset/code/lock/checkpoint/protocol/scoring identity and the
saved prediction prefix; a torn final write is preserved before removal. It
continues within a partially completed question and avoids repeating saved work.

The full run uses native token-count observation only. Reconstruction errors,
FPS indices and merge assignments remain in the pilot ZIP. Scoring is unchanged
`gqa-short-answer-v1`; full balanced Test-Dev coverage does not make our evaluation
the official full GQA metric suite. Raw answers support later independent rescoring.

CPU verification:

- Seven tests passed, including interruption/recovery, duplicate/corrupt record
  rejection, scoring/protocol mismatch rejection, and lost/gained/type APR.
- Actual pinned parquet files were reused locally without another download.
  All 398 raw images decoded and all 12,578 source rows were retained, including
  repeated text with distinct IDs. Question groups range from 3 to 92 questions.
- Prepared directory: `research/results/gqa-testdev-full-v1-s590/`, ignored by Git.
  Total bytes: 85,090,073. Manifest SHA256:
  `67f955802a5ad7d42dba9fed902eb6f0f04c6109eb06804b4b52407bb4699066`.
- All 50,312 expected question/configuration pairs are unique.
- A synthetic 50,312-row summary took about 0.95 seconds locally and serialized
  to about 3.55 MiB. This checks summary scalability, not inference performance.
- New files parse with Python 3.10 grammar; all four notebook code cells compile.
- All eight monitored native source hashes match the passed-pilot reference.

Pretrained full inference has not run. Next: inspect pilot answer normalization,
observer parity and actual output lengths/timings; estimate full-run GPU cost;
then review/push the prepared version and enable its explicit start flag. Keep
the diagnostic pilot and full-run artifacts together for later RQ1 analysis.
