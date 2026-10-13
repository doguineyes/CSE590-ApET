# Fixed GQA study subsets — 2026-10-13

The user chose to stop the optional full Test-Dev run because four configurations
would require roughly eight hours. Preserve its partial ZIP; its predictions
are not overwritten or silently treated as a completed baseline. Repeated main
experiments now use smaller frozen sets, matching the proposal's intended scope.

Prepared from the same pinned labeled balanced Test-Dev source, without reading
model predictions or selecting errors:

- Development: 100 images × 8 questions = 800 questions.
- Reserved evaluation: 100 separate images × 8 questions = 800 questions.
- Diagnostics: 20 development images × 4 questions = 80 questions; nested in development.
- 198 source images remain unused by these two main sets.

Image IDs and raw image hashes do not cross development/evaluation. Previously
inspected pilot images are excluded from evaluation. The original 40 pilot
questions are retained in the default diagnostics. Sampling deliberately provides
equal questions per selected image and type diversity; these scores must not be
presented as full-population GQA accuracy. Future error cases come only from
development, with unchanged-answer controls and same-image question groups.

Study ID:
`3153bd275dbefd346e132c18ac16d7cbe46e5534c4f9e84fed2786abe038f512`.
Root study manifest SHA256:
`81a49ac9a5f534aa43c1de51eb843284e53515d4d46773f933948a4afea3f962`.
Real prepared directory: `research/results/gqa-fixed-subsets-v1-s590/`.
Portable ZIP: same path with `.zip`, 37,442,850 bytes. Both are ignored by Git.
Kaggle prepares these sets automatically; uploading the bundle is optional.

Semantic type composition:

| Set | attr | cat | global | obj | rel |
|---|---:|---:|---:|---:|---:|
| Development | 352 | 76 | 39 | 98 | 235 |
| Evaluation | 355 | 95 | 28 | 91 | 231 |

Added `prepare_gqa_subsets.py`, `eval_original_gqa_subset.py`, a four-cell Kaggle
notebook, guide and seven CPU tests. The evaluator shares existing project
generation/scoring/record-validation/summary helpers. It does not copy or replace
ApET's compression math. The full runner and old diagnostic runner are unchanged.
The native reference, checkpoint revisions, dependency locks, prompts, seeds,
FP16 greedy generation and short-answer scorer are unchanged.

Initial development baseline is 3,200 predictions across the existing four
configurations, one model load. The pilot suggests about 30 minutes, with actual
runtime dependent on question mix/output length. Future selected configurations
can use a completed matching-subset baseline as a read-only reference. Reports
distinguish new and saved predictions; stored baseline latency is historical.
Original native guards still apply; new algorithm variants require explicit
implementation/provenance, not silently edited baseline files.

CPU verification passed:

- Seven tests cover deterministic sampling independent of source row order,
  image separation, diagnostic nesting, source-data tampering, immutable saved
  baseline reuse, protocol/subset mismatch rejection and interrupted recovery.
- All real development and evaluation manifests and images were validated.
- The existing observed pilot loader accepts the real 80-question diagnostic manifest.
- All eight monitored original native hashes match the passed reference.
- New Python files parse with Python 3.10 grammar; four notebook code cells compile.

No new pretrained inference ran on the Intel Mac. Next: stop/download the old
run, commit/push, use `kaggle_original_gqa_subsets.ipynb` with development selected
and `RUN_SUBSET_TEST = True`, then download and audit its completed baseline ZIP.
Keep reserved evaluation for later confirmation after tuning is frozen.
