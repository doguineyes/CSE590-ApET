# 2026-10-13: native ApET MMStar pilot and scoring audit

Evidence: `research/results/original-mmstar-24-20261013T172800287175Z.zip`.
Commit `9bc740dc2a61ab55cbac857cd130f5af97e4d9de`, clean Kaggle checkout.
Original lock SHA256:
`6458cb0ab526857c483b41c19cb87e63179507ebaf6e48f4e9020d18d261f3a2`.

All 96 predictions completed: 24 images, four per MMStar category, across four
configurations. The requested reductions occurred; the baseline repeat matched.

| Configuration | Archived strict correct | v2 correct | v2 APR | Median generation seconds |
|---|---:|---:|---:|---:|
| Baseline | 9/24 | 9/24 | Reference | 0.702 |
| Input only | 7/24 | 8/24 | 7/9 = 77.8% | 0.505 |
| Decoder only | 9/24 | 9/24 | 9/9 = 100% | 0.530 |
| Combined | 7/24 | 8/24 | 7/9 = 77.8% | 0.435 |

Sample 6 returned `A: A woman in different polka dot dresses` under input-only and
combined. It names the correct option and copies its text, but the old parser
marked it unparsed. The v2 rule accepts strict choices and explicit choices plus
matching option text; arbitrary prose/conflicting answers stay unparsed. Answer
format compliance is separate. Neither policy is official MMStar scoring.

CPU rescoring created `original-mmstar-24-rescored-v2.json`, `.jsonl`, `-images/`,
and `.zip` in `research/results/`. Only those two rows changed score. No inference
was rerun. The source ZIP hash was verified unchanged. Raw answers, prompts, seeds,
traces, and generation timing are preserved; prior scores and scorer provenance
are retained in the new records. These result files are ignored by Git.

Baseline and decoder-only raw answers matched on all 24 examples. Input-only and
combined also matched on all 24, despite active decoder pruning. This is a small
pilot, not a general preservation result. After rescoring, input compression lost
baseline-correct IDs 12 and 137 and gained ID 358. ID 1274 changed between two
wrong choices. ID 137 changed dominant color from blue to red: a possible case of
distinctive foreground features overriding redundant background. Reconstruction
scores, token maps, and interventions are needed to test that hypothesis.

Model load: 5.254 seconds; warmup: 3.1 seconds; scored wall: 56.232 seconds including
an unscored repeat. Variable output lengths/diagnostic overhead limit timing
interpretation. Decoder-only GPU 0 peak was 9.152 GiB versus baseline 8.134 GiB;
compression does not guarantee lower peak memory.

Next: [full MMStar run](../ORIGINAL_MMSTAR_FULL.md), then dataset/error analysis
and RQ1 instrumentation. The [implementation scope](../IMPLEMENTATION_SCOPE.md)
separates original ApET, the existing mask-device fix, and project evaluation code.
The full GPU run is prepared, not yet executed locally.
