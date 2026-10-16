# RQ1 GQA calibration review — 2026-10-16

The user resolved the silent Kaggle “Adding data…” session by closing and
reimporting the notebook. That workaround succeeded; the underlying Kaggle cause
was not established. The first model calibration then completed. No optional
12-question GQA, MMStar or TextVQA run was performed.

## Preserved evidence

- `research/results/rq1-gqa-calibration-20261016T133855495304Z.zip` and adjacent
  `.log`, copied byte-for-byte from Downloads; original files retained.
- Extracted folder of the same name, including native source snapshots, original
  lock, accepted GQA manifest, cached image feature tensors, plots and raw scores.
- Commit `c906e28b04b0f223ad1501587e87e4b8bd4f1b2c`, clean runtime working tree.
  Original lock SHA256
  `6458cb0ab526857c483b41c19cb87e63179507ebaf6e48f4e9020d18d261f3a2`.
- All 37 ZIP entries passed CRC validation. Every recorded evidence-file digest
  was checked. The eight native monitored sources match the passed reference.

## Execution results

Two questions about image `n200692`, the same 32 uniformly sampled patch indices,
identical CLIP/projector feature hashes and archived image seed `1982176349`:

| Question | Fresh uncompressed answer | Literal measured target | Spearman E vs signed I |
|---|---|---|---:|
| `201636988`: cake vs plate color | `No` (scored correct) | `no` | 0.06014 |
| `201637316`: plate lies on what? | `Stove` (scored correct) | `stove` | 0.10924 |

All unperturbed-repeat, clone-only no-op, cached/native-forward and final-reference
score differences were **exactly zero** in this run. This supports execution and
numerical repeatability for these cases; it does not establish semantic target
alignment or general determinism.

End-to-end wall time was 150.58 seconds including checkpoint downloads and
24.71 seconds loading the model. The question measurement loops took 20.92 and
17.73 seconds. Mean singleton-forward times were 0.566 and 0.480 seconds.
Peak allocated GPU memory was 7.43/7.27 GiB. A simple token-forward extrapolation
for 12 questions × 576 patches is about one hour, excluding variable prompt,
startup, plotting and I/O effects; this is a planning estimate, not a benchmark.

The saved figures render and are aligned to the actual padded 336×336 CLIP canvas.
Tables preserve all 576 E values, with I=null for the 544 unmeasured patches.

## Measurement concern: answer formatting

Literal target probability (sequence likelihood without EOS) was approximately
`2.77e-6` for lowercase `no` and `2.43e-5` for lowercase `stove`, despite correct
greedy answers `No` and `Stove`. The original runner appended the unchanged raw
lowercase annotation. It did not check reference-token agreement with the actual
generated continuation. Numeric no-op/native-parity controls can pass while this
formatting mismatch remains.

All 32 cake-question I values were negative; 31/32 stove I values were negative.
Across the two questions the sensitivity ranks had Spearman 0.93628. These are
observations about the literal-string zeroing proxy. They do not show that image
tokens generally harm answers, that E is unrelated to task contribution, or that
ApET's mechanism is disproved. Capitalization/tokenization is a plausible confound,
not yet an experimentally confirmed explanation for these values. Padding token
558 was among the strongest negative examples in both questions, another reason
to inspect the measurement rather than immediately claim visual causality.

## Next action

Hold the 12-question intervention run. A new **target-format audit** mode runs
only these same two questions with no patch interventions. It records actual
native generated IDs and processed score probabilities, compares raw-label and
initial-capital spellings under teacher forcing, checks repeats and first-token
agreement, and includes the generated response only as an explicitly post-inference
diagnostic. No generated answer is automatically promoted to a gold label.

The audit is project-owned instrumentation. Native ApET sources, checkpoint
revisions, prompts, image inputs, labels, curation and ML dependency locks remain
unchanged. Input bundle `rq1-inputs-v1.zip` is reusable. Native/GPU validation of
the new observer remains for Kaggle; local CPU tests cover candidate rules and
unchanged gold labels. See [run instructions](../RQ1_INPUT_PILOT.md).

After the audit, define/version the reference-scoring policy explicitly and rerun
the small calibration if needed. Expand only after scoring alignment and controls
are satisfactory. Then continue with the 12-question GQA pilot, MMStar confirmation
and the retained TextVQA baseline; decoder-layer interventions remain later work.
Historical results stay intact and are not relabeled as corrected measurements.
