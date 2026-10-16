# RQ1 target-format audit and measurement v2 — 2026-10-16

The target audit **confirmed** the format mismatch suspected in the first
calibration. Its native generation and teacher-forced first-token log probabilities
agreed exactly. The earlier low reference probabilities arose from scoring the
literal lowercase annotations rather than the model's usual capitalized answer
forms in these two cases. This diagnoses our measurement representation, not an
ApET compression or dependency failure.

## Audit evidence

Preserved original:
`research/results/rq1-gqa-target-audit-20261016T140244407717Z.zip`, with extracted
folder of the same name. ZIP CRC and every recorded source/evidence hash were
verified. Commit `372b360b29f2a90ea0946947501ec7f3b16c8723`, clean Kaggle worktree,
original lock SHA256
`6458cb0ab526857c483b41c19cb87e63179507ebaf6e48f4e9020d18d261f3a2`.
All eight monitored native files match the passed reference. Audit wall time:
17.98 seconds with zero missing checkpoint-download bytes.

| Question | Literal annotation / token IDs | Initial-capital form / token IDs | Exact target-sequence probability, raw → capital |
|---|---|---|---:|
| `201636988` | `no` / `[694]` | `No` / `[1939]` | 0.000002774 → 0.515563 |
| `201637316` | `stove` / `[380,994]` | `Stove` / `[6639,345]` | 0.000024254 → 0.529833 |

These are exact string-prefix sequence probabilities, without EOS, not calibrated
semantic-answer probabilities. Both capitalized reference token sequences matched
the actual native generated answer prefix. Every target-form repeat difference
was zero. First-token generation/teacher-forcing differences were zero; the second
`Stove` token differed by about 0.00000472 nats under cached generation vs full
teacher forcing, well below the existing 0.001-nat tolerance.

The cake question's initial-token scores were `No`=-0.66250 and `Yes`=-0.72500
(0.0625-nat margin). Its correct greedy answer was close to a competing answer;
that matters when interpreting later compression flips. This observation does
not identify which patch causes a flip or prove an image-grounding mechanism.

The audit measured no interventions on capitalized forms. It therefore does not
establish how the old E–I correlations or negative-I pattern change. The first
calibration remains saved as a v1 literal-target diagnostic and is not relabeled.

## Implemented fix

- Measurement protocol is now `rq1-input-zero-v2`. Portable input version remains
  v1 and the existing `rq1-inputs-v1.zip` is unchanged/reusable.
- GQA reference rendering applies `raw[:1].upper() + raw[1:]`; it preserves the
  rest of the original label. MMStar option targets remain literal. TextVQA
  generation scoring is unchanged; its intervention target policy is still deferred.
- Rendered reference targets for every selected question are materialized before
  generation. No per-question target is chosen from model output or perturbed
  scores. Raw labels/manifests remain byte-for-byte unchanged.
- For fresh baseline-correct questions, the fixed target IDs must equal the
  native generated answer prefix. A mismatch is saved and stops the token loop.
  For matching prefixes, target log probabilities from generation and teacher
  forcing are compared within the existing control tolerance. Baseline-wrong cases
  retain their fixed gold target rather than adopting the wrong prediction.
- Records/tables/plots identify the raw and measured targets and scoring version.
  Original no-op, repeated references, native parity, native E, FPS seed, features,
  token sampling and singleton-zeroing definition are preserved.

This is a fixed spelling convention for the pilot, not full marginalization over
case variants/synonyms. Prefix/score checks can reveal another incompatible output
format; they do not establish semantic equivalence or visual grounding.

## Next run and validation

Rerun **the same two GQA questions × the same 32 patches** with v2, using a new
timestamped result name. No new dataset, environment install or baseline rerun is
needed in the live session. Only after its alignment/numeric controls and I values
are inspected should the 12-question × 576-token pilot proceed. Instructions:
[RQ1 input pilot](../RQ1_INPUT_PILOT.md).

Local validation includes deterministic reference rendering, unchanged annotations,
alignment rejection using real saved audit IDs, and baseline-wrong behavior.
Native GPU execution of the corrected intervention path remains to be checked
on Kaggle. The audit validates the capitalized token sequences, not a completed
v2 intervention result. Native ApET and dependency locks remain unchanged.
