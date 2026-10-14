# Native GQA repeatability results — 2026-10-14

Evidence: `research/results/original-gqa-repeatability-20261014T172413176023Z.zip`.
CPU audit: `research/results/original-gqa-repeatability-audit-20261014.json`.
The ZIP was read without modifying it. No inference or experiment-code changes
were performed during this analysis.

## Audit and run identity

The experiment completed all 72 planned trials in one model process, including
20 trials with native-value observations. All trial IDs, phase order, settings,
token-count traces, scores, diagnostic fingerprints and report summaries match
the saved JSONL. ZIP CRC and prediction/manifest/source/lock hashes pass.
Both real images match their historical raw bytes, decoded RGB hashes and sizes.
All 26 recorded source snapshots match their hashes and current local sources;
all eight monitored native files match the passed reference.

Clean Kaggle commit: `5881fb334ba061d356ab007b73bb29d2a9e4dd04`.
Python 3.10.18, Torch 2.1.2+cu121, Transformers 4.37.2, original dependency lock
`6458cb0ab526857c483b41c19cb87e63179507ebaf6e48f4e9020d18d261f3a2`.
The two T4s use driver 580.178.04. Model/CLIP revisions and the recorded model
device map/component devices match both pilot and development runs.
Questions, prompts, seeds, settings, image hashes and labels also match those
historical records for both cases and all four configurations.

Recorded trial wall time is 38.536 seconds; model loading is 29.431 seconds.
Downloads, setup and four short warmups are outside trial time. Diagnostic and
recording overhead make this a repeatability experiment, not a latency benchmark.

## Answer drift reproduced inside one process

Question `2058428`, image `n497658`:
`Are there airplanes in this scene that are huge?` Gold: yes.

| Configuration | Recorded trials | Answers | First three consecutive counts-only repeats |
|---|---:|---|---|
| Baseline | 8 | Yes × 8 | Yes, Yes, Yes |
| Input-only | 10 | No × 7; Yes × 3 | No, No, Yes |
| Decoder-only | 8 | Yes × 8 | Yes, Yes, Yes |
| Combined | 10 | No × 6; Yes × 4 | Yes, Yes, Yes |

These counts combine several different control blocks. They are descriptive
trial counts, not independent estimates of accuracy or answer probabilities.

Input-only trials 7, 8 and 9 are consecutive calls with identical question,
configuration and seed, and without full native-value tracing. The No → Yes
change therefore does not require switching configurations, restarting a session,
changing dependencies or enabling the full diagnostic tracer. Counts-only still
uses the existing token-count hooks. This does not establish what operation causes
the change or exclude a contribution from execution state/recording overhead.

Logging sandwich, input-only trials 10 → 11 → 12: No → No → Yes. Thus even the
two counts-only endpoints differ. The result cannot be explained simply by
native-values logging being on versus off.

The combined counts-only return control, trials 28 → 29 → 30, answers
No (combined) → Yes (baseline) → Yes (combined). It records another flip, but
switching itself is not established as the cause because consecutive input-only
calls already vary.

## Stable observations can accompany different answers

Observed input-only trials 11, 31 and 33 answer No, Yes and No. Their complete
native diagnostic fingerprints are identical:

`a25513cfc7df73f35b2bf4c5714a902b8494874372afe1f748826e66f55e29e3`.

This includes exact reconstruction-error arrays, FPS bases, selected indices,
basis mappings and merge assignments/counts. This particular answer flip is not
explained by choosing a different recorded token set or FPS basis.

It does **not** prove identical numerical feature tensors. The existing observer
does not save/hash the merged feature values, projected features, decoder hidden
states or output logits. Equal merge targets/counts can still accompany different
accumulated feature values. Identical error norms also do not identify all feature
components. The first differing tensor/operation remains unknown.

## Combined decoder observations

All three observed combined airplane trials answer No, while their decoder-stage
reconstruction errors differ. Relative to trial 23, the maximum absolute error
differences are 0.0102787 (trial 34) and 0.0182457 (trial 36). Input diagnostics,
retained decoder sets and FPS bases remain identical across these observations.

The logger's `merge_assignments_equal` compares the whole merge record, including
ordered target lists, counts and basis-slot ordering. A False value is not by
itself evidence that discarded tokens were sent to different retained targets:

| Comparison | Actual dropped-to-target mapping | Counts per target | Basis-slot-to-source mapping |
|---|---|---|---|
| Airplane 23 → 34 | Same | Same | Same |
| Airplane 23 → 36 | Same | Same | Two assignments change |
| Lamp 59 → 70 | Same | Same | Two assignments change |
| Lamp 59 → 72 | Same | Same | Two assignments change |

For airplane trial 36, the same basis-slot set is present, but slots 129 and 160
swap order relative to trial 23. Native decoder basis substitution associates FPS
features with these ordered slots, so the two slot-to-source assignments change.
These indices address the entering decoder stream, not automatically original
image patches. Pure target/count list reordering is distinguished from this
feature-identity change in the audit.

Different internal values do not necessarily change the final answer: these
observed combined airplane outputs are all No, and the lamp outputs remain Right.
Do not attribute the counts-only combined Yes/No flips to a particular observed
decoder difference; full diagnostics were not collected on those counts-only calls.

## Control question

Question `201882614`, image `n125122`:
`Is the lamp on the right or on the left of the photo?` Gold: left.

All answers match both historical runs: baseline Left × 8, input-only Right × 10,
decoder-only Left × 8, combined Right × 10. The compressed incorrect answer is
stable in this experiment. Combined decoder diagnostics still vary internally,
so answer stability is not the same as identical recorded internal computations.

This control supports case-specific sensitivity; two selected cases cannot
estimate how prevalent unstable answers are across the development set.

## Candidate explanation, not a demonstrated cause

The run records `deterministic_algorithms = False` and no
`CUBLAS_WORKSPACE_CONFIG`. Native input merging calls floating-point
`scatter_add_` after casting image features back to FP16
(`llava/model/llava_arch.py`, around line 183). Native decoder merging also uses
`scatter_add_` (`llava/model/modeling_llama_x.py`, around line 117).

The version-matched [PyTorch 2.1 scatter-add documentation](https://docs.pytorch.org/docs/2.1/generated/torch.Tensor.scatter_add_.html)
states that CUDA execution may be nondeterministic. Its
[2.1.2 CUDA kernel source](https://github.com/pytorch/pytorch/blob/v2.1.2/aten/src/ATen/native/cuda/ScatterGatherKernel.cu)
identifies atomic addition as the reason in the default scatter-add kernel.
These sources make numerical accumulation a concrete candidate worth testing;
the current artifacts do not demonstrate that it caused this answer change.

A fixed seed controls seeded random draws; it does not enforce all deterministic
GPU algorithms. The [PyTorch 2.1 reproducibility guide](https://docs.pytorch.org/docs/2.1/notes/randomness.html)
separates these controls. No first-token scores were saved here, so a close Yes/No
decision margin is another hypothesis, not a measured finding.

## Implication and next check

The original baseline archives remain valid records of their declared runs.
This check establishes an unresolved repeatability issue in an input-compressed
case. We should repeat selected loss/gain cases before using their individual
answer changes as causal evidence about token relevance. It does not establish
that all 34 development losses or 26 gains are unstable, or that compression is
generally ineffective.

The next useful experiment is smaller and more targeted than another baseline:

1. Record output feature fingerprints/values after native merging and projection,
   plus first-token scores for competing answers and generated token IDs. Locate
   the earliest observed difference on repeated input-only airplane trials.
2. Compare the current numerical policy with an explicitly recorded deterministic
   policy while keeping native math, weights, seed, prompt and placement fixed.
   The [version-matched deterministic API](https://docs.pytorch.org/docs/2.1/generated/torch.use_deterministic_algorithms.html)
   lists CUDA scatter-add among operations with a deterministic alternative and
   describes the required cuBLAS workspace configuration for affected matrix
   operations. Configure the subprocess before CUDA initialization; preserve any
   unsupported-operation errors as evidence. Do not silently change the baseline
   policy or treat stable repeated execution as proof of a particular cause.
3. If needed, isolate the merging operation with fixed captured inputs. Global
   deterministic settings affect several operations, so a successful global
   control alone would not specifically prove scatter-add causation.

No algorithm modification or full MMStar/GQA rerun is needed now. These proposed
controls have not been implemented or run in this analysis. Keep the raw archive
unchanged and reserved evaluation untouched.

References: [protocol](../GQA_REPEATABILITY.md),
[development baseline](2026-10-13-original-gqa-development.md),
[preparation checks](2026-10-14-gqa-repeatability-preparation.md).
