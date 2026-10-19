# First RQ2 pretrained parity failure — 2026-10-19

The supplied failure is preserved. This is not a missing input or model file:
the checkpoint loaded, native prerequisites passed, and all three arms executed
for the first question. The pilot stopped on an exact feature-fingerprint gate.
It has **zero admitted completed questions**, not a valid one-question RQ2 result.

## Archived files and independent checks

Copied byte-for-byte into `research/results/`, retaining Downloads originals:

- `rq2-input-merge-pilot-v1-20261019T185452294158Z.zip`: 2,213,140 bytes;
  SHA256 `26f8becc7a885cb50c7a320fb090b4ed54b046ef359e28b805bd130d863811ab`.
- Matching `.log`: 2,319 bytes;
  SHA256 `3ed3fc2f6c178fbe511801f6d7fde5dbe271e6e25d2f9cebe779119b70cd3baf`.

All 109 unique ZIP members pass CRC; all 108 artifact-inventory hashes, 101
evidence hashes and embedded measurement-source hashes verify. Runtime commit
`78422984246ea93fded0c0f653abd2e806cadd72` was clean. The native reference and
original environment lock match. The embedded setup log confirms **10 native
tests passed** on Kaggle, including the tiny real-model reference-score test.
Wall time was 14.43 seconds, including loading. This is not a performance result.

The separate [saved-data verification](../results/rq2-input-merge-failed-review-v1-2026-10-19/verification.json)
and failed question/setup copies preserve this CPU review without modifying the
raw ZIP or running a model locally.

## What matched

On `gqa:201636988`, the native and no-merge arms have exactly identical CLIP/source
patch hashes, pre-merge feature hashes, all E values, FPS source IDs, ranked selected
positions, masks and basis source-to-output-slot assignment. No-merge's pre/post
hashes are identical. The native E, FPS basis and merge assignments/counts also
match the archived reference exactly. The full baseline reference-score drift is
0.0. The selection/basis gate, not reached before the runtime stopped, passes when
independently evaluated from the saved records.

The raw control and instrumented native generation have identical masks, selected
positions and generated IDs `[3869, 2]` (Yes plus EOS). First-step generated token
log probability differs by about **−5.96e−8**. This small output-score difference
does not establish the magnitude of any feature difference.

## What failed and what is unknown

The raw control's projected-feature SHA256 is `9818c0c6...`; the instrumented
native generation's is `d9f6b97c...`. Both cover the entire **576×4096** projected
tensor, although only the 288 mask-selected states reach the language input.

The first runner compared a standalone uninstrumented encoding with another
encoding during instrumented generation. It did not establish whether separate
raw native executions themselves produce byte-identical features. It also stored
only a fingerprint of the raw output, so the current ZIP cannot locate, quantify
or restrict the difference to selected versus excluded coordinates.

FP16 merge accumulation or projection are candidate locations; no cause is yet
established. Matching answer strings alone does not establish tensor parity.
Changing a numerical tolerance now would be unsupported. Exact same-image and
repeat gates later in the pilot could also fail if native merge realizations vary.

The three observed strings are No / Yes / No for full / native merge / no-merge.
This is potentially useful, but remains **diagnostic evidence** until the failed
measurement check is understood. It is not yet an admitted merge-rescue result.

## Bounded next run

Added a separate `--diagnose-native-parity` mode to the same runner and notebook.
It uses the first frozen question only. It repeats raw native and observed native
encoding, includes observed no-merge controls and an observed generation capture,
and saves actual returned tensors plus available pre/post-merge states.

Comparisons record exact equality, changed elements/states, maximum and RMS
differences, relative L2, selected and excluded coordinates, source/E/basis/merge
identities, and fixed-reference score propagation. The generation capture checks
the context that failed in the original pilot. Raw encodes expose only their
returned tensors; their unobserved upstream stages are not retrospectively known.

The normal 30-case pilot and its exact gate remain unchanged. A successful
diagnostic means evidence collection completed, with `diagnostic_only=True` and
`scientific_pilot_passed=False`. It does not approve the original pilot or a new
tolerance. Review the stage deltas before choosing a repeatability/parity policy
or freezing one native realization per image for paired comparisons.

Use the existing live notebook/environment/input Dataset. The updated
[run guide](../RQ2_INPUT_MERGE_PILOT.md) includes a Git pull, new prerequisite checks,
updated helper load and one diagnostic invocation. A fresh notebook can instead
enable only `RUN_PARITY_DIAGNOSTIC = True`. Download its new ZIP and log.

No dependency, original native ApET file, RQ1 result, fixed cohort or reference
label was changed. The diagnostic extension has not yet been executed on pretrained
Kaggle hardware; its output is the next evidence needed.

Local validation: 23 CPU tests passed; the two Torch-dependent classes (12 tests)
remain for the original Kaggle runtime. All notebook code cells compile and have
no saved outputs, both execution flags default to False, all eight monitored
native source hashes match, and the original runtime lock is unchanged.
The optional regression against this private failure ZIP runs locally; the Kaggle
prerequisite invokes only the self-contained policy and Torch classes.
