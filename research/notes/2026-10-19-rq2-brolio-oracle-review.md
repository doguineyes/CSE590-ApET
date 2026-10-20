# RQ2 Brolio single-target restoration — 2026-10-19

The one-question oracle diagnostic passed. Restoring only ordinary target317's
pre-merge feature recovers the correct answer Brolio. The unchanged sham and
restoration of count-matched target141 remain Bottoli. This is a conditional
single-state rescue, stronger local evidence than disabling all merging, with
the limitations of a selected post-hoc target and one control.

## Preserved evidence and verification

Copied both supplied files into `research/results/`, keeping Downloads originals:

- `rq2-crossed-oracle-v1-20261020T000115185532Z.zip`: 39,588,690 bytes,
  SHA256 `fc634b1aae1277e0af70fa067aac87a2ebe76827d1a57ca229f3fa5280de7baf`.
- Matching `.log`: 1,865 bytes,
  SHA256 `cf8c1b1b838085fb51e8f8c0f44f4d13e7cfba7ad574faabdc5e80af13cba5c8`.

All 127 unique members pass streamed CRC/SHA checks: 126 artifact hashes, 112
evidence hashes, five feature-file hashes and one question hash. The separate
oracle report equals the embedded report and its recorded hash verifies. Source
copies match measurement/auxiliary digests. Current measurement hashes, the eight
pinned native source files and original lock match. Clean runtime commit:
`4d0cc311f5e8d6f731b585fd8b0a108dea18692c`. Setup evidence includes the passed
21 native prerequisites. No native source, model weights or environment changed.

Independently rescored all six predictions, checked reference-score arithmetic,
the fixed prompt/target IDs, source/selection/basis identities, generation/scoring
clone replay, and recorded single-position restoration controls. Runtime checks
assert unchanged mask and byte-identical non-target projected positions. CPU
review checks these recorded controls, their pinned implementation and artifact
bytes; tensor pickles were not loaded or model inference run on the Mac.

The sham is exactly identical to native E merge in encoded fingerprints,
generated IDs and fixed-reference token log probabilities. All three shared
primary generated-ID outputs match the preceding crossed full30 run. Oracle
mode correctly reports `diagnostic_only=true`, `scientific_pilot_passed=false`
and `calibration_passed=false`; those false fields distinguish this selected
case study from a full experiment/calibration and are not execution failures.

Derived verification, answer CSV, detailed oracle JSON and the review script are
saved in `results/rq2-crossed-oracle-v1-review-2026-10-19/`.

## Exactly what was changed

All compressed conditions keep native E selection, 288 output states, ten FPS
basis representations/slots, prompt, precision, generation policy and target.
Target317 is an ordinary retained merge target, indexed in the original 576-
position grid rather than the compact 288-state sequence. Its incoming dropped
sources are 103/109/331. The control target141 has three incoming sources too:
107/110/143. It was chosen by the frozen closest-incoming-count / lowest-index
rule, without testing its answer first.

The restoration starts from one frozen native merged tuple. It replaces the
chosen target's pre-projector state with that same target's pre-merge state,
applies the native projector, and splices only that target's projected state
into the original tuple. All other projected positions and the mask remain exact.
This changes one retained input state, not one raw image pixel or every later
decoder activation. Native basis insertion is retained.

The runtime records only projected position317 changed in the target intervention,
only141 in the control, and no position in the sham. The projector no-op check
passes in all three conditions. Other merge recipients are not recomputed or
rerouted by this diagnostic. The saved native graph describes the originating
native realization; project-owned restoration metadata marks its edited output.

## Result

| Condition | Generated answer | Correct | Mean gold-token log probability | First gold-token probability |
|---|---|---:|---:|---:|
| Full576 | Brolio | Yes | -0.139325 | 0.811604 |
| Native E merge | Bottoli | No | -1.503870 | 0.058736 |
| Same selection, no merge | Brolio | Yes | -0.265204 | 0.709970 |
| Sham: unchanged merged tuple | Bottoli | No | -1.503870 | 0.058736 |
| Restore only target317 | Brolio | Yes | -0.352819 | 0.556764 |
| Restore only control141 | Bottoli | No | -1.502551 | 0.058997 |

The fixed target is Brolio, token IDs 4358/5991. First-gold-token probability
increases about 9.48 times after restoring317; mean reference log probability
improves 1.151051. These are teacher-forced reference-token diagnostics, not a
whole-answer correctness probability. Restoration recovers the generated name
without reproducing the full576 score distribution. The control causes a much
smaller reference-score change and no answer recovery.

## Interpretation for RQ2

The earlier no-merge experiment changed all ordinary merge targets and therefore
did not localize the Brolio rescue. Here replacing one specified retained state's
merged value is sufficient to recover the correct answer while all other input
states and the selected positions stay fixed. The sham excludes a replay-only
rescue, and the count-matched control shows the same outcome did not follow from
restoring that one other target. This supports a local merge-dependent failure
under this fixed context.

It does not prove317 is uniquely necessary, identify the exact letters represented
there, or show that every similar target should be protected. CLIP states are
contextual features. Incoming count matching does not match location, E score,
semantics or incoming-source content. One post-hoc, gold-informed target and one
control do not establish a general protection policy or population-level gain.
No independent fresh repetitions of the restoration conditions were run here.

The next ordinary routing/protection experiment remains useful: freeze E-based
selection and test similarity routing and prespecified high-E/random protection
across the panel. Its targets are chosen by a policy rather than the correct
answer. Do not tune the protected count or targets to force this oracle recovery.
Keep the oracle separate from its aggregate scores and calibration gate. No
measurement fix or repeat run is needed to proceed with those existing controls.

Observed wall time 173.63 seconds includes loading and evidence work, not a speed
benchmark. The raw archive/log and all negative control outcomes are preserved.
