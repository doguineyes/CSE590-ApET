# First RQ1 measurement implementation — 2026-10-15

Dataset construction is paused. We implemented a GQA-first input-stage sensitivity
pilot on the existing native original-ApET runtime, plus optional TextVQA baseline
generation. [Instructions and notebook](../RQ1_INPUT_PILOT.md).

The registered calibration uses cake/plate image `n200692`: questions `201636988`
and `201637316`, 32 uniformly sampled patches each. The larger pilot fixes 12
questions on six images (four loss/control pairs and two preserved pairs), with
576 singleton interventions per question. GQA's accepted 49 questions, MMStar's
18 primary cases and all 100 TextVQA questions are copied unchanged into a
32.6 MiB portable Kaggle input ZIP. The qualitative MMStar appendix stays in its
existing artifact; it is not merged into the primary pilot.

E is native ApET reconstruction error in pre-projector CLIP space. The baseline
branch skips E, so our scoped replay feeds its exact captured CLIP values into
native input-only ApET; its FPS/reconstruction/merging code is unchanged. Paired
questions share cached features, native errors, seed and sampled token indices.

I is the signed change in mean teacher-forced canonical-reference log probability
from setting one post-projector visual embedding to zero. Sequence length, order,
positions and mask remain fixed. This measures a singleton zeroing proxy, not
deletion, merge effects, token interactions or answer accuracy. The target is
fixed by the manifest, never added to the free-generation prompt. We check target
tokenization boundaries, causal scoring indices and native multimodal mapping.

Each question has repeated unperturbed references, clone-only no-op, cached/native
forward parity and a final reference. The pilot threshold is 0.001 nats per
target token; failures are saved and stop expansion. This is a lightweight
measurement check, not a resumption of exhaustive numerical debugging.

Outputs preserve all 576 native E values and measured I values (unmeasured = null),
raw fresh baseline predictions, reference/perturbed log probabilities, exact
targets/prompts, seeds, source/environment/checkpoint hashes, timing, images and
processor metadata. Analysis creates per-question Spearman values, same-image
question comparisons, 24×24 maps aligned to the actual 336×336 CLIP canvas and
outer-quartile exemplars. Outcome-enriched selection and image dependence limit
interpretation; there are no independent-token population significance claims.

TextVQA keeps all ten annotator answers and all 100 cases. The optional generation
run defaults to baseline + input-only 288, uses native TextVQA consensus scoring,
and saves full/partial credit and paired preservation denominators separately.
No OCR transcripts or TextVQA interventions are used. A three-question MMStar
measurement check is available after GQA validation; all three datasets remain
in RQ1. Decoder-layer measurement is still future work.

Eight local CPU tests passed, covering protocol, scoring, input hashes, sampling, partial-table
export, syntax and notebook structure. Native Torch checks are supplied as a
mandatory notebook prerequisite and include real native preparation/forward on
a tiny random model, exact replay checks and cross-device scoring when available.
The Mac has no Torch or plotting stack; those tests and the first real inference/
plot rendering remain to be validated on Kaggle. Native source hashes and locks
remain unchanged. No new model results or RQ1 scientific finding are claimed.
