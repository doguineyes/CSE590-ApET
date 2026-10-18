# RQ1 input-stage IH extension — 2026-10-18

Decision: add source-informed 1×1 retain-one sensitivity wherever the canonical
input panel already measured native ApET E and zero-one sensitivity. Preserve the
30-question cohort (GQA12/MMStar12/TextVQA6) and all historical scores. The
completed six-case layer-16 full grid already includes both interventions and
does not need rerunning. Smaller calibrations/checks are nested within these
full grids, not additional cohorts. TextVQA's 100-question generation screening
is not expanded into a new intervention cohort.

New code is confined to project-owned protocol, native instrumentation, runner,
bundle preparation and tests. Existing native ApET and zero/layer/IH measurement
files and environment locks are untouched. The input notebook is updated in
place; the layer notebook only gains a completion/link note.

The new runner physically retains a single projected visual embedding before
block 1, preserving nonvisual embeddings and original gapped position IDs. Native
SDPA builds the shortened causal mask; temporary RoPE hooks preserve old lookup
geometry. It pairs first-target-token probability changes with the corresponding
archived zero-first probabilities, retains zero-whole log scores separately,
and logs native E in its original pre-projector space.

Reference reuse is strict: exact features/E/basis/seed/prompt/target/mapping,
fresh correct generation, exact full-reference/native/no-op scores, three exact
zeroing audit repeats per case, and retention no-op/null/final checks. Failures
preserve raw records and stop. Numerical tolerance for within-run controls is
unchanged; cross-run reference reuse must be exact. Calibration details and
source/model/profile hashes gate the full sweep, and its sampled IH values must
repeat exactly.

Built `research/results/rq1-ih-inputs-v1.zip`, 35,728,669 bytes (34.1 MiB), from
fully verified canonical ZIPs plus the unchanged accepted image bundle. SHA256:
`27c30658b6ec201a67e175dd1f42f1d59d981856254ddc73eb13e9009480f419`.
Reference index SHA256:
`dd91cca156a090fc6e2c236f16042358e2dbe3b9db5d49d12609a45618d49644`.
Raw results, old bundles and native code are preserved; this bundle is ignored by
Git and must be attached to Kaggle separately. A preliminary local build with
absolute audit paths is kept in `rq1-ih-input-build-draft/`; use only the final
top-level ZIP named above.

Start with six questions × 32 positions: the original first two recorded
questions per dataset, not cases chosen by new IH outcomes. This costs 192 IH
measurements plus 18 zeroing audit repeats and controls. After reviewing it,
the full sweep is 30×576 = 17,280 retention scores plus 90 zero audits, with one
model load per command. Estimate runtime from this calibration; layer-16 timing
does not transfer directly. The historical reference scores are joined, not
remeasured wholesale. No full run is enabled by default.

See [Kaggle instructions](../RQ1_INPUT_IH.md) for exact flags and cells. The
current [input notebook](../notebooks/kaggle_rq1_input_pilot.ipynb) contains the
new calibration/full helper before the disabled historical sections. No local
ML environment changes are required.

Keep the interpretation operational: the input null has no prior decoder image
processing but retains position gaps; CLIP features are contextualized. Layer-16
text states have already seen the image. Zeroing and deletion alter different
contexts/operators, and first target pieces are not whole-answer accuracy.
Preserve MMStar quality/grounding flags and report them separately; no hidden
filtering or pooled-token significance. This strengthens the bounded RQ1
comparison before moving to RQ2, without promising an intervention-independent
token-importance measure or complete IH benchmark reproduction.

This is an exploratory follow-up chosen after inspecting the previous results,
not a preregistered study. The cohort is unchanged and the new definition is
fixed before running the added measurements.

Validation on the existing editor Python: **29 CPU checks passed** (9 new input
IH checks, 13 existing input checks and 7 existing layer-IH checks). Both edited
notebooks have valid unique cell IDs, parseable code and disabled execution
defaults; the new helper's isolated UTF-8 child launch is exercised. Input ZIP
CRC, 207 unique members, reference-index pin and final SHA256 verify. The new
three tiny native tests and pretrained calibration remain for Kaggle; Torch was
not installed locally. No original native/shared measurement or lockfile changed.
