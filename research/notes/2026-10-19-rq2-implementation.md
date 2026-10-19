# RQ2 first pilot implementation — 2026-10-19

Implemented the first step of the [RQ2 plan](2026-10-19-rq2-plan.md). No GPU
experiment has run yet. The next action is the user's Kaggle execution of
[kaggle_rq2_input_merge_pilot.ipynb](../notebooks/kaggle_rq2_input_merge_pilot.ipynb).
See the [run guide](../RQ2_INPUT_MERGE_PILOT.md).

## Fixed scope

- Same 30 questions / 24 images as RQ1: GQA12, MMStar12 and TextVQA6. Original
  accepted questions, labels, seeds, historical answers and MMStar flags remain.
- Three arms: original baseline576; original native input288; same native
  selection+basis input288 with merging disabled. Basis count10, no decoder pruning.
- Original environment, checkpoints, native prompts and greedy 16-token generation.
- Primary outcome is generated-answer correctness/preservation and paired
  rescues/new losses. TextVQA partial consensus is reported separately from full
  credit. Fixed-reference whole-answer log scores are secondary diagnostics.

Random selection, random routing, protection and targeted Brolio restoration are
not implemented in this version. The first experiment estimates merging's effect
conditional on native selection/basis; it does not isolate E selection's benefit.

## Implementation boundary

`rq2_native.native_input_variant` calls the original bound `encode_images` method
and delegates the native merge in the reference arm. In the no-merge arm, the
instance-scoped merge hook returns its incoming tensor unchanged. Native FPS,
reconstruction, top-E selection, basis replacement and projection still execute.
All instance changes and return tracing restore on errors. The eight monitored
native source hashes and original lock remain unchanged.

The helper records raw source hashes, E, FPS source IDs, selected order, actual
spatially ordered basis assignment, merge graph/counts and output feature/mask
hashes. The runner gates interpretation on exact unchanged-native feature/mask
and generated-ID parity, exact native/no-merge source/selection/basis identity,
and same-image realized compression agreement across questions. It records old
versus fresh native strings without silently replacing historical records.

All baselines must remain correct and match the frozen target prefix. Full
baseline teacher-forced scores must match their archived references exactly.
Four predefined TextVQA questions repeat all arms as separate controls. Failed
gates retain partial records and a failing report instead of filtering samples
or loosening tolerances.

Native full and variant tuples are replayed only for fixed-reference scoring
through the original multimodal preparation, with explicit target-label and
position checks. Generation itself uses the actual native encoding path. No
compression math is copied into a new implementation.

## Outputs

Each timestamped directory contains the report, raw passed predictions JSONL,
per-question arm/control JSON, answer CSV/HTML, original images, actual model
canvases, source/protocol/reference/runtime evidence and a hash inventory.
Detailed pre/post-merge and projected tensor snapshots are bounded to the six
TextVQA images; all other images retain hashes and selection/merge metadata.
No weights or environments enter the ZIP.

The runner writes its ZIP on handled failure too. The notebook streams a separate
log and provides a labeled fallback archive only if the process exits without
its normal ZIP. Download both the result ZIP and matching log before stopping
Kaggle. Raw RQ1 artifacts remain unchanged.

## Validation and limits

**18 CPU tests passed**: frozen real bundle/protocol and existing scorers, gate
rejection, correct TextVQA partial/full-credit metrics, complete90-row mock export,
byte-preserving source/setup evidence, output-overwrite protection, early-failure
ZIP/hash export, and actual panel bookkeeping/same-image rejection with explicitly
synthetic model fingerprints. These are not pretrained result claims.

The native Torch test class is skipped locally because the Intel Mac editor
environment has no Torch. Its **eight tests** are staged for Kaggle: actual native
merge/no-merge parity and basis handling, scoped restoration and rejection, full
input observation, and a genuine tiny LLaVA decoder fixed-reference comparison
for all three arms. The notebook explicitly imports Torch and rejects zero tests
or skipped native prerequisites. Pretrained parity remains a gate in the pilot.

All five notebook code cells compile, have no saved execution output and retain
the False default guard. The native file hashes match the original reference;
`git diff --check` passes. No local ML runtime was installed. No code was committed
or pushed automatically, and no claim about RQ2 accuracy or rescue outcomes is
made before Kaggle results return.
