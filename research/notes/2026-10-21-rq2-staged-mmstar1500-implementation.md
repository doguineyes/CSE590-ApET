# Prepared staged MMStar1500 — 2026-10-21

The next experiment after GQA is implemented. It reuses the calibrated native
staged adapter and the same full576, native96→32/native288→96 and three
pure-random seeds at each schedule. No compression/math/runtime change was made.
This completes preparation of the planned three-dataset broad comparison;
TextVQA has passed, while GQA and this MMStar run still need results review.

## Cohort and records

All 1,500 pinned MMStar questions remain, including baseline-wrong questions.
The reference is derived from the preserved October 13 original full MMStar
archive, retaining question/label/category, raw/decoded image identities,
generation question, sample seed and historical baseline answers/scores.
No accuracy threshold controls admission. The earlier native baseline has
521/1,500 correct answers with the project explicit-choice-v2 scorer. This differs
from the earlier HF LLaVA baseline and is not an official leaderboard result.
Any new baseline drift is recorded without filtering or selective retries.

The new reference SHA-256 is
`917486e1f1a70c39a352d64c35b82a650dd81e916d87dd589ff406aaf3107e8d`.
The raw source ZIP SHA is recorded inside the reference. No model weights,
datasets or raw inference-result ZIPs were added to Git.

The dataset contains 1,430 distinct image-byte hashes. Six predeclared parts each
contain 250 questions, with category/count balancing that uses no answers and
keeps identical image hashes together. Native input capture retains original
`590 + index` seeds for each question, including repeated images; random masks
remain image/seed/stage based. Encoding filenames use question IDs, so repeated
image evidence is not overwritten. Paired bootstrap intervals resample images
and use question-weighted cluster sums/counts; they do not average unequal-sized
image means or count three seeds as three times as many questions.

Full execution is 13,500 primary generations plus six identity controls,
18 fixed replays, four forced cache probes and four fresh native repeats.
Full tensors are limited to two diagnostic questions (18 files); other encodings,
E/FPS/merge, memberships, scores, cache and source fingerprints are JSON.
Exports include raw predictions, source/runtime/setup/proof snapshots, images,
CSV, both category levels, strict/primary parsing metrics, APR, paired comparisons
and a local gallery. Partial evidence survives interruption. The combiner rejects
overlap or incompatible measurement code/runtime/checkpoints and validates
byte inventories, scores, controls and cache traces without loading tensors.

## Validation and use

Fourteen new CPU tests passed, including a simulated full 1,500-question run
(13,500 primary rows, 32 controls, 1,500 separate encoding records, 18 tensor-file
fixtures), interrupted export, stale/skip proof rejection, actual ZIP-reader
fixtures, score/cache corruption, unequal image clusters and notebook idle flags.
The eleven existing staged CPU protocol/runner checks also passed.
These are orchestration/contract checks, not pretrained accuracy results.
No Torch installation, local model inference or GPU job was performed.

The notebook requires the current 40 native/MMStar prerequisites to pass with
zero skips in Kaggle before loading pretrained weights. It downloads the pinned
MMStar parquet automatically and validates all question/image identities before
inference; no Dataset or RQ1 ZIP upload is needed. Earlier source/lock pins and
all TextVQA/GQA executable files remain unchanged.

After GQA review, push these new files and import
`research/notebooks/kaggle_rq2_staged_mmstar1500.ipynb`. Enable Internet/T4×2,
set `RUN_STAGED_MMSTAR = True`, then run the cells in order. Keep all six parts
for a full run or start with `[0]` and later `[1, 2, 3, 4, 5]` to plan runtime.
Download every ZIP/log pair. Full pretrained results and broader runtime remain
pending. The optional merge factorial/improvement experiments are not implemented
by this change.

See [the execution guide](../RQ2_STAGED_MMSTAR1500.md).
