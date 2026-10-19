# RQ2 crossed selection/merging implementation — 2026-10-19

Implemented the next comparison requested after the passed native/no-merge
panel: native E-ranked ordinary retained positions versus random ordinary
positions, each with and without merging, at 288 tokens. The optional Brolio
target-restoration diagnostic is implemented as a separate mode.

No new pretrained result is claimed. The next action is the seven-question
Kaggle calibration in [kaggle_rq2_crossed.ipynb](../notebooks/kaggle_rq2_crossed.ipynb).
The [run guide](../RQ2_CROSSED.md) includes new-notebook and existing-live-session
commands. The input remains `rq1-ih-inputs-v1`; runtime, checkpoints, native
sources, old v2 measurements/results and accepted dataset identities are unchanged.

## Frozen design

- Same 30 questions / 24 images, with the previous seven-question calibration.
- Full576, E-merge, E-no-merge, random-101/202/303 each merge/no-merge: nine
  primary conditions, 63 calibration rows and 270 full rows.
- The ten native FPS source vectors and original spatial basis replacement slots
  stay fixed. Uniformly sample 278 distinct ordinary positions from the other
  566 coordinates with an isolated, image/seed-specific Python RNG. Allow FPS
  source duplication at distinct ordinary coordinates, as the native path does.
- Canonically order the random subset by native E then index for routing; append
  the unchanged native basis-slot tail. Set membership is random; ordering is not
  another uncontrolled random factor. Final native mask ordering is spatial.
- Compact output positions of basis slots may change with the subset. The exact
  control is source-to-original-spatial-slot assignment, not a false claim that
  both spatial and compact positions can stay identical under arbitrary subsets.

This estimates the incremental value of E selection conditional on native basis
insertion. It is not a fully basis-free or E-free random compressor.

## Implementation boundary and checks

The scoped project-owned hook intercepts only the top-k call made directly by
the pinned native `encode_images` function. It first verifies native E and top-k
against the E reference, substitutes a predeclared random index list, and lets
original FPS/reconstruction/basis assignment/merge/projection continue. Other
top-k calls are unchanged. All hooks restore on error; original native source
hashes remain the authority for baseline ownership.

Exact source/CLIP/pre-merge/E/FPS/basis-spatial-slot checks apply across selectors.
Each selector's merge/no-merge pair additionally requires identical selected
ordering, mask, basis compact mapping and no-merge identity. The existing passed
v2 clone-replay implementation supplies generation and reference scoring; its
code is reused unchanged. First primary captures are saved before generated
answers and reused across same-image questions. GPU banks are released only
after an image's last question, without recapturing that image.

Five declared GQA/TextVQA controls repeat full/E-merge/E-no-merge/random101-merge/
random101-no-merge on fixed tuples. Independent E-merge and random101-merge
captures repeat twice on those controls. Numerical and answer changes are
recorded; structural source/selection/basis/merge-graph changes still fail.
Fresh candidates are saved before scoring/comparison so partial failures remain
reviewable. The other random seeds do not receive extra fresh repetitions.

Summaries retain unique-question denominators, all seed-wise values, seed means,
E-minus-random effects for each merge rule, merge effects under each selector,
their interaction and paired rescues/losses. Seed repeats do not count as new
questions. TextVQA full-credit correctness and consensus means remain distinct.
Existing MMStar flags and same-image dependence remain limitations.

The full30 requires the new crossed calibration's actual detailed questions,
feature/evidence hashes, complete nine-condition coverage and controls, source,
runtime and checkpoint identity. A v2 calibration or oracle run cannot satisfy
this gate. Every run preserves a timestamped ZIP and separate console log;
full ZIPs embed the prerequisite calibration evidence.

## Optional Brolio diagnostic

The separate `oracle` mode runs only `textvqa:37533`. It captures full576, native
E-merge and E-no-merge, then runs sham, restore317 and a matched-control restoration.
The control is the other ordinary target with closest incoming merge count,
breaking ties by index. Matching is only on merge count, not content or location.

A native-projector no-op must first reproduce the frozen projected tuple exactly.
Restoration replaces one target's post-merge visual feature by its pre-merge
feature, applies the native projector and splices only that projected state into
the original tuple. All other projected positions and the mask must remain byte
exact. The sham must reproduce native generation and reference scoring.

Target317 was selected from inspecting Brolio and is therefore post-hoc and
gold-informed. Recovery would identify a sufficient local feature change in
this context, not unique importance, literal letter blurring or a deployable
protection policy. The matched control may also recover the answer. Both success
and failure must be retained. Oracle output is excluded from crossed summaries
and cannot be a main calibration/full result.

## Local validation

45 CPU checks passed across old and new RQ2 modules. New checks cover frozen
conditions, random set/RNG invariants, basis spatial-versus-compact identity,
seed-wise effects without denominator inflation, same-image bank reuse, complete
hashed calibration requirements, safe mode/export separation and matched control
choice. Four Torch-dependent classes skip in the Intel Mac reading environment.

The four new Torch tests await Kaggle: actual native random pairs and RNG/source/
basis identity, top-k/method restoration on failure, actual tiny decoder replay
of a random tuple, and projector no-op/single-target oracle geometry. The notebook
imports Torch explicitly and rejects skipped prerequisites before pretrained work.
It also runs the previously passed native/v2 tests.

All notebook code compiles, all flags default to False and stop before setup,
and mocked helper routing selects the correct calibration/full/oracle CLI mode.
Eight original native file hashes and the locked environment match; no local
Torch installation or model inference was performed. No commit or push was made.
