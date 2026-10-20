# RQ2 routing/protection implementation — 2026-10-19

Implemented step 4 of the RQ2 plan at the user's request. The passed native/no-merge
results remain unchanged. E/random ordinary-token selection and Brolio restoration
are separate experiments; neither is silently used as a gate for this runner.

## Frozen design

Reuse the accepted RQ1 IH bundle and same 30 questions / 24 images, including
GQA same-image pairs and prior MMStar limitations. Input budget is 288, with ten
native FPS basis sources and exactly the original basis spatial assignment. E
selection and decoder-disabled settings remain fixed across compressed arms.

Ten conditions: full576, native nearest, no merge, destination shuffles101/202/303,
protect-high-E28 and protect-random28 with seeds101/202/303. Shuffling permutes the
native destination multiset across dropped sources, preserving recipient counts.
Protection routes every source to its nearest unprotected ordinary target. Random
protection matches the protected count; incoming mass/location/semantics need not
match. The protection result therefore includes rerouting/load redistribution.
No E/I/gold target is selected from newly observed outcomes and no protected-count
tuning is implemented. Randomness is isolated from native FPS and generation.

## Implementation ownership and controls

Added `rq2_routing_protocol.py`, `rq2_routing_native.py`, `run_rq2_routing.py`, a
versioned JSON profile, CPU/native tests, notebook and run guide. The native hook
is restricted to the cosine call directly inside pinned native input merging.
Original ApET FPS/E/selection/basis/FP16 merging/projection still execute; native
files, environment lock, existing RQ1/RQ2 runners and archived results are unchanged.

Every variant checks exact source/pre-merge/selection/basis/mask identities.
Shuffles check exact planned destinations/count preservation. Protection checks
zero incoming sources and unchanged protected features; all non-target features
must remain exact. Tiny native tests independently check nearest eligible
destinations, FP16 averaging, unrelated-call delegation, exception restoration,
RNG state and actual decoder/scorer clone replay.

Capture the first unconditional realization per image/condition and share it across
same-image questions. Fixed-tuple repeats cover six conditions on five declared
controls; two fresh captures cover native, shuffled101, protect-high-E and random-
protected101 on those controls. Numerical drift remains uncertainty evidence;
source/selection/graph drift fails. No speed claim is made with this instrumentation.

Seven calibration questions give 70 primary predictions; full30 gives 300.
Full requires this runner's detailed hashed calibration evidence, complete conditions,
recomputed routing plans/graphs, repeat controls and identical source/runtime/
checkpoint identity. Older v2/crossed calibration reports are rejected.

Outputs retain graphs/counts/protected sets, per-seed and paired outcomes,
TextVQA consensus/full-credit summaries, feature banks, raw answers, reference
scores, source/runtime identities, source copies and setup evidence. Export
ZIP/log, including partial handled failures. Seeds do not multiply question counts.

## Next run

Commit/push, import `research/notebooks/kaggle_rq2_routing.ipynb`, attach the same
private rq1-ih-inputs-v1 Dataset and enable only `RUN_ROUTING_CALIBRATION`.
Use T4 ×2 and Internet. The [run guide](../RQ2_ROUTING.md) also gives an existing-
session cell that avoids reinstalling the environment. Native and pretrained
execution remain pending; successful local CPU checks are not GPU evidence.

## Local validation

`environments/editor/.venv/bin/python -m unittest discover -s research/tests -p
'test_rq2*.py' -v` passed 57 CPU tests. Five Torch-dependent classes were skipped
because the Intel Mac editor environment has no Torch. The five new routing
native tests run with the previous native/frozen prerequisites in Kaggle (21
checks, zero allowed skips), followed by the pretrained calibration.

All six notebook code cells compile, contain no outputs, and the default flags
stop before any setup command. Helper lookup and all archived source paths verify.
All eight pinned native source hashes and the original environment lock match;
Git whitespace checks pass. These checks do not establish a scientific outcome.
