# RQ2: fixed selection, merge routing and target protection

The [full30 run passed](notes/2026-10-19-rq2-routing-full-review.md): native 19/30,
no-merge 21/30, shuffled 19–20/30, high-E protection 19/30 and random protection 19–20/30.
All 30 fixed replays and 40 fresh answer repeats are stable. High-E protection
redirects only 21 of 6,912 sources and changes no assignments on 15/24 images.
The first bounded comparison is complete; its efficacy limits and next steps are
recorded in the review.

The [seven-question calibration passed](notes/2026-10-19-rq2-routing-calibration-review.md).
All 30 fixed replays are exact and 40 fresh repeats preserve answers. Policy effects
are mixed; high-E protection changes only 7 source assignments across 6 images,
with no change on 4 images. That calibration gated the completed full30 above.

This is step 4 of the [RQ2 plan](notes/2026-10-19-rq2-plan.md). The
[E/random-selection comparison](RQ2_CROSSED.md) and optional Brolio restoration
remain separate experiments. This runner has its own passed calibration and full30
pretrained results.

## Frozen comparison

Reuse the same 30 questions / 24 images, original runtime, checkpoint, preprocessing,
prompts, scoring and reference targets. All compressed conditions use **native E
selection, 288 tokens, ten FPS basis representations with identical spatial
placement**, and no decoder compression. The 278 ordinary targets stay fixed.

| Condition | Operation |
|---|---|
| Full576 | Uncompressed reference |
| Native nearest | Original ApET routing and averaging |
| No merge | Same selection/basis; ordinary vectors unchanged |
| Shuffled routing, seeds 101/202/303 | Shuffle the native destination list across spatially ordered dropped sources; preserve every target's incoming count |
| High-E protected | Protect the 28 highest-E ordinary targets; reroute sources to their nearest remaining eligible target |
| Random protected, seeds 101/202/303 | Protect 28 randomly chosen ordinary targets; otherwise the same nearest-eligible rule |

There are **ten conditions**, 70 primary predictions in calibration and 300 in
full30. The 28-target count is frozen before inference and will not be tuned on
rescued examples. E ties use ascending original coordinate; nearest-neighbor
ties retain native target ordering. Basis slots remain ineligible merge targets.
Random policies have isolated image/seed-derived RNG streams and do not advance
FPS or generation RNG. Every dropped source still goes to exactly one target.

Shuffling breaks the native similarity assignment while preserving averaging
loads. It may leave some assignments unchanged; their number is recorded.
Protection preserves its chosen vectors exactly, but also redistributes incoming
sources and changes the loads of other recipients. Its outcome is the combined
effect of protection and rerouting. Random protection matches the number of
protected targets, not their incoming counts, locations or semantics. Record
native protected incoming mass and before/after counts instead of attributing
every effect solely to protected E scores.

## Original implementation versus project changes

Native files and the original dependency lock are unchanged. A project-owned,
scoped hook changes only the cosine-result tensor read by the single pinned
native input `token_merging` call. Shuffling encodes the predeclared destination
permutation; protection masks protected target columns before native argmax.
Native source selection, basis insertion, FP16 scatter/normalization and projector
still execute. The hook restores on success and failure. Actual cosine identities,
destinations, protected sets, counts and ownership are archived explicitly.

Source/CLIP, pre-merge features, E, FPS, selection order, mask and full basis
assignment must match the native reference exactly. Shuffling must preserve all
recipient counts; protection must give its targets zero incoming sources and
byte-identical pre/post-merge vectors. Basis and dropped coordinates remain
unchanged. Each first unconditional image/condition realization is captured before
answers and replayed by generation and reference scoring. Same-image questions
share that realization. Independent fresh captures record numerical/answer
variation rather than demanding bitwise equality across native CUDA reductions.

On five declared GQA/TextVQA controls, fixed-tuple repeats cover full576, native,
no-merge, shuffled101, high-E protection and random-protected101. Two fresh captures
also cover native, shuffled101, high-E protection and random-protected101. The
other random seeds do not receive extra fresh repeats.

## First Kaggle run

1. Commit and push the new files to `apet-analysis`.
2. Import [kaggle_rq2_routing.ipynb](notebooks/kaggle_rq2_routing.ipynb) as an editable
   draft. Enable Internet and T4 ×2.
3. Attach the same private **rq1-ih-inputs-v1** Dataset; no new input ZIP is needed.
4. Set:

   ```python
   RUN_ROUTING_CALIBRATION = True
   RUN_ROUTING_FULL = False
   ```

5. Run the cells, then download `rq2-routing-calibration-v1-<timestamp>.zip` and
   its `.log`. Review the controls before starting full30.

After a passed calibration, in the same live session:

```python
routing_full_dir = run_rq2_routing(
    mode="full",
    calibration_report=routing_calibration_dir / "report.json",
)
```

For a new session, extract the complete accepted routing calibration ZIP,
including its feature/evidence subdirectories. Set `CALIBRATION_REPORT` to its
`report.json`, enable only `RUN_ROUTING_FULL`, then execute setup. The full gate
checks actual evidence hashes, conditions, repeated controls, measurement sources,
runtime and checkpoint identity. Older v2/crossed reports cannot gate this runner.

## Reuse a live original-ApET notebook

When `repo`, `inputs`, `original_python`, `setup_log`, `stream_command` and
`CHECKPOINT_ARGS` still exist, add this cell after pushing. It refreshes provenance
and runs prerequisites without reinstalling the environment:

```python
import json, re, shutil, subprocess, zipfile
from datetime import datetime, timezone
from pathlib import Path

subprocess.run(["git", "pull", "--ff-only"], cwd=repo, check=True, timeout=120)
updated = json.loads((repo / "research/notebooks/kaggle_rq2_routing.ipynb").read_text())
helpers = ["".join(c["source"]) for c in updated["cells"]
           if c["cell_type"] == "code"
           and "".join(c["source"]).startswith("def verify_rq2_routing_prerequisites(")]
if len(helpers) != 1:
    raise RuntimeError("Expected exactly one routing helper cell.")
exec(compile(helpers[0], "rq2-routing-helpers", "exec"), globals())
stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
previous_setup_log = setup_log
setup_log = Path(f"/kaggle/working/rq2-routing-setup-{stamp}.log")
shutil.copy2(previous_setup_log, setup_log)
runtime_report = Path(f"/kaggle/working/rq2-routing-runtime-{stamp}.json")
verify_rq2_routing_prerequisites(refresh_runtime=True)
routing_calibration_dir = run_rq2_routing(mode="calibration")
```

New notebook sessions do not share virtual environments/checkpoint caches. The
notebook kernel's packages are not modified. Existing live sessions reuse their
environment and model downloads; each runner loads the model once.

## Interpretation and records

Report native-minus-shuffled effects, protected-minus-native effects,
high-E-minus-random-protection effects and paired rescues/new losses. TextVQA
consensus score and full-credit accuracy remain separate. Random seeds repeat
the same questions and must not inflate the question denominator. Raw answers,
target scores, graphs, feature evidence, source/runtime hashes and first/fresh
realizations are preserved in ZIP/log outputs, including on handled failure.

This is a selected mechanism panel, not population accuracy, a timing benchmark,
or a demonstration that protection helps universally. Gold-informed target317
restoration is not used in this policy. Broader pool evaluation and a second
budget remain later decisions after these bounded controls pass.
