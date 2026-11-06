# CPU FPS replay and norm/source audit — 2026-11-06

Used saved layer16 states from basis-localization-textvqa-full-v1-20261106T205534587657Z and CLIP patches from the earlier20-seed source archive. Restricted tensor reader; no Torch/GPU inference. Question records and state bytes checked against saved hashes; recomputed norms match recorded norms. CPU script: scripts/analyze_localization_fps_cpu.py. Results: research/results/localization-fps-cpu-v1/summary.json and per-arm-question.csv. Synthetic known-overflow ordering and input-immutability checks passed.

## Precision comparison

NumPy native-style half arithmetic (half subtraction/square, fp32 reduction rounded back to half, fp32 running distance buffer) reproduced the recorded CUDA centroid sequence exactly for every logged trajectory:800/800 across eight arms. FP32 replay uses the exact same first centroid and the same saved quantized states. This is a CPU diagnostic, not a GPU precision or answer rerun.

| Decoder trajectory | Overflow cases | Same FP32 set | Same FP32 order | Mean overlap |
|---|---:|---:|---:|---:|
| Native basis10 |100|95|89|9.92/10|
| Native-drop basis10 |100|99|96|9.99/10|
| Native-input→E basis10 |100|99|96|9.99/10|
| FPS→FPS selection32 |100|97|38|31.93/32|
| E→FPS selection32 |0|97|87|31.93/32|

E→E and E→native bases:98/100 same sets,96 same order; detail→E99 same sets,98 same order (2 overflow cases). Thus overflow is real but rarely changes membership in this dataset. Order can still matter for native basis-to-slot assignment even when sets match; direct FPS gathers sort positions so only membership matters there. Finite rounding can change selections even without overflow. No precision-related answer effect established.

## Norm and source lineage

| Arm/pool | Median largest norm | Median max/median norm | Largest state retained |
|---|---:|---:|---:|
| Native |694.4|24.6|100/100|
| Native-drop |694.4|23.9|100/100|
| Native-input→E |same exact pool as native-drop|same|0/100|
| FPS→FPS |693.2|24.4|100/100|
| E→E |36.2|1.27|14/100|
| Detail→detail |53.4|1.87|20/100|

For native/native-drop/native-input→E, the top layer16 state comes from a stage1 inserted FPS basis source in100/100 cases; that source is CLIP norm rank1 in100/100. FPS→FPS also has CLIP norm rank1 source in100/100 (but has no special10-token input basis).

Native-drop has1 state above10×median in26 cases,2 in66,3 in8. Native has1 in21,2 in63,3 in15,4 in1. Therefore "a single massive token" is not generally correct. E pools have no such states; detail pools have them in2/100 images. Threshold10 is descriptive, not a validated definition of a sink.

The maximal source patch is fully padded in31/100 native/native-drop/FPS cases. This is positional provenance only: CLIP and decoder attention mix information across positions. A padding-location feature is not automatically useless. Pixel-source locations and CLIP norm ranks are preserved in the per-case CSV.

## Interpretation and review corrections

Evidence supports a high-norm source preserved through the basis path and systematically discarded by the E-only decoder on the same native-drop pool. It does NOT establish attention-sink behavior (no attention measurement here), that this state is necessary, or that it alone explains the score gain. Other basis/ordinary tokens also differ between native-drop and E-drop outputs.

Do not state "merging contributes nothing", "basis is necessary at both stages", "E is poor in general", or "padding explains all GQA gains" from these bounded experiments. Native merge−drop interval includes meaningful gains/losses; a cohort-specific intervention is not universal necessity. FPS and native point estimates are not equivalence evidence.

## Recommended bounded GPU follow-up (plan, not implemented here)

Prioritize answer-effect tests over a broad numerical sweep. Use TextVQA100, existing runtime, five-question calibration before full execution. Preserve the current native-dtype references.

1. Exact frozen-output/no-op replay controls for native-drop and FPS→FPS.
2. Remove the top-norm retained source at layer16 while keeping32 slots, replacing it with a prespecified next eligible candidate. Save the exact replacement, source lineage and positional policy. Match controls by removing another retained token and using the same replacement policy. Use several fixed random-control seeds; do not choose them by answer outcome.
3. Because multiple high-norm states remain, failure of a single-token removal does not rule out their collective contribution. Consider a bounded top2-removal variant with equal-count controls if the first calibration verifies selection/count contracts.
4. Small decoder-FP32-FPS checks for native-drop and FPS→FPS can accompany this test; keep stage1 and pre-prune states frozen. This isolates numerical selection changes from an upstream precision change.

Avoid the proposed force-in of a token identified from another model trajectory for this first follow-up: it changes stage1, subsequent states and stage2 selection together. That can be a later restoration experiment with explicit matched controls. Do not spend GPU time solely to exhaust quota. Target <=1hour, including calibration, but estimate from actual controls/serialization timing. No new GPU runner was implemented in this CPU-check task.
