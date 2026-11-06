# Basis/localization implementation — 2026-11-06

Implemented separate `basis-localization-v1` protocol, runner, scoped adapter, original-runtime tests, reference outputs and Kaggle notebook. Existing selector-grid/GQA code, native model and environment files were not modified.

Four references (full, native, E→E drop, detail→detail drop) plus native-drop, FPS→FPS, E→FPS, E→detail, detail→E and E→random with five seeds:14 conditions,70 calibration predictions,1400 full predictions.

Native-drop uses the captured original input premerge features with basis replacement already applied; decoder executes original layer_prune with identity merge. Both native stage-specific assignment orders are preserved, including their different indexing conventions. FPS gathers96/32 directly using original FPS implementation and seeded RNG.

Common-input controls enforce exact input and pre-prune layer16 state equality for crossed E and detail arms. Fresh E/detail references are tested against the previous grid operator in the actual native fixture. Every research arm gets first-case replay and forced-four-token cache probes. Layer16 tensors and norms are saved, content-hash deduplicated; current masked states can later be compared with scoring-basis residuals/norms offline.

Validation locally:22 unittest tests ran successfully; two Torch-dependent classes skipped because Intel editor environment lacks Torch. Python and all notebook-cell syntax checks and isolated CLI help passed. Kaggle zero-skip prerequisites and pretrained calibration remain required. No GPU execution claimed.

User workflow: finish GQA without updating its checkout during execution; push new files; use kaggle_basis_localization.ipynb with existing rq1-ih-inputs-v1. Set RUN_LOCALIZATION_CALIBRATION=True. Review ZIP/log before RUN_FULL_TEXTVQA=True. See research/BASIS_LOCALIZATION.md.

Scientific scope: basis coverage enrichment motivates these tests. It does not yet prove that E adds nothing at stage2, that basis explains all ApET gains, or why detail behaves differently on GQA. Native-drop is a total pipeline merge ablation; crossed-stage comparisons localize conditional stage2 effects. No new arms chosen by calibration accuracy.

## Pre-run review amendment

Added E→native and native→E no-merge arms before first calibration. Current grid is16 conditions,80 calibration/1600 full predictions. Added composed direct-source lineage, exact native-input shared-state control and observational FPS finite/inf/nan/tie diagnostics (including native reference). Original GQA/grid files unchanged. Native Torch tests exercise all added arms plus deliberate fp16 overflow/RNG immutability. Local verification cannot execute these Torch checks.
