# High-norm follow-up implementation

Prepared `kaggle_norm_intervention.ipynb` and an independent versioned runner,
without editing original ApET, the selector grid, or basis/localization code.
The CPU FPS audit motivates this follow-up but does not establish a causal
mechanism or an attention sink.

31 conditions: baseline576, three parent references, A1/A2/B1/C1/C2,
two decoder-only FP32 FPS variants, four matched control families × five seeds.
Five-question calibration is required before TextVQA100. Existing input dataset
and original runtime are reused. Historical parent predictions from the completed
basis/localization archive are pinned for drift reporting.

Deliberate difference from suggested sorted-source swapping: native basis copies
already decouple source identity from carrier position. Keep carrier positions
fixed and replace only contents, recording both identities. This isolates contents
without a new positional/order intervention. B replacement continues farthest
selection from the remaining retained set; does not restart the whole FPS run.
C2 is exploratory and not count-matched to single-insertion controls. A2's removed
basis-slot membership is recorded rather than assumed. No attention capture.

Validation here: nine local policy tests passed; two Torch-dependent test classes
skipped because the Mac reading environment has no Torch. New actual native test
covers every arm's gather/replay/cache and shared pre-prune pool on Kaggle, with
zero skips required. Notebook code cells and new Python modules compile.
No pretrained run has yet been executed. Do not claim GPU calibration passed.

After results: `scripts/summarize_norm_intervention.py RUN.zip --output summary.json`
reports matched-control differences, image-cluster bootstrap intervals and
source-padding strata. Interpret replacement-specific effects; avoid universal
necessity/sufficiency or attention-sink claims.
