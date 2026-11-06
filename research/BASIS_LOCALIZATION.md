# TextVQA basis and stage localization v1

Purpose: test whether retaining basis sources and changing the second selector improves answers. The CPU evidence motivates these tests; it does not establish that stage2 E contributes nothing or that FPS explains ApET's full benefit. GQA's smooth-region explanation is also a hypothesis pending image review.

## Conditions (16)

All compressed arms use96 input tokens and32 after decoder block16. Same fixed TextVQA100, prompts, original locked runtime, checkpoint and deterministic generation settings.

| Input → decoder | Basis replacement | Merge | Conditions |
|---|---|---|---:|
| Full576 | none | none | 1 |
| Original native96→32 | original at both stages | original | 1 |
| E→E | no | no | 1 |
| Detail→detail | no | no | 1 |
| Native-drop | original at both stages | disabled | 1 |
| E→native-drop decoder | decoder only | disabled | 1 |
| Native-drop input→E | input only | disabled | 1 |
| FPS96→FPS32 | direct gather, no insertion | no | 1 |
| E→FPS32 | no | no | 1 |
| E→detail | no | no | 1 |
| Detail→E | no | no | 1 |
| E→random32 | no | no | 5 seeds101–505 |

Native-drop projects the captured native **pre-merge** input features, which already contain exact original basis replacement, retaining the same native input mask/RNG continuation. Decoder calls the original `layer_prune`, replacing only `token_merging` with identity; it preserves original topk ordering and basis assignment. A native CPU/Torch fixture compares input features with a fresh original encoder with merge disabled. Runtime verifies decoder replacement/gather against the captured states. This avoids treating sorted union(FPS,topE) as equivalent to the original slot-replacement implementation.

FPS uses the original `llava.model.utils.fps` with incoming feature dtype and the image's existing seed. It directly gathers96/32 diverse states; it is not a new basis insertion mechanism. Duplicates fail the unique-count check. FPS-only uses one fixed image seed, not a multi-seed claim of robust superiority.

E stages use the established no-basis selector implementation and ridge residual. E-input arms share exactly the same input features/mask and pre-prune layer16 states; run aborts if equality fails. Detail→E shares the detail-reference input and layer16 states. These controls isolate a second-stage operator change conditional on its fixed input state, not an additive decomposition of native ApET.

## Saved evidence

Each compressed arm saves pre-prune layer16 visual states `[1,96,4096]`, per-token FP32 L2 norms, fingerprints, condition-specific input/source mappings, final indices, errors and basis where relevant. Identical tensors share a content-hashed `.pt` file, including across repeated E→random seeds. Native reference also saves its layer16 source states. State files contain tensors only, CPU fp16; use trusted local Torch or the repository's restricted CPU tensor reader. Budget roughly1GiB extra output capacity; deduplication should reduce actual usage.

Stage1 selected slots and direct source positions are recorded separately when native basis replacement occurs. Decoder native-drop records compact slot→basis-source pairs. Contextual states still contain information from other patches. Positional lineage is not semantic information flow.

Fresh baseline/native/E/detail outputs are compared to the archived selector-grid reference; drift is recorded, not repaired by answer-based retries. Raw predictions, states, runtime/lock/source evidence, logs and failure ZIPs are retained.

## Kaggle

1. Wait until running GQA finishes before pulling updated code in that session. Push these new files to GitHub.
2. Import `research/notebooks/kaggle_basis_localization.ipynb`, attach the same `rq1-ih-inputs-v1`, enable Internet and T4×2. No new dataset ZIP or answer boxes needed.
3. Set `RUN_LOCALIZATION_CALIBRATION = True`. Run setup and **zero-skip** original Torch prerequisites, then calibration: first5 fixed questions ×16 conditions =80 predictions.
4. Download calibration ZIP/log for review. Existing selector-grid calibration cannot authorize this new experiment.
5. After review, set `RUN_FULL_TEXTVQA = True` and run its cell:100×16=1600 predictions. Same session reuses environment/checkpoints. Estimate roughly20–30 minutes, but calibrate first; serialization and FPS96 overhead can vary.
6. Download full ZIP/log before ending session. The full run contains all100, including calibration cases; analyze the full run once rather than pooling duplicated cases.

Original `llava/`, original environment, existing grid runner and notebook remain unchanged. New protocol/signatures gate this separate family. CPU checks on Intel Mac do not validate actual Torch execution; notebook prerequisites and pretrained calibration are required.

## Interpretation

A native-drop gain over E-drop would support retaining basis sources in this implementation, but both-stage changes affect downstream states and masks. Native-drop vs native measures total removal of merging, not a fixed-selector effect at both stages. E→FPS/detail/random vs E→E localizes effects at the second cut on common states. Detail→E checks whether E degrades a different first-stage pool. No outcome is assumed in advance. Do not add conditions after seeing calibration accuracy; calibration is for correctness and runtime.

## Pre-calibration review amendment

Before any v1 calibration was run, added `loc-error-native` and `loc-native-error`. These complete basis insertion on/off at each stage without merging. The existing `loc-native-drop` name retains its both-stage meaning. Exact pre-prune state equality is enforced between E→E and E→native, and between native-drop and native-input→E. This is an implementation-specific factorial: stage1 basis changes propagate into stage2 representations, while within a fixed input pool the second-stage intervention is controlled. Ordinary topk tie policies can still differ between the stable research E implementation and original Torch topk; retain the saved rankings to audit ties.

`direct_source_original_positions` in the decoder now explicitly composes both-stage mappings. Carrier slots remain separately recorded. Never use carrier positions as source evidence when basis replacement occurs.

FPS diagnostics replay the distance calculations for the actual ordered centroids in the incoming dtype/device, without new RNG draws or mutation. Input FPS96, scoring FPS10, decoder FPS32/FPS10, and original native references are logged. Finite squared-distance maxima, infinity/NaN counts, argmax tie counts, remaining1e10 counts and reproduction of the selected sequence are recorded. These are diagnostics, not a precision fix. Inf alone does not prove it changed selection; tie counts can be nonzero without overflow. Arbitrary pair distances above the fp16 limit do not necessarily lie on the visited FPS trajectory. Finite rounding can also matter without inf. No native code or selector precision was changed.

A synthetic high-distance fp16 fixture tests overflow detection and verifies RNG/input immutability in Kaggle prerequisites. Readiness still requires zero-skip Torch tests and the new80-prediction calibration.
