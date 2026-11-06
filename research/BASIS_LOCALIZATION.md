# TextVQA basis and stage localization v1

Purpose: test whether retaining basis sources and changing the second selector improves answers. The CPU evidence motivates these tests; it does not establish that stage2 E contributes nothing or that FPS explains ApET's full benefit. GQA's smooth-region explanation is also a hypothesis pending image review.

## Conditions (14)

All compressed arms use96 input tokens and32 after decoder block16. Same fixed TextVQA100, prompts, original locked runtime, checkpoint and deterministic generation settings.

| Input → decoder | Basis replacement | Merge | Conditions |
|---|---|---|---:|
| Full576 | none | none | 1 |
| Original native96→32 | original at both stages | original | 1 |
| E→E | no | no | 1 |
| Detail→detail | no | no | 1 |
| Native-drop | original at both stages | disabled | 1 |
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
3. Set `RUN_LOCALIZATION_CALIBRATION = True`. Run setup and **zero-skip** original Torch prerequisites, then calibration: first5 fixed questions ×14 conditions =70 predictions.
4. Download calibration ZIP/log for review. Existing selector-grid calibration cannot authorize this new experiment.
5. After review, set `RUN_FULL_TEXTVQA = True` and run its cell:100×14=1400 predictions. Same session reuses environment/checkpoints. Estimate roughly15–25 minutes, but calibrate first; serialization and FPS96 overhead can vary.
6. Download full ZIP/log before ending session. The full run contains all100, including calibration cases; analyze the full run once rather than pooling duplicated cases.

Original `llava/`, original environment, existing grid runner and notebook remain unchanged. New protocol/signatures gate this separate family. CPU checks on Intel Mac do not validate actual Torch execution; notebook prerequisites and pretrained calibration are required.

## Interpretation

A native-drop gain over E-drop would support retaining basis sources in this implementation, but both-stage changes affect downstream states and masks. Native-drop vs native measures total removal of merging, not a fixed-selector effect at both stages. E→FPS/detail/random vs E→E localizes effects at the second cut on common states. Detail→E checks whether E degrades a different first-stage pool. No outcome is assumed in advance. Do not add conditions after seeing calibration accuracy; calibration is for correctness and runtime.
