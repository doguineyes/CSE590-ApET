# 2026-10-13: original ApET executes on Kaggle

Evidence: `original-apet-smoke-results (1).zip`, supplied after the corrected
Kaggle run. Runtime and pretrained smoke reports passed; preflight completed.

## Recorded setup

- Commit: `2fe7d0f351140810ec9c69f92e3eef32cdb35f17`, clean checkout.
- Original lock SHA256:
  `3a838c268ffdb607c05b3e26e473c5fb53d975d7bb6e0c900e89ee6ba95a5a5c`.
- Python 3.10.18, Torch 2.1.2+cu121, torchvision 0.16.2+cu121,
  Transformers 4.37.2, Accelerate 0.21.0, NumPy 1.26.4, Pillow 10.4.0,
  huggingface-hub 0.33.4.
- Two Tesla T4 GPUs, 15,360 MiB each. Native FP16 with SDPA, no CPU/disk
  parameters. Original `LlavaLlamaForCausalLM_X` loaded in 23.702 seconds.
- Original-format `liuhaotian/llava-v1.5-7b` revision
  `4481d270cc22fd5c4d1bb5df129622006ccd9234`; CLIP revision
  `ce19dc912ca5cd21c8a653c79e251e808ccabcd1`.
- Layers 0–15 and embeddings on GPU 0; layers 16–31, projector, and language
  head on GPU 1. Actual vision tower on GPU 0: component device records are more
  informative here than the initial automatic device map's vision entry.

## Execution evidence

Question: “What color is the square? Answer briefly.” The synthetic image is a
red square on a white canvas. Both runs use the native Vicuna conversation prompt,
seed 590, greedy generation, and a 4–16 new-token bound to exercise cached decoding.

| Setting | Visual tokens | Answer | Output tokens | Generation seconds | Peak allocation GPU 0 / 1 (GiB) |
|---|---|---|---:|---:|---|
| Baseline | 576, unchanged | Red and white square | 5 | 3.503 | 8.134 / 7.259 |
| Combined | 576 → 288 at input; 288 → 96 after layer 16 | Red. It is a red square. | 9 | 1.141 | 8.134 / 7.212 |

The combined decoder sequence shrank from 339 to 147, exactly the 192 visual
tokens removed. Native input and decoder paths therefore execute together,
including cached generation, on the available GPUs.

These timings are not speedup evidence: the baseline ran first without a separate
warmup, and the answers had different lengths. The white canvas also means the
baseline wording should not be declared an accuracy failure from this smoke.
GPU allocation excludes reserved memory and other CUDA usage; no capacity claim
for longer contexts or other models follows from this single image.

## Decision

Proceed with the original LLaVA implementation and a locked compatible environment.
The required changes so far were operational: force Matplotlib `Agg` in standalone
scripts, and move the native image selection mask to the projected features'
device for two-GPU indexing. Compression math is retained. This is execution
feasibility, not proof of full paper reproduction or of every stage configuration.

The previous modern HF input-only experiments remain useful workflow evidence.
Create fresh native baselines: checkpoint format, prompt, and preprocessing differ,
so their aggregate scores cannot establish a direct native/HF comparison.

## Next artifact

The [same-notebook MMStar pilot](../ORIGINAL_MMSTAR_PILOT.md) checks 24 selected
images across baseline, input-only, decoder-only, and combined settings with one
model load. It records paired APR, category scores, lost/gained IDs, raw predictions,
image bytes, prompts, and source/environment provenance before formal baselines.

The pilot adds only PyArrow 19.0.1 to the original environment. Existing locked
package versions are unchanged. Its new lock SHA256 is
`6458cb0ab526857c483b41c19cb87e63179507ebaf6e48f4e9020d18d261f3a2`;
this is distinct from the archived passed-smoke environment above. Data/scoring
and mocked orchestration checks passed locally; the new GPU pilot awaits Kaggle.

After inspecting this pilot: full MMStar baseline/compression runs, small GQA and
TextVQA scoring checks, then systematic dataset baselines. RQ1 still requires
per-token reconstruction diagnostics and controlled answer contribution
interventions; the current execution traces do not provide that evidence.
