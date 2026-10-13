# Upstream ApET and CSE590 project changes

Inference uses original `llava/` input and decoder compression, FPS,
reconstruction, and merging. It does not call the extracted `apet_compression/`
core or the modern Hugging Face LLaVA adapter.

Upstream: [Maqkccx/ApET](https://github.com/Maqkccx/ApET), commit
`655f5c299eea44fa5768ef1d709d7c0eb91679da`. Our passed pilot reference is
`9bc740dc2a61ab55cbac857cd130f5af97e4d9de`.

## Native code and the existing compatibility patch

Seven monitored files match upstream exactly:

- `llava/model/modeling_llama_x.py`
- `llava/model/language_model/llava_llama_x.py`
- `llava/model/utils.py`
- `llava/model/builder.py`
- `llava/model/multimodal_encoder/clip_encoder.py`
- `llava/mm_utils.py`
- `llava/conversation.py`

`llava/model/llava_arch.py` retains the existing two-GPU compatibility patch from
`2fe7d0f351140810ec9c69f92e3eef32cdb35f17`: both image-encoding returns move the
selection mask to the projected features' device. It changes device placement,
not reconstruction scores, selection, merging, or token budgets.

The [reference manifest](provenance/original_llava_reference.json) records these
eight native file hashes. Evaluation refuses to call changed monitored sources
this baseline. Reports record the reference, hashes, patch, and match result.
Later algorithm variants need explicit names and recorded source differences;
do not silently replace the baseline reference.

## Our code

| Location | Responsibility |
|---|---|
| `scripts/smoke_original_apet.py` | Native loading, stage settings, seeds, generation, and observation hooks |
| `scripts/eval_original_mmstar.py` | Data validation, paired evaluation with one model load, raw records and summaries |
| `scripts/mmstar_scoring.py` | Versioned choice parsing, correctness and format compliance |
| `scripts/rescore_original_mmstar.py` | CPU rescoring that preserves source evidence |
| `scripts/prepare_gqa_pilot.py`, `scripts/eval_original_gqa.py`, `scripts/gqa_scoring.py` | Pinned same-image subset, short-answer scoring, and grouped diagnostic evaluation |
| `scripts/original_diagnostics.py` | Read-only tracing of native reconstruction, FPS, selection and merge values |
| `scripts/check_original_env.py`, `bootstrap/`, `environments/original/` | Locked runtime and provenance checks |
| `research/` | Protocols, notebooks, notes, tests and artifacts |

The loader points the vision configuration to the separately pinned local CLIP
snapshot. Standalone scripts force Matplotlib `Agg`. These are operational choices.
Observation hooks call native methods without replacing their compression math.
Scoring is our policy, not upstream ApET or the official MMStar evaluation procedure.

GQA adds observational instrumentation in project code, not native source edits.
The observer reads native locals at function return and copies small arrays to CPU.
It does not recalculate FPS or modify native tensor values. Tests compare actual
native function outputs with logging off/on; the pilot also checks answer-level
logging parity. This instrumentation affects timing and is not an algorithm variant.

For the full baseline, budgets, checkpoints, native prompts, seed rule,
preprocessing, FP16, greedy generation, 16-token output maximum, and cached decoding
remain as in the passed pilot. The new work is scoring, full-dataset selection,
bounded downloads after restarts, and clearer provenance. ML dependency versions
and native model sources are unchanged by this update.

Results in `research/results/` are ignored by Git; pushing code does not upload
the ZIPs. Keep them separately. Rescoring always writes new files with previous
scores, raw answers, source hashes and scorer version retained.
