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
| `scripts/prepare_gqa_full.py`, `scripts/eval_original_gqa_full.py` | Full balanced Test-Dev data, native evaluation, paired/type summaries and guarded resume; no compression reimplementation |
| `scripts/prepare_gqa_subsets.py`, `scripts/eval_original_gqa_subset.py` | Frozen image-disjoint development/evaluation sets, nested diagnostics, resume and saved baseline comparisons |
| `scripts/repeat_original_gqa.py` | Two inspected GQA cases, repeated generation, logging off/on and configuration-return controls; unchanged native compression |
| `scripts/prepare_rq1_candidates.py`, `scripts/rq1_candidates.py`, `research/templates/rq1_gallery.html` | CPU-only provisional GQA/MMStar/TextVQA selection, original image copies, provenance and review exports; no inference or compression changes |
| `scripts/prepare_gqa_rq1_round2.py`, `research/curation/gqa_rq1_round2_assistant_review.json` | Verified human-review import, explicit question-quality screening, recorded assistant visual judgments and a separate reviewed GQA proposal |
| `scripts/prepare_mmstar_rq1_round2.py`, `research/curation/mmstar_rq1_round2_assistant_review.json` | Small baseline-correct MMStar RQ1 extension with a separate qualitative gain/baseline-failure appendix; reuses the read-only review renderer |
| `scripts/check_original_env.py`, `bootstrap/`, `environments/original/` | Locked runtime and provenance checks |
| `scripts/prepare_rq1_pilot.py`, `scripts/rq1_protocol.py`, `research/protocols/rq1_textvqa_input_v1.json` | Hash-verified accepted inputs, frozen GQA/MMStar/TextVQA pilot ordering, TextVQA screening provenance/audit gate, token sampling/statistics and native scorer access |
| `scripts/run_rq1_input.py`, `scripts/rq1_native.py`, `scripts/analyze_rq1_input.py` | Project-owned teacher-forced scoring, cached feature replay, shape-preserving singleton zeroing, controls, TextVQA generation and gated interventions, and plots; native sources remain unchanged |
| `scripts/run_rq1_validation.py`, `scripts/rq1_validation.py`, `scripts/rq1_validation_native.py`, `research/protocols/rq1_validation_v1.json` | Bounded case/measurement validation: true text-only and neutral-image MMStar trials, content-preserving option rotations, fixed cross-dataset patches with zero/half/neighbor replacement and numerical controls; no native compression change |
| `scripts/analyze_rq1_validity.py` | CPU verification of archived trial identities, scoring, controls, sensitivity arithmetic and evidence hashes; separate reconstructed UTF-8 HTML/CSV review, with original ZIP/log preserved |
| `scripts/analyze_rq1_scores.py`, `research/protocols/rq1_score_reanalysis_v1.json` | CPU first-token/whole-answer reanalysis and audits using original archived per-token scores; unchanged singleton counterfactual and cohorts, no inference or IH reproduction |
| `scripts/run_rq1_layer.py`, `scripts/rq1_layer.py`, `scripts/rq1_layer_native.py`, `research/protocols/rq1_layer16_v1.json` | Matched input / block-16 measurement: original decoder reconstruction on protected copies with compressed output discarded; temporary output hooks for observation, no-op and single-state zeroing; six frozen questions and random positions, no native source/lock changes |
| `scripts/analyze_rq1_layer.py` | CPU verification and reconstructed layer review from original archives; matched targets/positions, reference controls, historical input repeats, first/whole scores and provenance; no tensor loading or inference |
| `scripts/run_rq1_layer_ih.py`, `scripts/rq1_layer_ih.py`, `scripts/rq1_layer_ih_native.py` | Optional project implementation of IH's released physical retain-only operator at the same block-16 boundary, with the supported 1×1 unit and first-gold-token probability difference. Existing zeroing is reused unchanged. Temporary hooks preserve original positions and adapt old native SDPA/RoPE geometry; separate source-pinned protocol and new calibration gate, no native source or lock edits. This is not an exact IH benchmark reproduction. |
| `scripts/analyze_rq1_layer_ih.py`, `scripts/analyze_rq1_layer_ih_full.py` | CPU verification of archived IH arithmetic, retained-position geometry, controls, source/feature hashes and exact historical repeats; full runner also verifies the embedded rerun calibration and canonical input repeats, compares sampled/full effects and exports padding-aware maps. Separate derived results; no inference or native code change |
| `scripts/rq1_input_ih.py`, `scripts/rq1_input_ih_native.py`, `scripts/run_rq1_input_ih.py`, `scripts/prepare_rq1_ih_inputs.py` | 30-case input-stage 1×1 physical retention with strict archived E/zero reuse, fresh baseline/reference/zero-audit controls, original positions and temporary RoPE lookup hooks; portable hash-pinned references, CPU arithmetic/geometry checks and a six-case calibration gate. Original input/layer/native measurement files and locks stay unchanged. The calibration and 17,280-effect full sweep passed. |
| `scripts/analyze_rq1_input_ih.py` | CPU verification of input IH archives, exact historical feature/E/zero reuse, reference associations, fresh-answer scoring, retained geometry and evidence/source hashes; separate descriptive effect/timing summaries and calibration-gate validation. No model or tensor loading and no measurement-source change. |
| `scripts/summarize_rq1_input_ih_full.py` | Saved-data synthesis of all 30 verified input questions and the existing six-case layer panel: per-question/dataset summaries, same-image comparisons, basis/padding/flag sensitivity, sparse retained-state examples and an interactive gallery. Raw archives and primary cohort stay unchanged; no inference or RQ2 algorithm implementation. |
| `scripts/rq2_protocol.py`, `scripts/rq2_native.py`, `scripts/run_rq2_input_merge.py`, `research/protocols/rq2_input_merge_v1.json` | First RQ2 30-case generated-answer pilot: original full576, native input288, and project-owned native selection+basis with only merging disabled. Scoped instance hooks retain original native FPS/E/selection/basis placement; exact source/selection and native feature/generated-token parity gates, repeated references, whole-reference scores, paired rescue/harm summaries, and ZIP/log evidence. Native files and runtime lock remain unchanged. Kaggle native/pretrained execution is required before scientific results are claimed. |
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

The [RQ1 candidate preparation](RQ1_CANDIDATES.md) uses saved native baselines
for GQA/MMStar and a pinned TextVQA screening pool. Its manifests are provisional
project-owned selections. The later 100-case TextVQA generation baseline is now
archived separately; human review overlays leave original labels and scores intact. Candidate preparation does not
implement an RQ1 intervention runner or resume numerical debugging.

The separate [RQ1 input pilot](RQ1_INPUT_PILOT.md) now implements a small GQA-first
measurement run and optional TextVQA generation baseline. It reads E from the
native compressed input branch on cached full-path CLIP features, then measures
reference-answer log-probability changes from zeroing a full-context projected
embedding. Temporarily replaying cached vision/encoding outputs is measurement
instrumentation, not an upstream method or an ApET variant. It does not validate
decoder-layer interventions or resume the deferred airplane numerical study.
The no-op, repeated references and cached/native parity checks gate each question.
The added target-format audit records actual native generation tokens and compares
literal/capitalized reference spellings. It diagnoses project measurement alignment;
it does not edit manifest labels, automatically select a new target, or alter ApET.
After the confirmed casing mismatch, measurement v2 uses fixed initial-capital
GQA target rendering, preserves raw labels and adds generated-prefix/token-score
alignment controls. This changes our reference-probability measurement only;
native compression, generation prompts and dependency locks remain unchanged.

The [TextVQA extension](RQ1_TEXTVQA_INPUT_PILOT.md) adds six frozen diagnostic
cases, an annotation-only initial-capital reference policy and a prerequisite
six-case target audit. It gates a two-question calibration and later six-question
input-stage sweep using the existing scoring/zeroing implementation. Fresh full
consensus credit and exact token-prefix alignment are required; no prediction is
adopted as a replacement gold target. Accepted manifests and all 100 baseline
records stay unchanged. This extension awaits Kaggle execution validation.
