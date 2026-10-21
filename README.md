# CSE590: Understanding ApET Visual Token Compression

This repository contains my CSE590 research project at Stony Brook University,
built on [ApET](https://github.com/Maqkccx/ApET). The project studies when and why
reconstruction-based visual token compression works in vision-language models.

## Research goals

- Compare reconstruction error with a token's contribution to answering a question.
- Separate the effects of token selection and merging.
- Explore adaptive token budgets based on these findings.

See the [project proposal](research/reports/Peng_Chen_CSE590_ApET_Project_Proposal_v2.pdf)
for the research plan.

## Current progress

The LLaVA input-stage compression core has been extracted into a model-independent
module and connected to Hugging Face LLaVA-1.5-7B. Baseline and compressed runs on
all 1,500 MMStar samples, plus fixed-output timing benchmarks, have been completed
on Kaggle with two T4 GPUs. See the [experiment note](research/notes/2026-10-09-kaggle-apet.md).

The original ApET LLaVA implementation also passed a baseline and combined
input/decoder smoke on the same hardware. We will use this implementation for
the planned research experiments. See the [passed smoke note](research/notes/2026-10-13-original-apet-smoke.md)
and [full MMStar results](research/notes/2026-10-13-original-mmstar-full.md).
The [same-image GQA pilot](research/notes/2026-10-13-original-gqa-pilot.md) also passed,
with native reconstruction/selection/merge logging. Main GQA experiments use
[fixed development and evaluation subsets](research/GQA_SUBSETS.md), with 800
questions each and separate image groups.
The [800-question development baseline](research/notes/2026-10-13-original-gqa-development.md)
is complete; reserved evaluation remains for later confirmation.
The [repeatability notebook](research/notebooks/kaggle_original_gqa_repeatability.ipynb)
checks the airplane answer drift with a stable control; see its
[instructions](research/GQA_REPEATABILITY.md).
The [RQ1 candidate preparation](research/RQ1_CANDIDATES.md) provides image/question
review galleries for GQA, MMStar and TextVQA before freezing the research subsets.
Dataset curation is complete for now: GQA has 49 questions, MMStar has 18 primary
plus 10 qualitative cases, and all 100 prepared TextVQA questions are retained.
See the [current dataset summary and next RQ1 steps](research/notes/2026-10-15-rq1-dataset-preparation-summary.md).
The [RQ1 input pilot](research/RQ1_INPUT_PILOT.md) passed its small GQA measurement
calibration after correcting reference-answer casing. The 12-question GQA
input-stage pilot is also complete; see the [pilot review](research/notes/2026-10-16-rq1-gqa-input-pilot-v2-review.md).
The [three-case MMStar calibration](research/notes/2026-10-16-rq1-mmstar-calibration-v2-review.md)
also passed. The [TextVQA baseline](research/notes/2026-10-16-rq1-textvqa-baseline-review.md)
is complete. The [three-case MMStar full-patch check](research/notes/2026-10-16-rq1-mmstar-fullpatch-review.md)
also passed. Full input-stage intervention sweeps are now complete for GQA (12
questions), MMStar (12) and TextVQA (6): 17,280 measurements across 24 images.
See the [findings and next steps](research/notes/2026-10-16-rq1-input-stage-summary-and-next-steps.md).
The [saved-data analysis](research/notes/2026-10-17-rq1-saved-data-analysis.md)
compares reconstruction-error groups and GQA same-image questions without new inference.
The [bounded validity checks](research/RQ1_VALIDITY_CHECKS.md) completed on Kaggle:
several selected MMStar diagrams show option/grounding concerns and sensitivity
rankings depend on the intervention. See the [review](research/notes/2026-10-17-rq1-validity-review.md).
The [saved-score reanalysis](research/notes/2026-10-17-rq1-score-reanalysis.md)
checks first-token scoring, answer pieces, same-image pairs and flagged subsets;
the broad weak-alignment finding remains, with important TextVQA qualifications.
The [matched input / layer-16 calibration](research/notes/2026-10-17-rq1-layer16-calibration-review.md)
passed for six questions and 32 positions at both stages, with zero reference
control drift. The full 576-position comparison is now complete using the same
inputs and original locked environment.
The [IH comparison calibration](research/notes/2026-10-17-rq1-layer-ih-calibration-review.md)
also passed at layer 16. The [full six-case review](research/notes/2026-10-17-rq1-layer-ih-full-review.md)
verifies all 10,368 measurements. Neither sensitivity measure aligns consistently
positively with reconstruction error; the full grid also finds strong retained
states missed by the 32-position sample. These are intervention-specific effects,
not intrinsic token importance or generated-answer accuracy.
The [input-stage IH extension](research/RQ1_INPUT_IH.md) is complete for the same
30-question panel: 17,280 retention effects with exact archived E/zero reuse and
calibration repeats. The [full review](research/notes/2026-10-18-rq1-input-ih-full-review.md)
compares both counterfactuals across datasets and the six-case layer panel.
Alignment varies by question and operator; strong sparse states can be missed by
small position samples. Next is a controlled selection/merging comparison.
The [RQ2 v2 pilot](research/RQ2_INPUT_MERGE_V2.md) compares full LLaVA, native input
ApET and identical selection/basis without merging on the same 30 questions.
It reuses captured features and records native numerical variation separately.
The [Kaggle full-panel review](research/notes/2026-10-19-rq2-v2-full-review.md)
records 19/30 correct with merging and 21/30 without it, including recovery of
Brolio. This selected diagnostic panel is not a benchmark accuracy estimate.
The [E/random × merge comparison](research/RQ2_CROSSED.md) passed calibration and
full30; see the [review](research/notes/2026-10-19-rq2-crossed-full-review.md).
Random controls recover several known failures, with seed-dependent merge effects.
The separate [Brolio restoration diagnostic](research/notes/2026-10-19-rq2-brolio-oracle-review.md)
recovers the correct name by restoring one merged state, while a matched control does not.
The separate [routing and target-protection comparison](research/RQ2_ROUTING.md)
uses fixed native selection, count-preserving shuffled destinations and high-E/random
protected targets. Its [Kaggle full30 passed](research/notes/2026-10-19-rq2-routing-full-review.md).
Policy effects are small and high-E protection rarely changes routing in this cohort.
The [pure-random calibration](research/RQ2_PURE_RANDOM_CALIBRATION.md) now adds
an FPS/E-free random baseline and native merge/no-merge comparisons at 288 and
64 input tokens. The [six-question calibration passed](research/notes/2026-10-20-rq2-pure-random-calibration-review.md),
verifying actual 64-token inputs; native ApET does not outperform pure random on
this small selected panel. The [TextVQA100 comparison](research/RQ2_TEXTVQA100.md)
is available as an optional input-only ablation. The
[staged calibration passed](research/notes/2026-10-21-rq2-staged-calibration-review.md):
native96→32 and288→96 versus three random seeds at identical boundaries/counts.
Next is the [staged TextVQA100 comparison](research/RQ2_STAGED_TEXTVQA100.md), now
implemented for all100 questions with900 primary predictions, compact evidence
and paired preservation/score summaries. Its
[Kaggle run passed](research/notes/2026-10-21-rq2-staged-textvqa100-review.md): native
ApET outperforms all three random seeds at both schedules in this fixed cohort,
while exact-text failures remain.
The following [GQA development800 comparison](research/RQ2_STAGED_GQA800.md) is
also prepared, using the same schedules, shared image inputs and four optional
image-based parts. Reserved evaluation remains untouched; GPU validation is pending.

## Code layout

- `apet_compression/`: compression core, HF LLaVA adapter, and source-parity checks.
- `scripts/`: inference, MMStar evaluation, comparisons, and benchmarks.
- `research/`: proposal, papers, notebooks, tests, and research notes.
- `bootstrap/`: environment setup scripts.
- `llava/`, `qwen/`, `video-llava/`: original model integrations retained for reference.

Start with the [code reading guide](research/notes/2026-10-10-code-reading-plan.md)
and [compression module documentation](apet_compression/README.md).

## Development workflow

Read and edit on the Mac, version code with GitHub, and run research tests on
Kaggle. Google Cloud is planned for later experiments. Separate `uv` lockfiles
manage the Linux experiment runtime and Intel Mac editor environment.

**Intel Mac editor setup:**

```bash
bash bootstrap/setup_editor_macos.sh
```

**Original ApET on Kaggle T4 ×2:**

```bash
bash bootstrap/setup_original_linux.sh --require-cuda --expected-gpus 2
```

Run these commands from the repository root. Follow the
[Mac editor guide](environments/editor/README.md) for VSCode configuration and the
[original ApET guide](environments/original/README.md) for the first Kaggle test.
The [earlier HF runtime guide](research/KAGGLE_RUNTIME.md) documents the extracted
input-stage experiments.

## Original ApET

The upstream documentation is preserved in [ApET_README.md](ApET_README.md).
See [implementation scope](research/IMPLEMENTATION_SCOPE.md) for original code,
compatibility patches, and our experiment/scoring code.
The extraction covers LLaVA input-stage compression. The separate original-code
runtime supports studying both input and decoder compression. Project RQ1
instrumentation measures input-stage sensitivity and includes a decoder-layer-16
zeroing/retain-one runner, validated on a bounded six-question full-grid panel.
