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
The [RQ1 input pilot](research/RQ1_INPUT_PILOT.md) is ready for a small GQA
measurement check on Kaggle, with optional TextVQA baseline generation.

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
runtime supports studying both input and decoder compression. Next: validate
the small RQ1 input-stage measurement on Kaggle before expanding it.
