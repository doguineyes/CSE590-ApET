# Project review and next steps — 2026-10-14

Today is a review checkpoint. We have a working environment, native ApET
baselines and useful diagnostic cases. Keep these references fixed while checking
repeatability and interpreting the results. No inference, dependency changes,
scoring changes or native algorithm edits were performed during this review.

## What is complete

- Mac editing/Git → Kaggle isolated runtime → downloaded evidence works. The
  original runtime has its own Python and dependency lock; it does not depend on
  downgrading the notebook kernel. Kaggle remains our primary execution platform.
- Original LLaVA-1.5-7B runs in FP16 across two T4s. These are two devices serving
  one sharded model, not two independent inference workers.
- Original MMStar: all 1,500 questions, four configurations, 6,000 predictions.
- GQA: 40-question observed pilot and frozen 800-question development set on
  100 images, with 3,200 development predictions. The separate 800-question
  evaluation set has not been run.
- Predictions, prompts, image identities, seeds, settings, source snapshots,
  environment identity and paired results are archived and audited. The complete
  GQA development ZIP restores the omitted matching dependency lock without
  changing any original archive member.

The current research reference is native ApET, with the documented two-GPU
mask-device compatibility patch. Our environment setup, evaluation scripts,
recording, scoring and diagnostics are project additions. The earlier extracted
HF adapter experiments are historical pipeline work, not this native baseline.

## Checks performed today

All three saved audits report passed with no errors, and their source ZIP hashes
still match. The complete GQA bundle also matches its recorded hash; every
original ZIP member is unchanged and its only added member is the dependency
lock. All eight monitored native source files still match the passed reference.
The corrected GQA notebook packaging filter remains in place.

These checks establish artifact integrity and continuity with the audited runs.
They do not establish repeated GPU answer identity or causal explanations.
There is no reason to rerun MMStar, the 800-question GQA baseline or the stopped
full GQA job for today's review. No partial full-GQA result ZIP is present locally;
that optional run is not needed for the current study.

## What the results support

| Configuration | MMStar correct / 1,500 | GQA development correct / 800 | GQA lost / gained versus baseline |
|---|---:|---:|---:|
| Baseline | 521 | 552 | Reference |
| Input-only | 524 | 549 | 31 / 28 |
| Decoder-only | 520 | 549 | 14 / 11 |
| Combined | 524 | 544 | 34 / 26 |

Nearly equal total accuracy hides different failures. Combined GQA loses 34
baseline-correct answers across 27 images and gains 26 previously wrong answers.
Every lost question has at least one baseline-correct, compression-preserved
question on its image. Open query questions have lower preservation than
verification questions in this development sample.

Combined mean generation time was about 40% lower on GQA and 42% lower on MMStar.
These are observed single-session timings with short, variable outputs and
recording hooks. They are not controlled repeated performance benchmarks.

The baselines find cases for the proposal; they do not yet answer RQ1 causally.
We have not measured an intervention-based answer-contribution proxy. All present
ApET configurations use merging, so input-only versus decoder-only is not a
selection-versus-merging experiment. Stage settings also use different budgets.

## Initial image review today

These are provisional human observations, separate from the unchanged exact-match
scores. They are not new labels or evidence of a compression mechanism.

| Question ID / image | Recorded change | Visual review and research use |
|---|---|---|
| `201428959` / `n64959` | Baseline Right → input/combined Left; gold right | The refrigerator is right of the visible cardboard boxes. This is a useful relation-loss case. |
| `201428953` / `n64959` | All configurations Yes; gold yes | A similar right-of relation is preserved in a verification question. Useful same-image control, though Yes alone does not prove visual grounding. |
| `20119066` / `n540852` | Tie → Suit under input/combined; gold suit | The men visibly wear both ties and suits. A scored gain does not necessarily repair a visual misunderstanding; the question permits different garment specificity. |
| `201570646` / `n382416` | Wood → Glass under all compressed settings; gold glass | The cabinet has transparent doors and a dark frame/base. The intended annotated part matters; this is an ambiguity/material case rather than an established grounding gain. |
| `201713417` / `n455563` | Coffee cup → Mirror under all compressed settings; gold mirror | Both a cup and mirrors are visible in front of the woman. The broad question needs reference-object review before interpreting this as a clear gain. |

Keep the scored losses/gains, raw answers and gold labels. Add human judgments
with reasons separately. Do not retrospectively relax the scorer for selected
configurations. The existing gallery includes these images and all 34 combined
losses with their same-image questions.

## What needs a small rerun

Priority: question `2058428`, image `n497658`,
`Are there airplanes in this scene that are huge?`, gold yes.
Input-only and combined answered No in the observed pilot and Yes in development;
baseline and decoder-only answered Yes in both.

Today's archive check confirms the configuration order was identical in both:
input-only → decoder-only → combined → baseline. A simple difference in that
question's configuration order therefore does not explain the change. Earlier
execution history and instrumentation are still possible differences. The pilot's
logging parity check covered its first question, not this question.

Next GPU check should be only a few dozen generations, in one model session:

1. Repeat identical settings with the same recorded seed and GPU placement.
2. Compare logging off/on for this actual question and include one stable control.
3. Switch configurations and return to the original configuration; if needed,
   repeat in a fresh model process to distinguish persistent execution state.
4. Save raw answers and native FPS bases, selected indices, reconstruction errors
   and merge assignments. Locate the earliest difference before changing code.

The helper resets Torch seeds per generation; this does not explain the observed
disagreement. Native merging uses `scatter_add_`, and FPS uses a random initial
index. These are code observations worth instrumenting, not established causes.
If repeats agree, report the scope of that check; do not claim the historical
disagreement is solved. No reason to repeat all 800 questions for this check.

## Research order after review

1. Audit a small development-only collection of losses, gains and preserved
   controls. Distinguish clear visual errors, reference ambiguity, vocabulary or
   specificity mismatches, possible annotation issues and uncertain cases.
2. After the repeatability check, freeze a manageable diagnostic collection.
   Record its outcome-based selection explicitly; it is for explanation, not an
   unbiased estimate of accuracy. Keep the reserved evaluation set untouched.
3. Start RQ1 with native error/selection observations and limited token-group
   interventions at a few stages. Define answer contribution operationally
   before interpreting reconstruction error as task relevance. Preserve token
   mappings through basis substitution and merging.
4. For RQ2, explicitly register controlled variants that compare dropping and
   merging at matched selected sets/budgets. Keep the native implementation as an
   unchanged reference. Adaptive budgets or basis changes come after these checks.

MMStar also has an unresolved A-answer imbalance (70.6% predicted A versus 28.6%
gold A). A later small option-order control can investigate it while retaining
the existing baseline. It need not delay focused GQA diagnostics.

We have enough evidence to begin mechanism research. Doubts about reconstruction-
based compression are hypotheses to test; explaining its limits is a valid
project outcome. Another broad baseline run or an immediate redesign would not
resolve the current questions.

## References

- [Native source reference](../provenance/original_llava_reference.json)
- [MMStar analysis](2026-10-13-original-mmstar-full.md)
- [GQA development analysis](2026-10-13-original-gqa-development.md)
- [Open research questions](2026-10-13-open-research-questions.md)
- [Frozen GQA subset protocol](../GQA_SUBSETS.md)
- Local gallery: `research/results/gqa-development-case-gallery/gallery.html`

Result ZIPs and the gallery are ignored by Git; retain them separately. This note
does not implement or run the proposed repeatability experiment.

Follow-up: the [repeatability preparation note](2026-10-14-gqa-repeatability-preparation.md)
records the subsequently added runner/notebook and CPU verification. GPU results
remain pending.
