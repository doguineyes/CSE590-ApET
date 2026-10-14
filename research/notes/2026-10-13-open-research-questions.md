# Questions to investigate next — 2026-10-13

Recorded after reviewing the native MMStar baseline, GQA pilot and 800-question
development baseline. The user plans to investigate these in the next session.
No new inference, scoring changes or algorithm changes were performed here.
Reserved evaluation stays outside this exploratory analysis.

## 1. Why did question 2058428 change between runs?

Image `n497658`: airplane photograph. Question:
`Are there airplanes in this scene that are huge?` Gold: `yes`.

| Configuration | Earlier observed pilot | Development run |
|---|---|---|
| Baseline | Yes | Yes |
| Input-only | No | Yes |
| Decoder-only | Yes | Yes |
| Combined | No | Yes |

Recorded image hash, prompt, FPS seed, stage settings and token-count trace
match. The cause is unknown. This is a repeatability case, not one of the 34
current combined losses. A fixed seed has not established exact repeated
compressed answers across every execution context.

Next controls: repeated identical generation; configuration switching in
different orders; logging off/on. Keep checkpoint, GPU placement, dependencies,
prompt and seeds fixed. Record actual FPS bases, selected indices, merge
assignments, error vectors and raw generated answers. Determine whether the
difference begins in compression or later generation before changing code.

Numerical/kernel behavior, instrumentation and persistent execution state are
candidate explanations, not established causes. Do not attribute this change
to question conditioning or a specific removed visual token from answer logs alone.

## 2. How can compression repair a previously wrong answer?

The user's observation is supported by development results:

| Configuration | Baseline-correct lost | Baseline-wrong gained | Final correct / 800 |
|---|---:|---:|---:|
| Input-only | 31 | 28 | 549 |
| Decoder-only | 14 | 11 | 549 |
| Combined | 34 | 26 | 544 |

Input-only and decoder-only have equal net accuracy but different answer
preservation and different correct-question sets. A gain is a recorded change
from incorrect to correct under our scorer; it does not establish improved
visual grounding by itself.

Compression changes the representation supplied to the decoder. Native ApET
also modifies features through merging and FPS basis substitution, so this is
more than simply hiding some unchanged tokens. Test these possible explanations:

- Reduced influence of irrelevant/distracting visual features improves an answer.
- Merging or feature reweighting makes a useful cue more influential.
- A change among close competing answers happens to select the gold label.
- The new wording matches the benchmark label even though both answers are
  semantically reasonable; the apparent gain is mainly a scoring effect.
- Numerical/execution variation produces a gain that does not repeat reliably.

Review gained cases alongside lost and preserved cases. Repeat selected examples
and use controlled interventions before treating a proposed mechanism as evidence.
Keep task type and same-image question comparisons in the analysis. Selection-
only versus merging controls at matched selected sets/budgets remain part of RQ2.

## 3. What does GQA exact matching miss?

The user raised two issues: free natural-language questions can be ambiguous,
and reasonable short answers can differ from the dataset's expected wording.
These need separate treatment from genuine compression-induced visual errors.

Our scorer handles capitalization, outer whitespace and trailing periods. It
does not infer synonyms, choose among plausible reference objects or judge
semantic equivalence. Examples already observed include `Field` versus
`pasture`, `Sunny` versus `cloudless`, `Brown` versus `dark brown`, and
`Woman` versus `Girl`. These are not all automatically equivalent; context,
specificity and the intended annotated object matter.

Image `n497658` also deserves annotation review: the logo-color question has gold
`white`, while all configurations answer `Green`. Inspect the intended object
and source semantics before declaring either the annotation or model incorrect.
The word `huge` in question 2058428 is another interpretation issue to note,
although it does not explain away the observed between-run answer difference.

For a qualitative audit, record whether each selected case is a clear visual
error, vocabulary/specificity mismatch, ambiguous reference/question, possible
annotation issue or uncertain. Keep raw answers, original gold labels and
original exact-match scores. Record any human judgment and its rationale
separately; do not silently change the baseline scorer or selectively relax
labels for a favored configuration.

## 4. Is reconstruction-based compression a good strategy?

The user doubts whether continuing with this older compression approach is a
good idea. Record this as a research hypothesis, not a conclusion requiring an
immediate rewrite or method change.

The present evidence shows computation savings and task-specific answer losses.
Near-equal total accuracy can conceal losses and gains; the same compressed
image can preserve one question and fail another. Neither dataset age nor
individual failures settle the usefulness of the method. The study should
identify where reconstruction-based redundancy aligns with task-relevant answer
contribution, where it fails, and whether merging/adaptive budgets help.

The proposal's success criterion includes reproducible, interpretable limitations
and negative findings. It does not require showing that ApET is universally
beneficial or producing a superior replacement. Keep the current native system
as a reference while these questions are investigated.

## Next session order

1. Revisit 2058428 with a tiny repeated-case/control run.
2. Manually audit a small set of lost and gained development examples, including
   label ambiguity and preserved-correct questions on the same images.
3. Freeze a focused diagnostic case set; collect native values and plan limited
   contribution interventions and selection/merge controls.

References: [development analysis](2026-10-13-original-gqa-development.md),
[pilot analysis](2026-10-13-original-gqa-pilot.md),
[fixed subset protocol](../GQA_SUBSETS.md).
The local gallery is `research/results/gqa-development-case-gallery/gallery.html`;
it shows all 34 combined losses across 27 images, their controls and the separate
airplane repeatability case. Gallery/result artifacts are ignored by Git.
