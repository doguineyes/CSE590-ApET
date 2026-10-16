# GQA RQ1: second candidate round and assistant selection

The first human review is incorporated without editing its decisions, labels or
saved predictions. A stricter question-quality rubric and a second round of
assistant visual review produce a **49-question / 27-image recommendation** for
the next RQ1 pilot. No additional human review is required for this round.
This is a reviewed proposal; the intervention protocol has not been frozen.

## What was selected

| Source | Review outcome | Adopted |
|---|---|---:|
| First-round human review | 30 Keep, 13 Exclude, 11 Ambiguous, 3 Uncertain, 3 annotation issues | 22 |
| Second-round assistant review | 39 initial candidates on 13 new images, plus 10 inspected same-image replacement questions | 27 |

The eight human Keep cases set aside by the stricter rubric remain Keep in the
copied original review. They involve broad scene naming, demographic qualifiers,
subjective size/cleanliness/body attributes or similarly avoidable confounds.
This does not assert that their answers are wrong. The exact IDs and reasons are
in `review-report.json`; the full audit displays every original decision.

All 13 new images were directly inspected at their archived resolution. Each of
the 49 second-round reviewed questions has an assistant decision, confidence and
specific visual/semantic rationale. Rejected candidates are retained in the audit.
The assistant kept 27, marked 11 ambiguous, excluded 5, marked 4 uncertain and
flagged 2 possible annotation issues. These are assistant judgments, not a second
human annotation or independently established ground truth.

The combined recommendation contains **38 preserved-correct, six loss, two gain
and three both-wrong questions**, using the saved input-only evaluator. All four
original configurations, task types, prompts, seeds and image hashes remain in
the manifests. The airplane image is excluded and reserved evaluation is unused.

## Open the results

Open `research/results/rq1-gqa-reviewed-v1-s591/index.html` for the recommended
questions, images, saved answers and review reasons. This gallery is read-only.
Open `review-audit.html` to see all 60 human decisions and all 49 assistant
decisions, including exclusions. No review export or confirmation is needed.

The adjacent ZIP contains original image copies, embedded-image offline galleries,
source review copies, manifests, an 800-question selection audit, policy and
preparation source snapshots. Artifacts under `research/results/` are ignored by
Git. The committed assistant review is
`research/curation/gqa_rq1_round2_assistant_review.json`.

## Selection procedure and limits

1. Verify the human export against the exact first-round manifest SHA256 and IDs.
2. Exclude all first-round images and the deferred airplane from the new-image pool.
3. Apply the recorded automatic question-quality rules consistently. Broad scene
   questions, demographic/subjective targets, known ambiguities, output-cap cases
   and likely singular/plural or synonym artifacts are flagged. Rules are
   conservative heuristics and can exclude otherwise reasonable questions.
4. With seed 591, choose three quality-reference images, seven loss-image slots
   and three gain-image slots, with a preserved control required in diagnostic
   images. A missing gain slot becomes a documented quality-reference fallback.
5. Choose three questions per image, then visually review the candidates. Inspect
   ten same-image alternatives as possible replacements when initial wording or
   labels are unsuitable. Their origin and every decision are recorded explicitly.
6. Combine assistant Keep cases with human Keep cases passing the stricter rules;
   recompute same-image control links within that recommendation only.

Reference-image sampling has no correctness quota but is **conditional on quality
filters that include saved-answer aliases and output lengths**. It is not an
outcome-blind random sample. Visual review also saw saved answers and is not
blinded. The collection is explanatory and outcome-enriched: neither its accuracy
nor its error/contribution correlations estimate the whole GQA population.

An apparent score loss can be a valid alternative answer: for example, two people
sit at a table, so changing `Boy` to `People` is not a clean perception failure.
A fork with exposed handle and partially hidden tines does not make `covered`
unambiguous. The second round excludes both examples. The same standard excludes
the apparent resting-dog gain because resting cannot be confidently read from
the static image. No gain quota is forced after visual review.

Four new loss candidates survive: television/phone and lamp-left/right in the
hotel image, refrigerator-left/right in the kitchen, and medicine-cabinet-right/
left in the bathroom. These are reviewed candidates for intervention, not causal
findings about compression.

The new coat/bin-color gain (`20596219`) is **tentative**: both appear dark, but
the bin is small and hue differences are possible. The retained cabinet loss
`201570615` has no retained same-image preserved control; use it as an unpaired
supplement. Broad absence controls, coarse scene controls and the refrigerator
paraphrase are tagged in the assistant review. In particular, the hotel image's
absence control provides no localized positive target and may have a strong room
prior. Run sensitivity summaries without these weaker controls/cases.

## Rebuild and next step

From the repository root, using the existing CPU editor environment:

```bash
environments/editor/.venv/bin/python scripts/prepare_gqa_rq1_round2.py \
  --human-review /Users/dog/Downloads/rq1-gqa-review.json \
  --assistant-review research/curation/gqa_rq1_round2_assistant_review.json \
  --output research/results/rq1-gqa-reviewed-rebuild-s591
```

Use a new output path; existing directories and ZIPs are protected. If the human
export moves, use `round1-human-review.json` from the prepared bundle. The source
native GQA development ZIP and first-round manifest are also required locally.
No download, PyTorch installation, new inference or environment change is needed.
The recorded assistant decisions are bound to the source predictions, exact
image hashes and the 39 initial candidate IDs; different sources fail validation.

Next implement a small RQ1 pilot on a few paired examples with no-op controls,
reference repeats and an answer-support proxy. The new GQA recommendation does
not replace the separate MMStar and TextVQA RQ1 subsets or automatically work with
the old baseline runner. Detailed numerical investigation remains deferred.
