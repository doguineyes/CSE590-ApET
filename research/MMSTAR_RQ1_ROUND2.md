# MMStar RQ1: small baseline-correct stress subset

MMStar remains a required RQ1 dataset, used for cross-task confirmation alongside
the main GQA mechanism analysis and the separate TextVQA fine-detail study.
This step adds a small set using saved predictions and original images. No new
inference, scoring-policy change, environment change or native ApET edit is needed.

## Review and recommended scope

The supplied human review matches the original 30-case manifest SHA256 and IDs:
**17 Keep, 12 Exclude and one Ambiguous**. The 17 Keeps contain:

| Saved input-only outcome | Human Keeps | Role now |
|---|---:|---|
| Baseline correct → wrong | 5 | Primary loss cases |
| Baseline correct → correct | 2 | Primary preserved cases |
| Baseline wrong → correct | 5 | Qualitative gain appendix |
| Baseline wrong → wrong | 5 | Baseline-failure context appendix |

All human Keep decisions are retained. The distinction is analysis scope, not
a judgment that baseline-wrong questions are invalid. We do not deliberately add
both-wrong or gain cases to the primary set.

The second round initially samples **12 baseline-correct candidates** to fill
category/outcome gaps. Assistant inspection of those images finds several
ambiguous or specialist-dependent choices. A bounded six-question replacement
review follows, with its origin and reasons recorded. **Eleven additions survive**.
The recommendation therefore has **18 primary questions**: six input-only losses
and 12 preserved answers, with all six MMStar categories represented:

| Category | Primary questions |
|---|---:|
| Coarse perception | 3 |
| Fine-grained perception | 2 |
| Instance reasoning | 4 |
| Logical reasoning | 3 |
| Math | 3 |
| Science & technology | 3 |

The ten original human Keep gains/both-wrong cases remain in a separate appendix.
No further human review is required for this round. Assistant review is not
independent human validation; the selection remains a proposal for the RQ1 pilot.

## What the added examples test

Accepted additions cover elementary polygon geometry, chart threshold counting,
axis labels, stacked-bar arithmetic, map labels, flowering-stage identification,
food-web edge counting, soccer recognition, applying-makeup captioning, an
explicit hydrant-relative location and image-coordinate localization.

The new math loss is independently checkable: the pentagon's five angles sum to
540°, giving x=102 and angle H=x−5=97° (option A). Its source predictions are
preserved. Added reference cases also exercise non-A answer positions.

The assistant rejected baseline-correct candidates when visible evidence was
insufficient: white versus gray jersey, natural grass versus artificial turf,
driving versus parked car, vague guitarist spacing, age/gender-only skiing
choices, specialist art attribution and an unclear anatomical pointer. A correct
benchmark option does not by itself establish usable visual grounding.

The existing bird-beak case `617` retains its original option-name typo, already
flagged by the human reviewer. Case `604` has two real choices and two `nan`
placeholders. Both are explicitly tagged for sensitivity summaries; their
questions, options, targets and scores remain unchanged.

## Selection and interpretation

New candidates must be baseline-correct, on images whose byte hashes are absent
from the entire first round, with one added question per image. Four distinct,
non-placeholder choices are required for additions. Original retained cases can
have fewer choices; we do not rewrite them to satisfy the new screening rule.

Slots favor missing math/science coverage and preserved comparisons for the other
categories. Obvious speculative, demographic, specialist and output-cap cases
are flagged. Within each declared task family, the initial selector prefers an
underrepresented gold option, then a seeded SHA256 rank (seed 592). Six subsequent
replacement questions are selected from eligible baseline-correct cases using
task wording and remaining coverage needs. The full 1,500-row audit records
automatic flags, source-image exclusions and reviewed IDs. Rejected images and
individual review rationales are preserved.

This is conditional, outcome-enriched sampling, and the review saw saved answers.
It cannot estimate benchmark-wide accuracy or an unbiased error/contribution
correlation. Baseline correctness can reflect guessing, language knowledge or
answer-position bias; it does not prove that the uncompressed model used the
required image evidence. Retain question-only and limited option-order controls
as useful checks in the eventual pilot, alongside no-op/reference repeats.
No same-image paired-question design is claimed for MMStar.

The native full baseline actually scored **521/1,500 = 34.73%** under project
policy `mmstar-explicit-choice-v2`; the supplied advice's 33.9% is not this run's
score. The archived baseline predicted A on 70.6% of questions. Some source
questions have two or three real choices and `nan` placeholders, so a blanket
25% random-guess comparison is inappropriate. The report records parsed option
counts and the complete baseline's option distribution without rescoring.

The evidence supports a capability concern, but it does not isolate reasoning,
perception or OCR as the cause of baseline errors. MMStar also has wording/label
ambiguities. Avoid treating “GQA quality versus MMStar capability” as a universal
separation or assuming that every retained loss is caused by visual evidence removal.

## Open or rebuild

Open `research/results/rq1-mmstar-reviewed-v1-s592/index.html` for the primary
set. `qualitative-gallery.html` contains the ten appendix cases; `review-audit.html`
contains all 30 human decisions and all 18 assistant decisions. These pages embed
original images and are read-only. The adjacent ZIP also includes manifests,
review copies, policy, audit and preparation-source snapshots. Artifacts are
ignored by Git; keep the ZIP separately.

The assistant decisions are committed in
`research/curation/mmstar_rq1_round2_assistant_review.json`. To rebuild from the
repository root using the existing CPU editor environment:

```bash
environments/editor/.venv/bin/python scripts/prepare_mmstar_rq1_round2.py \
  --human-review /Users/dog/Downloads/rq1-mmstar-review.json \
  --assistant-review research/curation/mmstar_rq1_round2_assistant_review.json \
  --output research/results/rq1-mmstar-reviewed-rebuild-s592
```

Use a fresh output path. If the export moves, use `round1-human-review.json` from
the bundle. The original full native MMStar result ZIP and first-round candidate
manifest are required locally; the script verifies their identities and protects
existing outputs. No network download or GPU is used.

Next use a few of these cases in the cross-dataset RQ1 intervention pilot. The
candidate schema is not automatically accepted by existing baseline runners.
GQA/TextVQA datasets and the deferred numerical investigation are unchanged.
