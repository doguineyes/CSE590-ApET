# RQ1 candidate preparation — 2026-10-14

Prepared the first reviewable collections for all three RQ1 datasets. These are
candidates, not frozen evaluation sets or new inference results.

| Dataset | Questions / images | Input-only outcomes in selected candidates |
|---|---:|---|
| GQA development | 60 / 20 | 31 preserved correct, 8 losses, 8 gains, 13 both wrong |
| MMStar | 30 / 30 | 4 preserved correct, 8 losses, 5 gains, 13 both wrong |
| TextVQA | 30 / 30 | Unmeasured; nested in fixed 100-question / 100-image screening pool |

GQA uses 10 metadata-selected and 10 outcome-selected images, with three distinct
questions each. All selected diagnostic input-only losses/gains have a selected
preserved-correct question on the same image. The gallery also includes 100
unselected same-image questions as context only. All questions from deferred
airplane image `n497658` are excluded. Reserved GQA evaluation remains untouched.

MMStar has five questions per category. Its math gain slot uses a documented
preserved-correct fallback because the saved math run has no input-only gains.
These enriched selections cannot be used as population accuracy estimates.

TextVQA uses pinned v0.5.1 validation questions and 100 individually downloaded
images. Downloaded source files total about 30.1 MiB. All ten answers, original
image bytes, dimensions and content hashes are retained. No OCR transcript or
model prediction was added; baseline screening remains a separate next step.

Artifacts: `research/results/rq1-candidates-v1-s590/index.html` and the adjacent
portable ZIP (~70.6 MiB, 168 files). Galleries embed images, display saved raw
answers and permit separate review-decision JSON exports tied to manifest hashes.
See [instructions](../RQ1_CANDIDATES.md). Artifacts are ignored by Git.

Verification: four CPU preparation tests passed, including real-archive checks;
collection/image/source hashes, exact TextVQA pool nesting, ZIP CRC/content equality
and all eight native ApET source hashes passed. No GPU inference, dependency
changes or native compression edits occurred. Browser preview was unavailable,
so interactive review/export behavior remains to be checked when opened locally.

Three image/question pairs were visually spot-checked. GQA's polished-table
example and TextVQA's printed word “child” match visible evidence. MMStar case 77
deserves wording/label review: the image shows soccer, while option A distinguishes
football and soccer players and option C also broadly describes the scene. The
saved gold option and scores are unchanged; this is a review observation, not a
rescored result. Baseline correctness alone does not guarantee an unambiguous case.

Next: review all three candidate galleries and export decisions; screen TextVQA,
then freeze reviewed IDs and implement a small intervention pilot across all
three datasets. Detailed numerical investigation remains deferred.
