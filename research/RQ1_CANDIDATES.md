# RQ1 candidate preparation

This is the first dataset-preparation step for **GQA, MMStar and TextVQA**.
It copies original image bytes and saved predictions into reviewable collections.
It does not run inference, change native ApET, freeze an evaluation set or perform
answer-contribution interventions. Numerical debugging remains deferred.

## Open and review

The prepared collection is under `research/results/rq1-candidates-v1-s590/`.
Open `index.html` in a browser, then open each dataset gallery. Images are embedded
in the galleries, so they work offline. For each candidate, choose Keep, Exclude,
Ambiguous, Possible label issue or Uncertain and record useful observations.
Click **Download review decisions** before closing. Browser storage is convenient
but the downloaded JSON is the portable record.

Review overlays identify the exact manifest SHA256 and case IDs. They do not
rewrite labels, saved predictions or original evaluator scores. Keep the three
review JSON files for the next step; we will reconcile them before freezing IDs.
Other GQA questions under the expandable context section are not selected cases.

## Candidate composition

| Dataset | Candidates | Selection |
|---|---:|---|
| GQA development | 60 questions / 20 images | 10 images selected without correctness, 10 diagnostic images; three distinct questions per image |
| MMStar | 30 questions / 30 images | Five per category: three selected without correctness and two diagnostic slots |
| TextVQA validation | 30 questions / 30 images | First 30 of a fixed 100-image screening pool, selected without model outcomes |

All three datasets belong to RQ1. These are starting candidates, not the final
case counts. GQA diagnostic selection requests mixed loss/gain/preserved images
and same-image controls, using **input-only** outcomes first. Decoder-only and
combined predictions and configuration-specific preserved controls are retained.
The entire deferred airplane image `n497658` is excluded, including context.
Reserved GQA evaluation is never read.

MMStar requests a loss and a gain per category. A quota that cannot be met gets an
explicit recorded fallback; the saved math category has no input-only gain and
therefore uses a preserved-correct example. Scores are preserved from the saved
project evaluators: GQA short-answer scoring and MMStar explicit-choice scoring.
They are observations, not human judgments of answer quality.

The metadata-reference stratum is selected without correctness; the diagnostic
stratum uses saved outcomes. Report their results separately. This mixture is
for mechanism discovery: its aggregate accuracy, preservation rate or correlation
must not be reported as an unbiased dataset-wide estimate. Selection is based
on seeded SHA256 ranks and is independent of input row ordering.

## TextVQA preparation and later screening

We have no TextVQA model results yet. Every prediction is `null`, every outcome
is `unmeasured`, and baseline screening is pending. Review whether the question
requires visible text, whether the answer is legible at model resolution and
whether the references are clear. Do not infer model competence from the gallery.

The fixed pool contains 100 distinct images and one question per image, including
the 30 provisional candidates. The next inference pilot will screen these inputs;
baseline-correct cases will define the preservation denominator. Retain baseline-
wrong cases for context and gain analysis. Any subsequent selection from that pool
must record its rule and counts before interventions.

All ten reference answers are retained. The provisional teacher-forcing target is
the most frequent **raw** reference answer, with a lexicographic tie-break, fixed
before inference. Free-generation evaluation must use TextVQA's multiple-answer
VQA consensus scorer (`llava/eval/m4c_evaluator.py`), not GQA exact matching.
No auxiliary OCR transcript is added to model input.

Questions use the official [TextVQA v0.5.1 validation annotations](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json),
with their SHA256 recorded and checked. Individual images come from the pinned
[TextVQA validation mirror](https://huggingface.co/datasets/maoxx241/textvqa_val/tree/ddf4bd3dab5428e3ecbb4c0fc88e1c5bcfaf59cd).
The revision, mirror content hashes, image SHA256, decoded RGB hashes and
dimensions are recorded. The downloader fetches only the selected 100 JPEGs and
annotations; the default total source-file budget is 100 MiB.

## Rebuild on the Mac

From the repository root, using the existing editor environment:

```bash
environments/editor/.venv/bin/python scripts/prepare_rq1_candidates.py
```

The first run needs Internet for the small TextVQA download. Subsequent rebuilds
reuse verified cached files under `/tmp/apet-rq1-textvqa-cache`. No PyTorch, model
weights, Kaggle kernel changes or GPU are required. Outputs already present are
protected: choose a new path when rebuilding, for example:

```bash
environments/editor/.venv/bin/python scripts/prepare_rq1_candidates.py \
  --output-dir research/results/rq1-candidates-v1-review2
```

The two saved native-reference result ZIPs are required locally; pass
`--gqa-archive` and `--mmstar-archive` if they were moved. Source ZIPs are read-only.
The script verifies archive CRCs, paired records, native reference source hashes,
image bytes and recorded prediction/manifest hashes where available. The original
MMStar report did not record a predictions hash; preparation computes it and
explicitly marks that no earlier prediction-hash verification was available.

The resulting directory and adjacent ZIP are ignored by Git. The ZIP includes
all images, manifests, offline galleries, original reports and preparation source
snapshots; it excludes model weights, environments and large source archives.
`collection.json` records manifest/source hashes and preparer versions. Attach
this ZIP to Kaggle later when an RQ1 runner is ready. The candidate schema is new;
existing GQA baseline runners do not accept it automatically.

## After review

Freeze reviewed IDs, targets and exclusion reasons. Then implement a small RQ1
intervention pilot on 3–5 questions **from each dataset**, with a no-op control,
reference repeats and a runtime estimate. Expand only after those checks work;
input-stage and selected decoder-layer measurements need explicit token mappings.
This preparation step does not implement that instrumentation.
