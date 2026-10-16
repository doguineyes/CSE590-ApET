# RQ1 dataset preparation completed — 2026-10-15

Today we incorporated the GQA and MMStar human reviews, added a bounded second
round with assistant image review, and agreed to retain every prepared TextVQA
question. Dataset curation is complete for now. Next work moves to RQ1 measurement
implementation and a small Kaggle pilot, rather than further candidate expansion.
All three datasets remain part of RQ1.

Research question: **How does reconstruction-based visual redundancy relate to
task-relevant answer contribution, and how does this relationship change across
task types and model layers?**

## Working datasets

| Dataset | Working scope | Role and status |
|---|---:|---|
| GQA | 49 questions / 27 images | Main mechanism analysis; human review plus assistant additions |
| MMStar primary | 18 questions / 18 images | Baseline-correct cross-task stress subset, covering all six categories |
| MMStar qualitative appendix | 10 questions / 10 images | Five gains and five both-wrong human Keep cases; outside the primary subset |
| TextVQA | All 100 prepared questions / 100 images | Fine-detail/text-reading study; untested; no questions excluded |

TextVQA's earlier 30-question candidate gallery is **nested within the same
100-question pool**, not an additional dataset or a limit on retained questions.
The user's decision is to keep all prepared questions until we have measurements;
this does not mean running the entire 5,000-question validation set. No further
human selection round is requested. Later reporting can separate baseline-correct
and baseline-wrong cases while preserving all 100 inputs and results. Preservation
metrics must use the appropriate baseline-correct denominator; untested questions
are not counted as correct or incorrect.

These are the accepted working inputs for implementation. Existing manifests
remain reviewed proposals/candidates; the intervention protocol and final
experiment registration have not yet been frozen. No RQ1 contribution measurements
have been run.

## What today's reviews established

GQA's first human review kept 30 of 60 questions. A stricter mechanism rubric
adopted 22 and set eight aside without changing the user's decisions. Assistant
inspection of a second round adopted 27 additions. The resulting 49 questions
contain 38 input-only preserved-correct cases, six losses, two gains and three
both-wrong cases. Seventeen images have multiple retained questions (39 questions);
ten have a single question.

New GQA losses include television/phone recognition and lamp, refrigerator and
medicine-cabinet positions. The coat/bin-color gain `20596219` remains tentative;
cabinet loss `201570615` has no retained same-image preserved control. Broad
absence, coarse scene and paraphrase controls are tagged for sensitivity checks.
The entire airplane image `n497658` remains excluded and its numerical investigation
is deferred. Reserved GQA evaluation remains untouched.

MMStar's human review kept 17 of 30 questions, but only seven were baseline-correct
(five losses, two preserved). Eleven assistant-reviewed additions give 18 primary
questions: six losses and 12 preserved. The other ten original Keeps remain in
the qualitative appendix. This was a small extension, not one-for-one replacement
of every excluded question. The full native baseline is 521/1,500 = 34.73%; source
questions also vary in effective option count and show answer-A bias.

The methodological lesson is to distinguish question/label suitability from
model competence. Both datasets contain wording or annotation ambiguities;
baseline-wrong cases also mix perception, reasoning, knowledge and OCR limits.
A baseline-correct multiple-choice answer is useful screening evidence, not proof
that the model grounded its answer in the image. Saved gains and losses are
observations to investigate, not causal findings about reconstruction error.

Original human reviews, raw answers, scores, labels and native ApET sources are
preserved. Assistant decisions are separately identified, include uncertainty,
and are not independent human validation. These outcome-enriched subsets support
mechanism analysis, not unbiased population accuracy or correlation estimates.

## Saved evidence

- [GQA reviewed selection](../results/rq1-gqa-reviewed-v1-s591/index.html), its
  [complete audit](../results/rq1-gqa-reviewed-v1-s591/review-audit.html), and
  [selection rules](../GQA_RQ1_ROUND2.md).
- [MMStar primary selection](../results/rq1-mmstar-reviewed-v1-s592/index.html),
  [qualitative appendix](../results/rq1-mmstar-reviewed-v1-s592/qualitative-gallery.html),
  and [selection rules](../MMSTAR_RQ1_ROUND2.md).
- [All 100 prepared TextVQA questions](../results/rq1-candidates-v1-s590/textvqa/screening-gallery.html)
  and their [manifest](../results/rq1-candidates-v1-s590/textvqa/screening-manifest.json).
- Detailed notes: [GQA](2026-10-15-gqa-rq1-round2.md) and
  [MMStar](2026-10-15-mmstar-rq1-round2.md).

The adjacent result ZIPs preserve images, manifests, source/review hashes,
galleries and preparation code. Results are ignored by Git; keep the ZIPs
separately from the source-code repository.

Working manifest SHA256 values (no manifests were rewritten for this summary):

```text
GQA recommended:       07a6d8f449d1b3081f34b9568828712178085c16923dc7a9a72832b90feb5111
MMStar primary:        1c01e44702cac2cda7f3f0fd5f10b54de899caeb498af5389fc95d44d302d5b5
MMStar qualitative:    63a6479dc86828b2cad424f23fe5109bea555b996f2d9cdd953163ed1583edf5
TextVQA all 100:       d01ad107653ba9d0b215d927d35037464c39f1309756fd27329a82649812ab0f
TextVQA nested 30:     66c4a0c918a93371111bf4a07179d05e3c26be81950c541992aae7359bd7bc9f
```

The latest curation validation passed nine CPU tests, including source/review
identity, unchanged predictions, selection rules and the shared read-only
renderer. Image hashes, gallery links and ZIP contents were also checked.
This notes update verifies current manifest counts and that TextVQA predictions
remain unmeasured. No new GPU inference or environment work occurred.

## Next session: start the RQ1 pilot

1. Establish TextVQA model results using the existing original-ApET Kaggle
   environment. Check a few inputs and the multi-answer VQA consensus scorer,
   then record baseline and compressed predictions for the retained 100-question
   pool. Preserve all ten reference answers and all raw responses; do not reuse
   GQA exact matching or add OCR transcripts to model input.
2. Define and implement the contribution proxy in project-owned instrumentation.
   Start with change in mean gold-answer token log probability under a controlled
   visual-token-group perturbation. Fix the canonical target before interventions;
   use teacher forcing only for measurement, not in scored generation prompts.
3. Pilot on roughly 3–5 questions from **each** dataset, using paired GQA questions
   where possible. Begin at the input stage with a few equal-sized groups spanning
   low/middle/high reconstruction error and a seeded matched-size random control.
   Record the exact representation perturbed and preserve sequence shape/order.
   Use no-op and repeated unperturbed references to quantify noise.
4. Match native reconstruction errors to the corresponding full-context visual
   tokens and exact compression seed. The uncompressed path does not automatically
   produce those errors. Use the same image-derived groups across GQA questions;
   do not infer question effects from different random compression realizations.
5. Inspect behavior and measure runtime before expanding. Then add a limited
   decoder-layer study with explicit token mappings and controlled upstream
   context. Existing input/decoder baselines use different budgets and do not
   themselves isolate a layer effect.

For every pilot run preserve manifest/version hashes, code commit and source
hashes, dependency lock, model/processor revisions, prompts, target/scorer rules,
seeds, configuration, stage/layer, token-group membership, reconstruction errors,
unperturbed/no-op/perturbed answer scores, raw predictions, timing and memory.
Keep image grouping and selection/review strata in analysis; token groups and
questions sharing an image are not independent images.

Continue with native original ApET and the existing Kaggle environment. A new
compression rewrite, another full GQA/MMStar baseline run and detailed airplane
numerical debugging are outside the immediate next step.
