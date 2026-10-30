# Midterm presentation — revised script v3

Matches `PengChen_midterm_slides_v3.pptx`. The same narration is embedded in the PowerPoint speaker notes. Read the paragraphs; headings are slide cues.

**658 spoken words**; approximately 5.4–6.3 minutes at 110–130 words per minute with brief pauses. Rehearse once with a timer; allow about two minutes for slide 3.

## Slide 1 — Motivation

Hello, I’m Peng Chen. My project asks when and why ApET preserves a vision-language model’s answers.

Visual tokens make inference expensive. ApET keeps features that are difficult to reconstruct from a small basis, and combines discarded features with similar retained ones.

The paper reports about ninety-five percent of its average benchmark performance at a much smaller token budget. The speedup shown here also comes from the paper, not my Kaggle measurements.

My goal is to understand how this geometric criterion relates to preserving information needed for actual answers.

## Slide 2 — Method and research questions

I use LLaVA one point five, with seven billion parameters, and the native ApET implementation.

The model starts with five hundred and seventy-six visual tokens. ApET compresses before the language model and again after block sixteen. Our two schedules average sixty-four or one hundred and ninety-two tokens across the decoder layers. Reconstruction scores selection; compression itself retains and merges features.

Experiments run on two Kaggle T4 GPUs with a locked environment. The larger comparisons cover GQA, MMStar and TextVQA.

RQ1 compares reconstruction-error rankings with measured answer sensitivity. RQ2 asks how selection and merging contribute to preservation. RQ3 considers a lightweight improvement only if the evidence supports it.

The small RQ1 study is largely complete. Component attribution remains the main open question.

## Slide 3 — Four distinct findings

There are four findings.

First, reconstruction error does not consistently agree with measured answer sensitivity. On thirty selected questions, we compared E rankings with an IH-inspired retain-one ranking. Median correlations were slightly negative, although individual questions differed. This scores the first reference-answer token, not true token importance.

Second, retained tokens receive very different amounts of merging. In the earlier input-only panel, the lowest-E quarter of ordinary retained targets received about sixty-six percent of incoming merges. The highest-E quarter received about two percent.

This may be expected: discarded low-error features may resemble these targets. It also explains why protecting the highest-E targets barely changed the computation. Some retained states may preserve distinctive features, while others collect discarded information. Whether this division helps answers still needs investigation.

Third, ApET’s advantage over random selection depends on the task. At the same average budget of sixty-four tokens, the gap is about twenty-one score points on TextVQA, two and a half on GQA, and unclear on MMStar. TextVQA uses a consensus score; the other two use accuracy.

Fourth, similar overall accuracy can hide different answers. At an average budget of one hundred and ninety-two, GQA loses thirty-four previously correct answers and gains twenty-six. On MMStar, it loses thirty-six and gains forty, while accuracy changes very little.

So overall performance and answer preservation are separate outcomes. Both matter when we explain why compression works.

## Slide 4 — Roadblocks and fixes

There were three main roadblocks.

First, Kaggle package updates caused compatibility problems. The locked environment now makes the setup reproducible.

Second, some fresh compressed runs changed answers, while replaying saved features gave identical outputs. One diagnostic found differences after merging. Floating-point accumulation is a possible cause, but the exact cause is unconfirmed. I keep first captures and repeat critical cases.

Third, the diagnostic panel includes known ApET failures. It supports case analysis, while broader outcome-unselected cohorts are needed for method comparisons.

## Slide 5 — Next steps

The next steps are to explain these observations before choosing an improvement.

First, separate selection from merging on a cohort not selected from ApET’s wins or losses. Compare E-based and random selection with basis handling and the merge rule fixed. Compare merging with dropping under the same selection.

Second, investigate the different roles of retained tokens. Study merge loads and feature changes alongside answer turnover, across stages and tasks. This can test whether collecting discarded information helps some answers while disturbing others.

Third, RQ3 remains exploratory. We have not chosen a fixed improvement direction. If the analysis identifies a consistent problem, we can evaluate one modest change.

The main deliverable is one controlled comparison on at least one hundred outcome-unselected questions, with matched budgets, random seeds where relevant, and paired uncertainty estimates. A clear negative result also counts. Thank you.

---

## Changes and recording cues — not spoken

- Slide 3 has four separate findings: E–IH ranking mismatch; unequal merge routing; task-dependent native-versus-random performance; answer turnover.
- Finding 3 uses one fixed schedule, 96→32 (average 64). It makes no budget-dependence claim. TextVQA consensus points and GQA/MMStar accuracy points are different metrics; do not pool them. MMStar’s near-zero gap is uncertain.
- Finding 4 compares native ApET with the uncompressed model at 288→96 (average 192). Loss means baseline-correct → wrong; gain means baseline-wrong → correct under the frozen scorer.
- The routing percentages remain scoped to the earlier selected 24-image input-only288 panel. Retained-token roles are a hypothesis, not proof of a flaw.
- Native ApET uses linear reconstruction error to score selection. It retains and merges features; it does not feed reconstructed features into downstream layers. A reconstruction-replacement experiment would be our own diagnostic.
- The presentation no longer commits to the reconstruction-safety test. Small merge/linear-perturbation checks remain possible internal diagnostics; RQ3 has no fixed direction yet.
- IH-inspired sensitivity is specific to the intervention and first reference-answer tokenizer piece; it is not true token importance.
- Published speedup on slide 1 comes from the paper, not our Kaggle measurements.
