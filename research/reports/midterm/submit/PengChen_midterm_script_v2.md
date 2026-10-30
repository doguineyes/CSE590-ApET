# Midterm presentation — revised script v2

Matches `PengChen_midterm_slides_v2.pptx`. Read the paragraphs; headings are slide cues. The same narration is embedded in the PowerPoint speaker notes.

**655 spoken words**; approximately 5.4–6.3 minutes at 110–130 words per minute, with short pauses. Rehearse once with a timer. Give slide 3 the most time.

## Slide 1 — Motivation

Hello, I’m Peng Chen. My project asks when and why ApET preserves a vision-language model’s answers.

Visual tokens make inference expensive. ApET keeps features that are difficult to reconstruct from a small basis, and combines discarded features with similar retained ones.

The paper reports about ninety-five percent of its average benchmark performance at a much smaller token budget. The speedup shown here also comes from the paper, not my Kaggle measurements.

My goal is to understand how this geometric criterion relates to preserving information needed for actual answers.

## Slide 2 — Method and research questions

I use LLaVA one point five, with seven billion parameters, and the native ApET implementation.

The model starts with five hundred and seventy-six visual tokens. ApET compresses before the language model and again after block sixteen. Our two schedules average sixty-four or one hundred and ninety-two tokens across the decoder layers.

Experiments run on two Kaggle T4 GPUs with a locked environment. The larger comparisons cover GQA, MMStar and TextVQA.

RQ1 compares reconstruction-error rankings with measured answer sensitivity. RQ2 asks how selection and merging contribute to preservation. RQ3 considers a lightweight improvement only if the evidence supports it.

The small RQ1 study is largely complete. Component attribution remains the main open question.

## Slide 3 — Three connected findings

There are three findings.

First, reconstruction error does not consistently agree with measured answer sensitivity. On thirty selected questions, we compared E rankings with an IH-inspired retain-one ranking. Median correlations were slightly negative, although individual questions differed. This scores the first reference-answer token, not true token importance.

Second, retained tokens receive very different amounts of merging. In the earlier input-only panel, the lowest-E quarter of ordinary retained targets received about sixty-six percent of incoming merges. The highest-E quarter received about two percent.

This may be expected: discarded low-error features may resemble these targets. It also explains why protecting the highest-E targets barely changed the computation: few sources were merging into them. Some retained states may preserve distinctive features, while others collect discarded information. Whether this division helps answers still needs investigation.

Third, complete ApET performs better than random in some settings, but the advantage depends on task and budget. On TextVQA, the gap is about twenty-one points at the smaller budget and nine at the larger budget. It is smaller on GQA and unclear on MMStar.

Average scores also hide answer turnover. At the larger budget, GQA loses thirty-four previously correct answers and gains twenty-six. Accuracy drops by only one point.

These findings connect reconstruction, aggregation, and answer preservation. They also suggest specific next experiments.

## Slide 4 — Roadblocks and fixes

There were three main roadblocks.

First, Kaggle package updates caused compatibility problems. The locked environment now makes the setup reproducible.

Second, some fresh compressed runs changed answers, while replaying saved features gave identical outputs. One diagnostic found differences after merging. Floating-point accumulation is a possible cause, but the exact cause is unconfirmed. I keep first captures and repeat critical cases.

Third, the diagnostic panel includes known ApET failures. It supports case analysis, while broader outcome-unselected cohorts are needed for method comparisons.

## Slide 5 — Next steps

The next steps follow directly from the findings.

First, separate selection from merging on a cohort not selected from ApET’s wins or losses. Compare E-based and random selection with basis handling and the merge rule fixed. Compare merging with dropping under the same selection.

Second, investigate the different roles of retained tokens. Check merge loads across stages and budgets, measure actual feature changes, and test whether changing the strength of merging affects answers.

Third, test reconstruction more directly. Replace a few features with their linear approximations while keeping sequence length unchanged. Control the size of the feature change, so a smaller perturbation is not mistaken for a special reconstruction effect.

RQ3 stays conditional. If these tests identify a consistent problem, we can evaluate one small modification.

The main deliverable is one controlled comparison on at least one hundred outcome-unselected questions, with matched budgets, random seeds where relevant, and paired uncertainty estimates. A clear negative result also counts. Thank you.

---

## Alignment and recording cues — not spoken

- Slide 3 follows the agreed order: ranking mismatch → uneven merge roles → task/budget effects and answer turnover.
- Slide 5 mirrors those observations: component comparison → retained-token roles → reconstruction replacement; RQ3 is conditional.
- On slide 3, 66.1% and 1.8% come from the earlier selected 24-image input-only288 panel. “Lower E” means within ordinary retained targets, not all image tokens. The division-of-roles explanation is a hypothesis.
- IH-inspired sensitivity scores the first reference-answer token; it is not whole-answer accuracy or ground-truth importance.
- The unselected cohort addresses bias in the earlier component comparison; it does not retroactively change the RQ1 cohort.
- Published speedup on slide 1 is from a single-A100 paper benchmark. Our project has no controlled speedup estimate.
- Pause between the three columns on slide 3. Do not read every table cell on slide 4 or every milestone date on slide 5.
