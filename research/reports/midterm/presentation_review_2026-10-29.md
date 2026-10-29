# Review of the submitted midterm report and five-slide deck

Reviewed October 29, 2026. Sources are the actual files in `submit/`: `PengChen_midterm_report_v8-2.pdf`, `PengChen_midterm_slides.pdf`, and `ApET Midterm — 5 Slides.pptx`. All four report pages and five PDF slides were rendered and visually inspected; PowerPoint text and existing speaker notes were read separately. Submitted files were not edited.

## Overall advice

Keep the five-slide structure. It covers motivation, method/research questions, preliminary results, roadblocks, and next steps. The report is four pages including references, and the slides have no visible overlap or clipping. The central result numbers agree with the experiment records. No further GPU run is needed to prepare this presentation.

The main improvement is to use precise, short claims and give slide 3 enough speaking time. It contains the evidence for most of the presentation. Slide 4 contains the densest prose and is the first place to shorten text for a video.

## Suggested changes, slide by slide

### Slide 1

- Keep the paper-versus-project distinction explicit when speaking. The paper's Table 1 reports 95.2% average relative performance for ApET versus 92.7% for VisionZip. Its Table 7 reports 1.46× total inference speedup on LLaVA-1.5-7B, measured on POPE with a single A100. This is not a speed measurement on our two T4s.
- Replace “With 64 of 576 tokens” with **“At an average budget of 64 visual tokens.”** Stage 1 keeps 96 and stage 2 keeps 32 in our matched schedule.
- “Without attention” is easily misread as saying that the model does not use attention. Prefer **“Without using attention scores for selection.”**
- If revising the method sentence, say discarded states are averaged into similar ordinary retained states, with basis representations protected. The current sentence is an acceptable overview but omits that native detail.

### Slide 2

- Change **“Information Horizon (IH) test”** to **“IH-inspired retain-one test.”** Our one-position measurement is an adaptation, not a reproduction of the original benchmark.
- The brief spoken definition should mention the first reference-answer token and the all-visual-removed comparison. No equation is necessary.
- “Planned” under RQ2 can make existing work look unperformed. Prefer **“Component ablations piloted; broader matched comparisons next.”** We already ran the selected-panel selection/merge and routing/protection studies; their population-level interpretation is limited by selection.
- Explain that pure random pruning removes E-ranking, basis handling and merging together. The larger result therefore compares complete procedures, not E-ranking alone.

### Slide 3

- Replace **“E does not predict which tokens matter”** with **“E does not consistently track answer sensitivity.”** The latter matches the report and measurement limits.
- Replace the plot description with: **“Retain one visual state; score the first reference-answer token relative to removing all visual states.”**
- The numerical headline is correct: the E–retention medians are −0.068 / −0.121 / −0.136 for GQA/MMStar/TextVQA, and 11 of 30 question-level correlations are positive. These are selected questions, not 17,280 independent samples or evidence of a population-wide negative relationship.
- The middle column should say **“Score (%)”**, because TextVQA uses partial-credit consensus rather than binary accuracy. The displayed scores and differences are correct.
- Read one comparison at a time. Point to TextVQA first, then GQA. Mention the lack of a clear MMStar difference verbally so the audience does not infer a universal advantage from the two plotted datasets.
- The right-hand counts and accuracy changes are correct. Keep “lost” and “gained” relative to the uncompressed model explicit. A gain is a scoring outcome, not automatic proof of improved reasoning.
- The slide title could become **“Compression performance and answer preservation differ.”** This avoids reading “not answer by answer” as a claim that ApET never preserves individual answers.

### Slide 4

The table is accurate in its overall status but has substantial text in approximately 14-point body type. Shorter cells would be easier to follow in a recorded video. Suggested condensed wording:

| Issue | Observation | Action / status |
|---|---|---|
| Compatibility | Kaggle packages drift | Locked environment; resolved |
| Numerical variation | Fresh encodes vary; fixed-feature replays agree | Keep first captures; repeat critical cases; cause unconfirmed |
| Selection bias | Diagnostic cases include known native losses | Use broader, outcome-unselected cohorts for method ranking |
| Attribution | Selection/basis/merge effects remain entangled | Choose one controlled comparison; RQ3 stays conditional |

If editing, trim words before enlarging body type toward 16–18 points. Keep the technical `scatter_add_` explanation for the report or a question; it need not be read aloud.

The archived one-question diagnostic found identical pre-merge features and assignments but changed post-merge features; no-merge repeats were exact. It localizes variation for that input but does not isolate a specific CUDA operation or explain every changed answer, including the deferred GQA case. The current “cause not isolated” label is appropriate; preserve it in narration.

### Slide 5

- Keep one clear main deliverable: **one controlled comparison on at least 100 outcome-unselected questions**. Avoid implying that every proposed variant will be completed.
- Split “vary only selection, merging fixed” from the keep/merge/drop/reconstruct idea. The former is a selection ablation; the latter changes how a feature is represented or intervened on. They need separate controls.
- A concise replacement is: **“Selection: compare E and random with basis/merge rule fixed. Representation diagnostic: keep, merge, drop or reconstruct selected states.”**
- “Unselected” means not chosen because ApET won or lost, not absence of all dataset sampling. Choose the pool and analysis before running the comparison.
- Keep the RQ3 decision conditional and the negative-result success criterion. This is appropriate for a midterm presentation.

## Report checks and small wording polishes

The report's six main score rows, random SDs, saved intervals, correlations, panel composition and loss/gain counts agree with the archived summaries. The numerical-variation paragraph appropriately labels the root cause as a hypothesis. The model/environment, stage counts and cohort sizes are consistent with the protocols.

Three small edits would improve precision:

1. In Table 1, **“share kept”** means score divided by uncompressed score, not APR. Rename it **“% of baseline score.”**
2. In Section 4.6, the saved-data probe tested an E-based loss predictor, not an adaptive-budget policy. Say **“The tested E summaries do not yet justify an image-only adaptive budget.”** “E features did not predict failures” should likewise become “did not provide a reliable cross-dataset predictor.”
3. In Section 6, remove **“only”** from the claim that a discarded token's information arrives only through a merged average. ApET does not explicitly reconstruct that token downstream, but basis and other retained features can still represent correlated information. That distinction is part of the open mechanism question.

A minor layout polish: page 3 starts after Table 1 with the sentence fragment “is small.” Keeping that sentence together would read more smoothly. Figures 2–3 are small in print, so retain their explanatory captions. These do not require a redesign of the report.

## Evidence used

- `supporting_data_2026-10-25/rq1_correlations_30.csv` and `table2_preservation.csv`.
- `research/notes/2026-10-18-rq1-input-ih-full-review.md`.
- Staged GQA800, MMStar1500 and TextVQA100 review summaries from October 21.
- `research/notes/2026-10-19-rq2-native-parity-diagnostic-review.md`.
- Native merging code in `llava/model/llava_arch.py` and the frozen experiment protocols.
- The local ApET paper, especially Tables 1 and 7.

The accompanying narration deliberately avoids treating sensitivity as true importance, claiming our own speedup, proving a numerical root cause, or announcing a completed RQ3 improvement.
