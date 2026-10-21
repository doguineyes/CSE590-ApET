# Midterm draft: evidence and submission checklist

Prepared October 21, 2026. Start with [the report draft](midterm_draft.md).
The report closes the **current experimental phase**, not every causal question
in the proposal. No new inference was run to prepare these materials.

## Requirements and scope

- **User-confirmed requirement:** 3–4 pages, preliminary experiments, results and
  roadblocks; a short in-class presentation accompanies the report. No special
  template or citation style has been supplied as of October 21.
- The current Fall 2026 course syllabus repeats this requirement and lists the
  midterm report beside October 27. The schedule is tentative; check any later
  instructor announcement before submission. The 6–8-page requirement concerns
  the final report, not this deliverable.
- Title, author, email and the three research questions follow the
  [submitted proposal](../Peng_Chen_CSE590_ApET_Project_Proposal_v2.pdf).
- RQ1: completed for the specified diagnostic cohort and two boundaries. RQ2:
  completed input-stage component diagnostics and broad staged native/random
  comparisons; **staged selection-versus-merging attribution remains partial**.
  RQ3: not implemented or evaluated.
- The supplied AI handoff and discussion were treated as planning references.
  Their suggestions are not new experimental results. The draft follows saved
  evidence where wording needs qualification, especially operator identity,
  depth-score definitions and whole-procedure versus component comparisons.

## Numeric claims in the draft

All paths below are relative to the repository root. Detailed raw data under
`research/results/` are intentionally ignored by Git. The compact
[materials snapshot](materials/verified_summary.json) and two CSV tables can
travel with the report; they include source hashes and the preparation commit.

| Claim | Exact reviewed data | Tracked interpretation / limits |
|---|---|---|
| RQ1: 30 questions, 24 images; GQA12/MMStar12/TextVQA6; 576 positions per question | `research/results/rq1-input-ih-synthesis-v1-2026-10-18/question-synthesis.csv` | [RQ1 consolidation](../../notes/2026-10-18-rq1-consolidation-and-scope.md); 17,280 positions **per complete input operator**, not independent questions |
| Table 1: all nine median correlations | Same `question-synthesis.csv`, independently reduced and checked against `cohort-summary.json` | [Full input review](../../notes/2026-10-18-rq1-input-ih-full-review.md); [compact table](materials/rq1_correlations.csv) |
| Individual E–retention range −0.486 to +0.404; GQA photo-only/nonbasis median +0.087 | Same CSV: `rho_E_ih_first`, `photo_nonbasis_rho_E_ih_first` | Primary table includes all positions; filtering is a sensitivity analysis, not a changed primary cohort |
| Six same-image pairs; between-question retention rho −0.443 to +0.655 | `research/results/rq1-input-ih-synthesis-v1-2026-10-18/same-image-pairs.csv` | Exact input-feature/E match for each pair; question, target and confidence all change |
| Six matched depth questions; input/layer retention rho −0.038 to +0.198 | `research/results/rq1-input-ih-synthesis-v1-2026-10-18/depth-comparison.csv` | [Layer review](../../notes/2026-10-17-rq1-layer-ih-full-review.md); first-piece retention score, two boundaries only |
| Dominion best retained first-piece probability 20.2% / 98.1% | Same `depth-comparison.csv`, row `textvqa:35670` | Post-hoc maxima over 576 contextual states; not full-word generation accuracy |
| Table 2: six rows of scores, paired differences and intervals | `research/results/rq2-staged-mmstar1500-review-v1-2026-10-21/cross-dataset-staged-summary.json` | [TextVQA review](../../notes/2026-10-21-rq2-staged-textvqa100-review.md), [GQA review](../../notes/2026-10-21-rq2-staged-gqa800-review.md), [MMStar review](../../notes/2026-10-21-rq2-staged-mmstar1500-review.md); [compact table](materials/rq2_staged_results.csv) |
| RQ2: 2,400 question evaluations / 21,600 primary predictions; nine arms | Same cross-dataset JSON: `question_evaluations`, `primary_predictions`; 100 + 800 + 1,500 questions | Controls/repeats are outside primary denominators; seeds are not extra independent questions |
| MMStar native 288→96 loses 36, gains 40; APR 485/521 = 93.09%; severe APR 442/521 = 84.84% | Cross-dataset JSON and `condition-summary.csv` in the same review directory | Accuracy can mask turnover. APR is conditional on baseline correctness |
| Input-only, selected 30-case panel: native merge 19/30, same selection/no-merge 21/30 | `research/results/rq2-crossed-full-v1-review-2026-10-19/summaries.json` and `paired-selection-outcomes.csv` | [Crossed review](../../notes/2026-10-19-rq2-crossed-full-review.md); failure-enriched and baseline-correct, not broad evaluation |
| High-E target protection barely changes routing | `research/results/rq2-routing-full-v1-review-2026-10-19/routing-activity.csv` and `native-rank-activity.json` | [Routing review](../../notes/2026-10-19-rq2-routing-full-review.md): only 21/6,912 incoming assignments rerouted; a largely inactive test |
| Brolio → Bottoli with input-only merge; no-merge and target317 restoration recover Brolio; sham/control141 do not | `research/results/rq2-crossed-oracle-v1-review-2026-10-19/answers.csv` and `oracle-summary.json` | [Oracle review](../../notes/2026-10-19-rq2-brolio-oracle-review.md); fixed native tuple, post-hoc selected target, conditional rescue only |

The build script checks the RQ1 reductions, counts, ranges, pair/depth identities
and RQ2 summary arithmetic. It relies on the prior archive audits rather than
rehashing the large raw ZIPs again. Component/oracle claims were checked against
their saved review records separately.

## Definitions that must survive editing

1. **Two answer-response operators.** Zeroing is full-context vector replacement;
   retention is physical singleton retention versus visual removal. Sequence
   geometry, null and competing context differ. Our retention is an IH-inspired
   1×1 adaptation, not an exact reproduction of the released default 2×2 study.
2. **Score units.** Primary matched comparisons use a frozen gold first-tokenizer-
   piece probability difference. Earlier whole-answer mean-log-score results
   remain auxiliary. In particular, the handoff's input/layer **zeroing** map
   agreement range used historical whole-answer scores; it is not presented as
   a matched first-piece range in this draft.
3. **Visual E and causal order.** The image precedes question text. Later visual
   positions cannot attend to future question tokens. Late text positions have
   already seen the image, so late visual removal is not a true question-only
   baseline.
4. **Native versus project code.** Native compression stays distinct from
   instrumentation, scorers and registered compatibility fixes. Do not call the
   entire fork an untouched official repository or claim paper-level numerical
   reproduction. Main runtime: Python 3.10.18 / Torch 2.1.2+cu121 / Transformers
   4.37.2, FP16 LLaVA-1.5-7B on two T4s. Checkpoint/runtime/operator identity was
   verified across the three broad runs.
5. **Two different random controls.** Early component tests held ten FPS basis
   representations and their native placement fixed while sampling ordinary
   retained positions. Broad staged **pure random** has no FPS, E, replacement
   or merging. Neither is the paper's basis-randomization ablation.
6. **Sample roles and scoring.** GQA800 is development data; reserved evaluation
   remains untouched. TextVQA100 is screening/development data, with six prior
   diagnostics included. MMStar1500 is the full pinned set, using the project
   scorer rather than a claim of official leaderboard reproduction. Do not pool
   their different metrics or describe all three as held-out benchmarks.
7. **Uncertainty.** Intervals use 2,000 paired bootstrap draws: TextVQA questions,
   GQA image groups, MMStar's 1,430 image-byte-hash groups. Unequal MMStar groups
   use question-weighted score/count ratios. Seeds are averaged inside questions.
   Intervals condition on the fixed cohorts, three seeds and captured native
   realizations. Zero-containing intervals are not equivalence tests.
8. **Bounded conclusions.** No universal importance measure, general depth law,
   percentage decomposition of selection/merge benefit, controlled speedup,
   unique causal text patch or demonstrated RQ3 improvement has been established.

## Raw archive entry points

Each of these has its matching `.log` archived under `research/results/`; the
linked tracked review notes document byte hashes and prior integrity checks.
Original archives and predictions were not changed for the report.

| Evidence | Canonical raw ZIP |
|---|---|
| RQ1 full input retention | `rq1-input-ih-thirtycases-v1-20261018T140840768523Z.zip` |
| RQ1 matched layer study | `rq1-layer16-ih-sixcases-v1-20261018T014830278963Z.zip` |
| Input selection/merge panel | `rq2-crossed-full-v1-20261019T234435011493Z.zip` |
| Routing panel | `rq2-routing-full-v1-20261020T004043438261Z.zip` |
| Brolio restoration | `rq2-crossed-oracle-v1-20261020T000115185532Z.zip` |
| Staged TextVQA100 | `rq2-staged-textvqa100-v1-20261021T134751714241Z.zip` |
| Staged GQA800 | `rq2-staged-gqa800-v1-20261021T142303483991Z.zip` |
| Staged MMStar1500 | `rq2-staged-mmstar1500-v1-20261021T153730115642Z.zip` |

## Rebuilding and preparing submission

From the project root, the existing editor interpreter is sufficient:

```sh
environments/editor/.venv/bin/python research/reports/midterm/build_materials.py
```

The script uses only the standard library. It reads local saved results, writes
three compact material files, and does not install dependencies or use a GPU.

- [x] Title, author and RQs checked against the submitted proposal.
- [x] Main numerical tables checked against saved data; references checked against
  original paper pages.
- [x] Completed work, limitations and future hypotheses separated.
- [ ] User review of the scientific narrative and any instructor updates.
- [ ] Typeset to 3–4 pages and inspect the final layout. **The Markdown draft is
  not a verified page count or submission-ready PDF.** Assume references count
  toward the limit unless the instructor says otherwise.
- [ ] Prepare the short presentation after choosing the final report narrative;
  a slide deck has not been created in this step.

The two tables are sufficient for the first layout. If space permits, one
compact Brolio merge/restoration example is more useful than a large experiment
inventory. Existing local galleries and raw images are evidence sources; a
contextual token's grid coordinate must not be labeled a causal pixel region.

Possible future experiments are recorded separately in the
[stage-one closeout note](../../notes/2026-10-21-stage1-closeout-and-future-ideas.md).
