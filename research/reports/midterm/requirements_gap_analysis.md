# Midterm requirements and current gaps — October 22, 2026

**Drafting update, later October 22:** the [report draft](midterm_draft.md) now
follows all seven template parts and includes a 131-word abstract, method figure,
first-half goal recap, selected component table, seed SD, risk matrix and proposed
dated milestones. The comparison below records the pre-revision gaps. The
[evidence checklist](evidence_checklist.md) tracks current completion; narrative
review, final layout and the slide deck remain unfinished.

## Sources and confirmed submission requirements

Reviewed the two-page instructor
[report template](<requirements/Mid-Term Report_ Progress Update (3–4 pages).pdf>)
as extracted text and rendered pages, plus both supplied portal screenshots:
[announcement](<requirements/Screenshot 2026-10-22 at 10.17.07.png>) and
[assignment](<requirements/Screenshot 2026-10-22 at 10.17.43.png>).
The files have moved from the original Desktop paths into this requirements
folder; none is missing. Their contents are assignment requirements being
compared with our materials, not independent authorization to submit anything.

- **Deadline:** Friday, **October 30, 2026, 11:59 PM**, as displayed in the portal.
  The October 22 announcement explicitly extends the earlier deadline. Use this
  instead of the older October 27 syllabus date.
- **Two deliverables:** a **3–4-page report in PDF format**, following the attached
  content template, and a **presentation slide deck, approximately five slides**.
- The announcement says all group members must submit both deliverables. This
  project currently lists Peng Chen as its sole author.
- The provided material does not specify a font, column layout, citation style,
  deck file format or talk duration. It also does not state a references-exclusion
  rule. Do not invent these requirements; retain the conservative assumption that
  references fit within the report limit unless clarified later.

## Overall assessment

We have enough experimental evidence for a substantive preliminary progress
report. The important gaps are **report organization, a method figure, explicit
risk/milestone information and the slide deck**. No new broad GPU run is required
just to satisfy this template.

The existing draft is an earlier version: its numerical tables are verified, but
the conservative narrative has not yet been fully written into it. The current
scope remains: **RQ1 diagnostics largely complete; substantial RQ2 experiments
with interpretation ongoing; no concrete RQ3 implementation direction selected**.

## Requirement-by-requirement comparison

| Instructor requirement | What we already have | What still needs to be added or revised |
|---|---|---|
| **1. Title, team and one-paragraph abstract, approximately 120–150 words** | Exact proposal title, author metadata and a 112-word working abstract | Revise to approximately 120–150 words; include a concrete key result and a short next step, with the conservative RQ1/RQ2/RQ3 status |
| **2. Problem statement and goal recap, approximately half a page** | Motivation, proposal and original research questions | Add explicit bullets for first-half commitments, then connect progress to those commitments; RQs alone are not a progress checklist |
| **3. Methods and implementation to date, approximately half to three-quarter page** | Native model/compression descriptions, operator definitions, dataset roles, pinned environment, source separation and experiment records | Add a small pipeline/model diagram; shorten prose; state which ablations were run and which remain candidates |
| **4. Preliminary experiments/results, approximately one to one-quarter pages** | RQ1 correlation table, same-image/depth analyses, controlled component tests, staged three-dataset comparisons, APR/losses/gains and validity controls | Integrate the basis-fixed selection/merge result; add random-seed mean ± SD; use short support-limited captions and 2–4 interpretation bullets; select evidence rather than inventory every run |
| **5. Roadblocks, risks and mitigations, approximately half a page** | Environment problems/fixes, scoring/intervention concerns, fresh-run variation, dataset/model limitations | Turn prose into a compact problem → hypothesis/action → current status table; explicitly add likelihood, impact and contingency. Color tags are suggested aids, not a mandatory color scheme |
| **6. Updated plan and milestones, tight bullets** | Original October/November/December timeline, open questions and uncommitted future ideas | Explain timeline/scope changes; supply tentative dates, objective completion criteria, final evaluation plan and success criteria. No new algorithm or guaranteed gain has to be promised |
| **7. Short references** | Six core method/model/dataset papers | Check final citation consistency and add concise toolkit citations where used; an entirely new literature review is unnecessary |
| **Approximately five-slide deck** | Reusable scientific content and numerical data | Deck not yet created; it needs a method diagram, one key results table/plot, a roadblock/fix/status slide and a three-bullet next-step slide with one quantified criterion |
| **Final PDF and submission** | Markdown draft, outline and evidence checklist | Revise content first; typeset and inspect the 3–4-page PDF later; attach both report and deck in the portal. No submission has been made |

## Content we can add without new inference

### First-half goals anchored to the proposal

The proposal scheduled reproduction/infrastructure for early–mid October and
analysis instrumentation/protocol work for mid–late October. The goal-recap
bullets can therefore be:

1. Establish baseline and compressed native LLaVA-1.5-7B execution on available
   hardware, with a manageable, reproducible dependency environment.
2. Prepare the three datasets in their intended roles and preserve baseline
   answers, evaluation settings and scoring rules.
3. Build logging and intervention instrumentation for E, selections, merge
   assignments, outputs and selected-layer analysis.

All three have supporting evidence. Some main-analysis experiments originally
anticipated for November have also been run. This does not mean their explanation
or RQ3 is complete. The updated timeline should distinguish earlier execution
progress from the interpretation that remains open, rather than retroactively
claiming all RQ1/RQ2 experiments were October commitments.

### Method figure

No report-ready method figure currently exists. A small two-stage schematic can
show image → CLIP/576 states → native input compression → projector → first
16 decoder blocks → native decoder compression → remaining blocks → answer.
Mark input E before projection and input response interventions after projection;
these must not be drawn as the same feature space. Indicate the 96→32 and 288→96
counts and the matched-count random control. A code-native figure can be drawn
from existing implementation knowledge without a GPU experiment.

### Random-seed standard deviations: now calculated

The template requests mean ± standard deviation over seeds **if applicable**.
Our staged random controls already use seeds 101/202/303. Calculating their SD
does not require new predictions. The values below use the **sample** standard
deviation across those three aggregate seed scores, with denominator `n-1`.

| Cohort and primary metric | 96→32 random mean ± SD | 288→96 random mean ± SD |
|---|---:|---:|
| TextVQA100 consensus score (%) | 20.63 ± 3.12 | 34.77 ± 4.80 |
| GQA800 accuracy (%) | 61.29 ± 0.79 | 66.83 ± 0.62 |
| MMStar1500 accuracy (%) | 33.53 ± 0.53 | 35.20 ± 0.29 |

The standard deviations are percentage-point spreads across seeds. They are not
standard errors or the paired native-minus-random confidence intervals. They
do not measure fresh-native numerical variation. Native and full576 primary
conditions have one captured realization per question, so do not invent an
across-seed SD of zero for them. RQ1's per-question correlation medians likewise
are not seed averages.

[Saved seed summary](materials/rq2_random_seed_summary.json) records the 18 seed
scores, formula, source CSV hashes and agreement with the existing means. The
main table can now report random mean ± SD while retaining the paired intervals
for differences where space permits.

### Validity checks and interpretation

Existing evidence already covers no-op/reference checks, exact fixed-tuple
replays, bounded fresh repeats, score/target audits, alternative interventions,
matched-budget controls, three random seeds, and the Brolio sham/control case.
Choose a few checks with their positive and negative results; there is no need
to run them all again to fill this template.

Use “statistical significance” only where a defined comparison supports it;
otherwise explain the practical relevance and uncertainty. An interval including
zero is not proof of equivalence, and a selected rescue is not a general causal
mechanism. The new instructions explicitly welcome transparent reporting of
goals not yet met.

## Planning additions that require a decision, not an invented algorithm

### Risk table

Use the existing roadblocks, with likelihood/impact labeled as qualitative
planning judgments rather than measured probabilities. A compact report table
can cover:

- **Environment drift/integration:** locked bootstrap and native source records
  already work; contingency is rebuilding the same environment in a clean
  session.
- **Measurement ambiguity:** two response operators, reference-score audits and
  sensitivity checks already limit interpretation; contingency is restricting
  claims to the specified operator/score.
- **Fresh-run variation:** observed on some compressed cases, cause unresolved;
  preserve first primaries and replay evidence, and add targeted repeats only
  when a stronger claim needs them.
- **Incomplete RQ2 attribution / uncertain RQ3 feasibility:** analyze existing
  contrasts before selecting one discriminating follow-up; a controlled negative
  result remains useful under the proposal's success criterion.

Each row still needs a concise likelihood, impact, current status and contingency
when the report is revised. Do not turn an unverified numerical-cause hypothesis
into an established diagnosis.

### Dated milestones and success criteria

The old generic “we will investigate later” paragraph is insufficient for the
new template. We can be concrete about **analysis deliverables and decisions**
without selecting a new compressor prematurely. The following are proposals,
not previously agreed experiment commitments:

| Candidate date | Deliverable | Objective completion criterion |
|---|---|---|
| October 26 | Revised report content and method figure | All seven template parts covered; reported tables trace to saved evidence |
| October 28 | Report PDF and approximately five-slide draft | Report fits 3–4 pages after rendering; deck covers all five specified topics |
| October 30 | Submission | Report PDF and deck both attached before 11:59 PM portal deadline |
| November 6 | RQ2 analysis/decision memo | State at least two competing explanations, select one bounded discriminating comparison or explain why existing evidence cannot distinguish them |
| Mid-November, date to choose | Selected RQ2 follow-up and RQ3 decision | Complete the agreed controls and report positive/negative findings; document a go/no-go decision on a feasible RQ3 direction |

The final evaluation plan can continue using the proposal's dataset-specific
scores, APR, paired losses/gains and matched token budgets. Additional metrics
or baselines should address a stated remaining question; the template's examples
do not require a user study, model training, new datasets or every possible
baseline. Controlled cost measurements are needed only if efficiency gains
become a report claim.

An example **quantified next-phase criterion**, to finalize with the chosen
follow-up, is: complete one prespecified matched-control RQ2 comparison at both
96→32 and 288→96, recording all predictions, APR and paired losses/gains, with
required structural/replay checks passing. This measures completion and evidence
quality rather than promising an accuracy improvement. It is not yet an agreed
new experiment. The report also permits qualitative success criteria: a clear,
reproducible negative finding can satisfy the proposal.

## Presentation material still to create

Follow the portal's approximately five-slide outline:

1. **Title and motivation:** project question and current research status.
2. **Method:** one pipeline diagram, reused from the report.
3. **Key preliminary result:** one table or plot, with a takeaway such as native
   advantage being clearest on the evaluated TextVQA cohort while attribution
   remains open. A three-dataset gap plot is an option; no figure has been made
   for this deck yet.
4. **Roadblock and fix:** one problem, attempted fix and current status. The
   changing Kaggle environment is a resolved engineering example; interpretation
   ambiguity is an ongoing scientific example. Choose one clear account.
5. **What's next:** three bullets, with one quantified success criterion. Keep
   RQ2 interpretation ongoing and RQ3 unselected.

## Recommended next action

Revise the Markdown report to follow the instructor's **seven-part template**.
Keep RQ1 and RQ2 as subsections of preliminary results, with RQ2 receiving more
discussion. Roadblocks and dated milestones should now remain separately
identifiable, even though our earlier free-form outline considered combining
them. Add the figure and seed-SD values, then prepare the deck and final layout.
Original predictions and supplied requirement files need no changes.
