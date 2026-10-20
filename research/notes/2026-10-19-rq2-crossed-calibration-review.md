# RQ2 crossed selection/merging calibration review — 2026-10-19

The seven-question / six-image calibration passed. The random retained-selection
control reveals meaningful differences, especially literal-text recoveries. This
is accepted to gate the fixed full30 experiment; no measurement fix or calibration
rerun is needed. It is not evidence that random selection is generally superior.

## Preserved evidence and independent checks

Copied ZIP and separate log into `research/results/`, leaving Downloads originals:

- `rq2-crossed-calibration-v1-20261019T233509971034Z.zip`: 479,438,544 bytes;
  SHA256 `2424a4b1a46af82191be4b5890f50e534feabb1ce3b776d40f09d5fa2f970724`.
- Matching `.log`: SHA256
  `094031364fcac810f9a764347dd97cd578544e59f8e5e2ef8b0ac4cc32932fe6`.

Verified all 168 unique ZIP members by streamed CRC/SHA, 167 artifact hashes,
112 evidence hashes, 32 feature-file hashes and seven question hashes. Measurement
and auxiliary source copies match their recorded digests. All eight pinned native
sources and the original lock match; measurement hashes also match the current
workspace. The runtime commit is clean `4d0cc311f5e8d6f731b585fd8b0a108dea18692c`.
The embedded setup log records all 21 native prerequisites passing with no skips.

Ran the existing full-run calibration validator against actual temporarily
extracted evidence. It passes. No tensor artifact was unpickled or model executed
on the Mac. Independently rescored all 63 predictions, recomputed summaries and
all random selected sets from their frozen image/seed rule, and checked native
E/FPS/pre-merge/basis placement, pair masks, exact clone replay and same-image
feature-bank reuse. All 25 fixed-tuple repeats are exact. All 20 fresh captures
(E merge and random101 merge on five controls) preserve generated IDs/correctness;
their largest reference-token log-probability difference is 0.00809538. Numerical
variation persists but did not change these repeated answers. These repeats do
not establish universal determinism or repeat every random seed.

All 21 shared-condition generated-ID comparisons against the earlier v2 full run
(seven questions × baseline/E merge/E no-merge) match. Native outcomes were not
replaced to favor this comparison. Wall time is 456.65 seconds, including loading
and evidence work; this is not a controlled runtime benchmark.

Derived verification, answer CSV, fresh-repeat CSV, historical comparison and
summaries are in `results/rq2-crossed-calibration-v1-review-2026-10-19/`.

## Descriptive outcome

All seven fresh full576 predictions are correct. Each compressed condition keeps
288 states: ten fixed native FPS basis representations/slots plus 278 ordinary
retained positions. Random selection changes the ordinary positions; it is not
random basis selection or random merge destinations.

| Ordinary selection | Native merging | No merging |
|---|---:|---:|
| E ranked | 2 / 7 | 4 / 7 |
| Random101 | 5 / 7 | 5 / 7 |
| Random202 | 5 / 7 | 6 / 7 |
| Random303 | 5 / 7 | 5 / 7 |

Keep seed-specific results; do not choose202 after seeing its favorable result.
These are seven repeated questions, not 21 independent examples, and two GQA
questions share one image. Dataset breakdown is GQA2, MMStar1 and TextVQA4.

| Case | E merge | E no-merge | Random observations |
|---|---|---|---|
| Cake/plate color | Yes, wrong | No, correct | All three seeds correct with/without merging |
| Same-image plate/stove control | Stove, correct | Stove, correct | All seeds correct in both modes |
| MMStar313 building count | B, wrong | B, wrong | All seeds wrong in both modes;101 says A,202/303 say B |
| Brolio | Bottoli, wrong | Brolio, correct | All seeds recover Brolio in both modes |
| Dominion preserved control | Dominion, correct | Dominion, correct | All seeds correct in both modes |
| Cookery school | Culinary school, wrong | Culinary school, wrong | 202 recovers exact name in both modes; 101/303 say Cooker school, still wrong |
| Give up | Gone, wrong | Go up, wrong | All seeds recover without merging; 101/303 recover with merging, 202 says Gone |

TextVQA full-credit counts are 1/4 for E merge, 2/4 for E no-merge, 3/4 for each
random merge, and 3/4, 4/4, 3/4 for random no-merge. Consensus scores here give the
same aggregate ordering; keep the two measures distinct in later panels.

## What this adds to RQ2

1. **Selection can matter even without merging.** Cookery and Give up recover
   under some different selected sets while E no-merge remains incorrect. Thus
   omitting merging alone does not explain every observed recovery. Changing a
   set still changes many features/context positions; no single source is
   identified as the cause by this comparison.
2. **Merging's effect depends on the selected set.** E merging loses cake/plate
   and Brolio relative to its no-merge pair. Random202 merging loses Give up
   relative to its otherwise identical selected set. Merging can introduce a
   loss under a random set too; randomness does not eliminate the issue.
3. **Brolio motivates the next routing/protection control.** Target 317 is an
   ordinary retained target in E and all three random sets. Its native incoming
   sources are 103/109/331; random101/202 give it no incoming sources, whereas
   random303 gives it 115/320/328 and still answers correctly. This is consistent
   with routing/context mattering beyond simple target presence. Other targets
   also change across selections, so it does not establish 317 as a unique cause
   or demonstrate letter blurring. The separate restore-only317 experiment is
   still optional and post-hoc.
4. **No universal fix is shown.** Cookery remains wrong for two seeds and the
   MMStar count case fails under every compressed condition. Preserve these
   negative cases alongside recoveries.

The main sampling qualification is stronger than merely small N: five of these
seven calibration questions were included as known native E-merge failures, with
two preserved controls. E selection/merging was therefore involved in choosing
the diagnostic cohort, while random outcomes were unseen. These paired tests
show recoveries on selected known failures, not an unbiased ranking of random
versus E on a population. Full30 adds more preserved controls but remains the
same selected mechanism cohort. A practical superiority claim would require a
frozen broader evaluation pool rather than adding attractive rescue examples.

## Next run

Proceed with the already frozen full30 crossed experiment, retaining all three
random seeds and all failures. No environment reinstall or new input upload is
needed in the live session:

```python
crossed_full_dir = run_rq2_crossed(
    mode="full",
    calibration_report=crossed_calibration_dir / "report.json",
)
```

Keep the complete calibration directory because the gate checks actual hashes
and embeds its evidence. Download the new ZIP/log afterwards. Routing/protection
has a separate implemented runner/calibration; avoid changing selection/routing
policies in response to these calibration answers.
