# RQ2 E/random selection × merge/no-merge full30 — 2026-10-19

The full30 run passed on all 30 questions / 24 images with all nine conditions.
The random ordinary-position controls preserve more correct answers than native
E selection for every predeclared seed in this diagnostic panel. Merging has
mixed paired effects under random selection. These are selected-cohort findings,
not a general ranking of compression methods.

## Preservation and verification

Copied both supplied files into `research/results/`, retaining Downloads originals:

- `rq2-crossed-full-v1-20261019T234435011493Z.zip`: 1,991,976,073 bytes,
  SHA256 `0a7df42a2cdb1a88b6260d903bc62a88298fcdb79efcae410c220d76c32ef070`.
- Matching `.log`: 17,062 bytes,
  SHA256 `a3c84e34e94034d0caad8c9c2d651c3050291d2df75fbd29edce36387236cd79`.

The archive was processed with bounded chunked reads. All 415 unique members pass
CRC/SHA verification, including 414 artifact hashes, 264 evidence hashes, 68
primary/fresh feature-file hashes and 30 question hashes. Measurement/auxiliary
source copies match recorded hashes, eight native sources match the pinned
reference, and the original environment lock matches. Current measurement hashes
still match. Runtime commit is clean `4d0cc311f5e8d6f731b585fd8b0a108dea18692c`.

The embedded prerequisite report is byte-identical to the supplied standalone
calibration. Its SHA256 is
`ff111c7712688127492d848c56952b84fe995e71b94243675b8f64aca546846c`.
Only that prerequisite subset was temporarily extracted to run the existing
full-run calibration validator against actual files. It passes; the full run's
calibration identity matches. No feature pickle was loaded or model run on the Mac.
The setup evidence records all 21 native prerequisites passing without skips.

Independently rescored 270 primary predictions, reproduced all dataset summaries,
and recomputed every random selected set from its frozen image/seed rule. Exact
E/FPS/source/pre-merge/basis-placement identities, merge/no-merge pair selection,
clone replay and same-image bank reuse pass. All 25 fixed-tuple repeats are exact.
All 20 declared fresh repeats preserve generated IDs and correctness, while the
largest reference-token log-probability difference is 0.00790381. These bounded
fresh repeats cover E merge and random101 merge on the five declared controls.

All 90 shared-condition generated-ID comparisons against the previous v2 full
run match: baseline, E merge and E no-merge on all 30 questions. Against the
crossed calibration, 62/63 generated-ID comparisons and all 63 correctness
comparisons match. The exception is `mmstar:313`, random101 merge: A → B, both
incorrect (gold D). Source/E/FPS/selection/basis/merge graph match, while merged
and projected feature hashes differ. This is compatible with the previously
observed numerical variability but does not establish its precise cause. The
case was not among fresh-repeat controls; their stable answers do not imply
stability everywhere. No retargeting or rerun is required for this wrong-to-wrong
change; preserve it as a repeatability limitation.

Derived JSON/CSV verification, per-question answers, paired outcomes, flag
sensitivity, calibration comparisons and the review script are in
`results/rq2-crossed-full-v1-review-2026-10-19/`.

## Why the ZIP is large

Approximate compressed payload sizes: 1.497 GB of primary/fresh feature tensors,
0.477 GB of embedded calibration evidence, and 0.017 GB of other JSON, images and
source evidence. The 44 primary/fresh tensor files are retained for auditing;
ordinary answer/summary analysis uses lightweight JSON/CSV. Do not put this ZIP
in Git. Keep the raw archive and log separately from tracked code and notes.

## Primary outcome

All 30 full576 baselines are correct under the existing dataset scorers. Every
compressed arm retains 288 states, with the same ten FPS basis representations
and original basis-slot placement; random selection changes 278 ordinary positions.

| Selection | GQA merge / no merge | MMStar merge / no merge | TextVQA merge / no merge | Total merge / no merge |
|---|---:|---:|---:|---:|
| E | 8 / 9 | 8 / 8 | 3 / 4 | 19 / 21 |
| Random101 | 9 / 9 | 11 / 10 | 5 / 5 | 25 / 24 |
| Random202 | 7 / 10 | 11 / 11 | 5 / 6 | 23 / 27 |
| Random303 | 10 / 11 | 11 / 10 | 5 / 5 | 26 / 26 |

Denominators: GQA12, MMStar12, TextVQA6, total30. Totals are descriptive, not a
benchmark average. Seed-averaged correct counts are 24.67 with merging versus
native E's 19, and 25.67 without merging versus E's 21. These are averages of
realizations on the same 30 questions, not 90 independent questions. Retain all
three seeds rather than choosing the best seed after seeing outcomes.

Relative to the corresponding E arm:

| Random seed | Merge: rescues / new losses | No merge: rescues / new losses |
|---|---:|---:|
| 101 | 7 / 1 | 4 / 1 |
| 202 | 7 / 3 | 6 / 0 |
| 303 | 8 / 1 | 5 / 0 |

Random selection is not uniformly beneficial: it introduces failures in previously
preserved cases, especially when merging is enabled. For example, the GQA stove-
left-of-refrigerator question becomes Dishwasher/Microwave for random101/202
merge, while their no-merge arms remain correct. The same-image pan/window and
paper-towel questions also expose different question/seed-specific merge effects.

## Selection and merging interact

Relative to its own identical selected-set no-merge arm:

| Selection | Rescued by merging | Lost after merging | Net correct-count change |
|---|---:|---:|---:|
| E | 0 | 2 | -2 |
| Random101 | 2 | 1 | +1 |
| Random202 | 0 | 4 | -4 |
| Random303 | 1 | 1 | 0 |

Random101 merging recovers the GQA pan/window answer and MMStar1030 but loses
the stove-left-of-refrigerator answer. Random303 recovers MMStar1030 but loses
the pan/window answer. Random202 loses three GQA answers and Give up. Thus
disabling merging does not dominate every selected-set/question combination.
Do not translate these interactions into a fixed additive percentage of ApET
performance attributable to selection or merging.

## Case-level links to RQ1

- **Brolio:** E merge gives Bottoli; E no-merge and every random condition give
  Brolio. The original target317 remains retained under all selections, but its
  incoming sources/context differ. This continues to motivate the separate
  routing/protection and optional targeted-restoration tests; it does not prove
  a literal letter-blurring mechanism.
- **Cookery school:** E merge/no-merge give Culinary school. Random202 gives the
  exact name in both modes;101/303 give Cooker school, still incorrect. Selection
  matters even after merging is removed, but one isolated causal token is not
  identified by changing hundreds of retained positions.
- **Give up:** E gives Gone with merging and Go up without. All random no-merge
  conditions recover Give up; random202 merging still gives Gone. This is a
  clean selected-set-dependent merge loss.
- **GQA television:** E gives Phone in both modes; every random condition gives
  Television. The full-grid RQ1 observations can annotate the source roles later,
  but this result itself measures generated-answer changes under compression.
- **MMStar:** all random merge arms get11/12; the building-count case313 remains
  wrong. Three E failures (beak617, bedroom895, angle1030) recover with merging
  across all seeds. Previous grounding/quality flags remain relevant, particularly
  angle1030; correct letters alone do not prove preserved visual reasoning.

Sensitivity analyses retain the flags instead of deleting difficult cases from
the primary cohort. Removing the five MMStar grounding-flagged cases leaves
E merge/no-merge4/7 and all random arms6/7. Removing all seven cases with either
MMStar flag leaves E3/5 and all random arms4/5. The direction persists in those
smaller subsets, with fewer available examples; this does not remove every
grounding or selection concern. Primary results still include all frozen cases.

## Interpretation and next step

The diagnostic results support a narrower statement: high-E ordinary-token
selection is not necessary for preserving these particular answers, and alternative
sets can rescue known failures; merging can both help and harm depending on the
set and question. They do not establish random selection as a better general
compressor or show that E/FPS are irrelevant to the pipeline. The random control
retains native basis machinery and its E-dependent spatial insertion positions.

The cohort was chosen using prior model/compression outcomes and review, including
baseline-correct known failures. It is not an unbiased sample of GQA, MMStar or
TextVQA performance. GQA questions also share images. A broader evaluation must
include all predefined pool members, and report accuracy plus baseline-correct
preservation with all fixed seeds, rather than selecting newly attractive cases.

The next implemented experiment is the separate routing/protection calibration
in [RQ2_ROUTING.md](../RQ2_ROUTING.md), followed by its full30 if controls pass.
It holds native E selection fixed to compare destination similarity and target
protection. A broader E/random comparison is now worth considering after those
bounded controls; no new broad sweep or protocol tuning is started by this review.
The optional Brolio restore-only317 diagnostic remains separate. Existing original
native source, environment and archived results require no change or rerun.

Observed wall time is 578.77 seconds including loading/evidence serialization;
it should not be extrapolated as an instrument-free inference benchmark.
