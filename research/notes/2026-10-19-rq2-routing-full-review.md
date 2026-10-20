# RQ2 routing/protection full30 review — 2026-10-19

The full routing/protection panel passed: 30 questions / 24 images, ten conditions,
300 primary predictions. Same-condition repeats remain stable, while the policy
effects are small and case-specific. The high-E protection policy has little
routing activity on this cohort. This completes the planned first bounded
routing/protection comparison, without establishing a generally superior policy.

## Preserved evidence and verification

Copied both supplied files into `research/results/`, retaining Downloads originals:

- `rq2-routing-full-v1-20261020T004043438261Z.zip`: 2,457,480,343 bytes;
  SHA256 `06947c58d3d785af446b84f8983d92e679a0d56a084a7a58527be733a5a0ae9a`.
- Matching `.log`: 18,116 bytes;
  SHA256 `219194354119d3145f9d2ae63b99b8c276f44e1f562fc5f6084798c329ea4c33`.

All 455 unique archive members pass streamed CRC/SHA checks: 454 artifact hashes,
284 evidence hashes, 88 feature-file hashes and 30 question hashes. Measurement
and auxiliary source copies match their recorded digests. Current measurement
hashes, all eight pinned native files and the original environment lock match.
The clean runtime commit is `2140bb6b89a1fb350954c1a6603048b6feafc713`. The setup
evidence records 21 native prerequisites passing without skips.

The embedded routing calibration report is byte-identical to the standalone
calibration, with SHA256
`ba44dbccdd012b9dd73a07e054ffc67ca09be34648aa57dfa6e85fea15e5a4ee`.
Its identity matches the full run. Temporarily extracted only the prerequisite
subset to run the existing full-run gate against actual evidence; it passes.
The full ZIP was processed with bounded reads, without loading feature pickles
or running a model on the Mac.

Independently rescored all 300 predictions, reproduced summaries, regenerated
the prescribed shuffles/protected sets, and checked source/E/FPS/selection/basis/
mask, recipient counts, protection records and exact clone-replay identities.
Same-image questions share the same feature bank. All 30 fixed replays are exact.
All 40 fresh captures preserve generated IDs/correctness, with maximum reference-
token log-probability difference 0.01103735. These bounded repeats do not imply
universal numerical determinism.

All 70 calibration/full generated-ID comparisons match, as do all 90 shared
baseline/native/no-merge comparisons with the earlier crossed full30 run. There
is no baseline or target drift requiring a rerun. Runtime tensor identity checks
are reviewed through their implementation, recorded assertions and artifact
bytes; tensors are not independently recomputed in the editor environment.

Derived verification, per-question answers, paired outcomes, routing activity,
calibration/history comparisons and native E-rank activity are in
`results/rq2-routing-full-v1-review-2026-10-19/`. The review script is saved there
with its digest. Original archives/logs remain unchanged and ignored by Git.

## Correctness and paired outcomes

All full576 baseline answers are correct. Compressed conditions use exactly the
same native E-selected 288 positions, ten FPS basis representations/slots and
decoder-disabled settings. Randomization here changes destinations or protected
targets; it does not change ordinary retained-position selection.

| Policy | GQA /12 | MMStar /12 | TextVQA /6 | Total /30 |
|---|---:|---:|---:|---:|
| Full576 | 12 | 12 | 6 | 30 |
| Native nearest | 8 | 8 | 3 | 19 |
| No merge | 9 | 8 | 4 | 21 |
| Shuffled101 | 9 | 8 | 3 | 20 |
| Shuffled202 | 9 | 8 | 3 | 20 |
| Shuffled303 | 8 | 8 | 3 | 19 |
| Protect high-E28 | 8 | 8 | 3 | 19 |
| Random-protected101 | 8 | 8 | 4 | 20 |
| Random-protected202 | 8 | 8 | 3 | 19 |
| Random-protected303 | 8 | 8 | 4 | 20 |

Totals are descriptive for this selected cohort, not benchmark accuracy estimates.
The seeds repeat the same questions, which must not inflate sample counts.

| Policy versus native nearest | Rescues | New losses | Raw-answer changes |
|---|---:|---:|---:|
| No merge | 2 | 0 | 3 |
| Shuffled101 | 1 | 0 | 3 |
| Shuffled202 | 1 | 0 | 3 |
| Shuffled303 | 1 | 1 | 4 |
| Protect high-E28 | 0 | 0 | 1 |
| Random-protected101 | 1 | 0 | 2 |
| Random-protected202 | 0 | 0 | 0 |
| Random-protected303 | 1 | 0 | 2 |

No merge rescues cake/plate and Brolio. Every shuffled arm rescues cake/plate;
shuffled303 additionally loses the previously correct pan/window question,
answering Picture. Random-protected101/303 recover Brolio, with no other
correctness changes. High-E protection changes Give up's wrong answer from Gone
to Grow up but produces no correctness gain or loss.

All MMStar correctness outcomes remain identical under these routing/protection
arms. In the earlier crossed study, random retained-position selection with
merging recovered three native MMStar failures across every seed. Thus changing
which ordinary positions survive has larger observed effects in this cohort
than the routing/protection variants tested here. This is a descriptive component
comparison, not an additive percentage attribution or a population ranking.

## High-E protection is mostly inactive on assignments

Each image has 288 dropped sources, for 6,912 assignments across 24 unique images.
The protection count is always 28 ordinary targets per image, but their native
incoming source masses differ greatly:

| Policy | Images with zero assignment changes | Total redirected sources |
|---|---:|---:|
| Protect high-E28 | 15 /24 | 21 |
| Random-protected101 | 0 /24 | 980 |
| Random-protected202 | 1 /24 | 720 |
| Random-protected303 | 0 /24 | 556 |

High-E protection redirects only 0.304% of sources. Consequently its unchanged
correctness is weak evidence about the efficacy of actively protecting an
answer-relevant feature: much of this policy has no incoming merge to block.
Random protection changes substantially more routing despite matching target
count. Its effect combines which targets remain unchanged with rerouting/load
changes elsewhere. Do not attribute that full difference exclusively to E score.

The count-preserving shuffles change 5,207 /5,213 /5,190 assignments for their
three seeds (about 75% of all assignments), while preserving each target's incoming
count exactly. Only a few generated answers change. This limits any claim that
similarity routing is essential for all these selected answers, but does not
establish that routing is universally irrelevant or that shuffled compression
would perform as well on a broader pool.

### Descriptive saved-data check: where do dropped sources go?

For each image, sort only the 278 ordinary retained targets by native E and split
them into four ranked groups, excluding basis slots. Count each image once.
This post-hoc graph summary is not a new intervention or an answer-importance
measurement:

| Retained-target E group | Incoming sources | Share of all 6,912 sources |
|---|---:|---:|
| Lowest quartile among ordinary retained targets | 4,569 | 66.10% |
| Lower-middle quartile | 1,909 | 27.62% |
| Upper-middle quartile | 308 | 4.46% |
| Highest quartile | 126 | 1.82% |

The lowest 28 ordinary targets receive 3,092 sources (44.73%); the highest 28
receive 21. This explains why protecting highest-E targets barely alters merging
in these images. It is consistent with distinctive high-E states being less
frequent nearest neighbors for dropped states, but does not establish a universal
causal explanation. Receiving many sources is not itself proof of answer relevance.
Group sizes differ by one target because 278 is not divisible by four.

## Brolio and the oracle remain compatible

The full run repeats the calibration's wine results exactly. Native and high-E
protection answer Bottoli. Random-protected101/303 answer Brolio;202 stays Bottoli.
Only303 protects317;101 retains its original incoming sources 103/109/331 and
still recovers while other context states change. Per-target byte equality at317
is not established by incoming-source graph equality across independent captures. Shuffling yields wrong names
Castello di volpaia/gabbiano despite preserving recipient loads.

The separate restore-only317 intervention remains a valid conditional rescue.
It did not establish317 as a unique necessary repair. Random101 gives another
route to recovery; random303 changes many targets and reroutes many sources,
so it cannot independently localize recovery to317. No letter-level corruption
mechanism or deployable gold-informed protection policy is established.

## What is complete and what should follow

The first bounded RQ2 comparisons are now executed and verified: native merge/
no-merge, E/random ordinary selection × merge/no-merge, fixed-selection shuffled
routing, high-E/random protection, and the separate Brolio oracle. No runtime
fix or repeat of these full panels is needed.

The supported story is heterogeneous: selected sets matter; merging can help or
harm particular questions; one retained state's restoration can rescue a failure;
and high-E protection barely acts on the native graph in this cohort. The
failure-enriched, baseline-correct panel and prior MMStar flags remain limits on
all general efficacy claims.

Next consolidate RQ1/RQ2 using saved data, then plan the broader frozen-pool E/
random confirmation already discussed. Further protection variants should be
motivated separately, with active-recipient/source-mass controls if needed, rather
than tuning this 28-target policy around its observed rescues. Broader comparisons
must retain all predefined cases and seeds, not just known losses or the best seed.
No new GPU experiment or algorithm tuning is started by this review.

Wall time 552.47 seconds includes loading/instrumentation/evidence serialization
and is not a controlled inference-speed benchmark.
