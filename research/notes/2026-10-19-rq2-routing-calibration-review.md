# RQ2 routing/protection calibration — 2026-10-19

The calibration passed on seven questions / six images with all ten conditions.
There is no uniform improvement across routing/protection policies, but the
same-condition repeatability checks are stable. An important structural finding
is that the chosen high-E protection policy changes almost no assignments on
these images. Mixed efficacy is not an execution failure or a reason to tune the
policy on these calibration answers.

## Preserved evidence and verification

Copied the supplied ZIP/log into `research/results/`, keeping Downloads originals:

- `rq2-routing-calibration-v1-20261020T003219559954Z.zip`: 651,848,652 bytes;
  SHA256 `e657d69cf7a6f9f8fc999c7d00139bab774e62e9eda0d7e9c2fccbc7f21c1d34`.
- Matching `.log`: 6,586 bytes;
  SHA256 `7a8779cb5c47deeb3a0846abf3da13223eb8b5548739f7bf9f5e7e95b1211087`.

All 188 unique members pass streamed CRC/SHA verification: 187 artifact hashes,
112 evidence hashes, 52 feature-file hashes and seven question hashes. Measurement
and auxiliary source copies match their recorded digests; measurement sources
also match the current workspace. Eight pinned native sources and the original
lock remain unchanged. Runtime commit is clean
`2140bb6b89a1fb350954c1a6603048b6feafc713`. All 21 native prerequisites passed
without skips. The existing full-run calibration validator passes against actual
temporarily extracted evidence, including recomputed routing plans/graph controls.

Independently rescored all 70 primary predictions and recomputed summaries,
prescribed shuffles/protected sets, routing-count controls and protected-feature
runtime-check records. Exact source/pre-merge/E/FPS/selection/basis/mask and
same-image bank identities pass. All 30 fixed-tuple replays are exact. All 40
fresh captures preserve generated IDs and correctness; their largest fixed-
reference token log-probability difference is 0.01134157. These cover native,
shuffled101, high-E protection and random-protected101 on five declared controls.
They do not imply determinism for every possible example/seed/hardware.

All 21 shared baseline/native/no-merge generated-ID comparisons against the
previous crossed full30 run match. No baseline drift or retargeting is needed.
CPU review loads JSON and hashes raw evidence only; feature pickles were not
loaded and no model was run on the Mac. Runtime byte-exact feature assertions
are reviewed through their source code and archived records.

Derived verification, summaries, answer CSV, fresh-repeat CSV, routing-activity
CSV and review script are in
`results/rq2-routing-calibration-v1-review-2026-10-19/`.

## What is and is not consistent

**Repeatability:** same fixed tuple produces the same answer/reference scores;
declared fresh captures also give the same answers in this calibration.

**Policy efficacy:** changing destination assignments or the protected set can
change an answer. Different seeds define different interventions, so their
different outcomes do not by themselves indicate runtime instability.

Every compressed arm keeps the same native E-selected set, ten FPS basis sources
and basis placement, and 288-token budget. Here random means routing/protection,
not random retained-token selection from the preceding crossed experiment.

| Policy | Correct out of 7 |
|---|---:|
| Full576 | 7 |
| Native nearest merging | 2 |
| No merge | 4 |
| Shuffled101 / 202 / 303 | 3 / 3 / 3 |
| Protect highest-E 28 targets | 2 |
| Random-protected101 / 202 / 303 | 3 / 2 / 3 |

The two preserved controls (plate/stove, Dominion) remain correct under every
policy. The MMStar building-count example remains incorrect under every
compressed policy. Shuffling rescues cake/plate for every seed but does not
recover any of the three literal-text failures. High-E protection changes Give
up's wrong answer from Gone to Grow up; this is a raw-answer change, not a gain.

## High-E protection has very little routing activity

Protection excludes a target from receiving dropped sources. If it originally
received no sources, that protection does not redirect anything.

| Image / case | High-E protection: redirected sources |
|---|---:|
| Cake/plate + stove pair, one shared image | 0 |
| MMStar313 | 0 |
| Brolio | 0 |
| Dominion | 0 |
| Cookery school | 1 |
| Give up | 6 |

Across six unique images, high-E protection changes only seven of 1,728 source
assignments. Four images have no assignment change. Fresh GPU reductions can
still differ numerically even when the graph is identical; this is a policy
no-op on assignments, not a claim that independent feature tensors are bitwise
identical.

Random protection of the same 28-target count redirects 238, 107 and 221 sources
over those six images for seeds101/202/303. For Brolio alone random303 redirects
171 of 288 sources. Thus equal protected counts do not match incoming-source mass,
location or effective intervention size. A high-E-versus-random accuracy comparison
measures these full policy differences. It does not isolate E score alone from
how much merging is actually altered.

This suggests a useful hypothesis for later inspection: highly distinctive
high-E targets may often be poor nearest neighbors for dropped tokens and thus
receive little merging. This calibration documents the pattern, not its universal
cause. Keep the original policy frozen. If full30 shows the same near-inactivity,
an additional prespecified policy among active recipients or a source-mass-matched
control could be considered separately; do not silently replace this experiment.

## Brolio: two routes to recovery

Native target317 is an ordinary merge target receiving 103/109/331. Its E rank is
155 of 278 ordinary targets, so protecting the top 28 does not protect it.

| Policy | Protects317? | Incoming sources to317 | Answer |
|---|---|---|---|
| Native nearest | No | 103/109/331 | Bottoli |
| Protect high-E | No | 103/109/331 | Bottoli |
| Random-protected101 | No | 103/109/331 | Brolio |
| Random-protected202 | No | 103/109/331 | Bottoli |
| Random-protected303 | Yes | None | Brolio |

Random303 is compatible with the prior oracle finding, but changes 28 protected
targets and many other recipients, so its rescue cannot be assigned uniquely to317.
Random101 also recovers Brolio without protecting317 or changing its incoming
source identities. Changes elsewhere in the context can therefore be another
route to recovery; per-target bitwise equality at317 across these independent
captures is not established by graph equality alone.

The prior restore-only317 result remains valid conditional sufficiency. It never
established317 as the sole necessary rescue location. This new observation is a
reason to keep that distinction explicit, not a contradiction of the oracle.

All shuffled policies still give a wrong wine name (Castello di volpaia/gabbiano),
despite preserving each recipient's incoming count. Content assignment changes
the generated answer beyond count alone. This does not establish a broad accuracy
advantage for native similarity routing because both names are incorrect here.

## Next step

Proceed to the frozen full30 routing/protection run. A calibration gates correct
execution and evidence, not a positive or monotonic scientific result. Keep all
three seeds, the 28-target rule, all negative outcomes and original source/selection
controls. These seven failure-enriched cases do not estimate benchmark efficacy.

In the same live notebook:

```python
routing_full_dir = run_rq2_routing(
    mode="full",
    calibration_report=routing_calibration_dir / "report.json",
)
```

Keep the complete calibration directory for the gate and embedded evidence.
No environment reinstall, new input upload, measurement fix or calibration rerun
is required. Save the full ZIP/log. Review assignment activity alongside correctness
before choosing any broader experiment or additional active-recipient control.
Observed 335.61-second wall time includes loading/instrumentation and is not a
speed benchmark.
