# TextVQA100 high-norm replacement and decoder FPS precision

## Archive and validation

Copied calibration/full ZIPs and separate logs from Downloads into research/results,
retaining the originals. Run stamps: calibration `20261106T235426498992Z`, full
`20261106T235839918437Z`. No supplied file is missing.

- Calibration passed: 5 questions ×31 =155 predictions, 221.3 seconds.
- Full passed: 100 questions/images ×31 =3,100 predictions, 1,965.2 seconds (32.8 min).
- ZIP CRC checks passed. All 112 calibration and 682 full manifest artifact hashes matched.
- 32 native prerequisites passed, zero skips. Full report embeds passed calibration.
- 2,700 full-run shared-input/pre-prune-pool checks; first-case replay/cache checks
  for 30 custom arms plus the 576 identity control were recorded.
- All four historical references (full576 and three compressed parents) reproduced
  their archived generated token IDs. Empty drift lists for new arms do not imply
  independent historical replication: those arms have no historical reference.

Machine-readable outputs: `research/results/norm-intervention-full-review.json`
and `research/results/norm-intervention-integrity-review.json`.
The former is reproducible with `scripts/summarize_norm_intervention.py`.

## Main results

TextVQA consensus score on a 0–100 scale; control mean averages five fixed seeds.

| Experiment | Parent | Intervention | Matched control mean | Intervention minus control, pp (95% image bootstrap) |
|---|---:|---:|---:|---:|
| A1: replace highest-norm native-drop state | 40.1 | 39.2 | 37.26 | +1.94 [0.06, 4.02] |
| A2: replace top two native-drop states | 40.1 | 39.2 | 36.78 | +2.42 [−1.00, 5.52] |
| B1: replace highest-norm FPS→FPS state | 40.5 | 41.2 | 41.40 | −0.20 [−2.40, 1.40] |
| C1: restore highest-norm source into native-input→E | 24.4 | 23.4 | 23.60 | −0.20 [−0.60, 0.00] |
| C2: restore all missing >10×median sources | 24.4 | 23.4 | — | Descriptive; no count-matched control |
| Decoder FPS FP32, native-drop | 40.1 | 40.1 | — | 0 score difference |
| Decoder FPS FP32, FPS→FPS | 40.5 | 40.5 | — | 0 score difference |

Full576 remains 49.7. Intervals are exploratory, not multiplicity-adjusted, and
condition on these questions and five control seeds. A1's near-zero lower bound
is not a strong stand-alone discovery.

## Do the interventions actually remove/restore the intended states?

Yes. A1/B1/C1 and their controls swap exactly one state in all 100 cases; A2
swaps exactly two. Carrier positions do not change in any replacement arm.
All A1/A2 removals and their controls are basis-assigned slots in these data.
All resulting source lists have 32 unique direct sources.

Native-drop initially retains one/two/three >10×median states in 26/66/8 cases.
A1 removes the largest source in every case, leaving 0/1/2 high-norm states.
A2 leaves none in 92 cases and one in 8. C1 restores the largest source in 100/100;
C2 restores 1/2/3 high-norm states in 26/66/8 cases. Thus C2's lack of recovery
is not caused by an already-present-source no-op.

Answer token IDs change in 6 cases underA1,10 underA2,4 underB1,6 underC1,
and7 underC2. Equal aggregate scores do not imply identical predictions.

## What this tells us

The simple hypothesis that a single extreme-norm decoder state explains the
native basis advantage is not supported by this experiment. Replacing the top
one/two states does not collapse performance, and adding them back to the weak
E-selected output does not restore it. Low-norm basis-slot controls tend to be
more damaging than top-norm replacement; this motivates looking at the other
basis states and their content/coverage rather than identifying norm with utility.
It does not yet establish that diversity, text evidence, or another particular
property explains their usefulness.

The fixed-pool design isolates changes *after block16*. High-norm states have
already participated in the preceding layers and may have influenced surviving
states. Replacement also supplies new information instead of simply deleting a
state. Therefore this does not prove high-norm tokens are globally unnecessary,
that they never act as attention sinks, or that all their earlier effects vanish.

## Precision and padding

FP32 native-drop changes carrier membership in 1/100 and ordered direct-source
lists in 4/100. FPS→FPS changes membership/source lists in 3/100. Each changes one
generated answer, but both affected cases remain score 0:

- Native-drop,34700: `10:10` → `12:00`.
- FPS→FPS,38665: `Fruit` → `Milk`.

Hence half-precision FPS overflow is real, but decoder-only FP32 arithmetic did
not change consensus scores in this cohort. It is not proof of universal precision
irrelevance or exact answer equivalence.

Largest-source patch is fully padding in 31 cases, non-fully-padding in 69. The
saved summary reports each stratum. For example, A1 score delta versus parent is
−2.90pp in the padding group and0 in the other; C1 is0 and−1.45pp respectively.
These small descriptive subgroups do not establish a padding-specific mechanism.
Contextual states can carry information beyond their source patch.

## Next step

No immediate broad GPU rerun is needed. CPU review should identify low-norm
basis slots whose replacement consistently loses correct answers, compare their
source patches and audited OCR boxes, and inspect paired parent/control predictions.
Then select a small, predeclared matched content-restoration experiment if that
review reveals a coherent pattern. Keep the high-norm null/weak result alongside
the earlier basis benefit: retention of extreme norm alone does not explain it.

## Follow-up per-question audit (November 7)

Extended the saved-data summarizer to export every changed answer, consensus
score increase/decrease and full-credit transition relative to each arm's parent.
The numerical drops are concentrated in very few questions:

| Arm | Changed token sequences | Score decreases | Score increases |
|---|---:|---:|---:|
| A1 | 6 | 1 | 1 |
| A2 | 10 | 2 | 2 |
| B1 | 4 | 0 | 1 |
| C1 | 6 | 1 | 0 |
| C2 | 7 | 1 | 0 |

A1's score change is entirely explained by two cases:
- 36584: `7 to 7` (0.9) → `7-7` (1.0).
- 36773, “what is the name of the beer?”: `Meantime` (1.0) →
  `Meantime pale ale` (0.0).

The latter retains the original word and adds detail. Without an image/annotation
review it should not be called proof of losing the beer-name evidence. It also
loses score under all five A1 low-norm controls, so it is not specific to high norm.

A2 additionally changes `Pizza`→`Vegetables` (1→0,39313) and
`Glen`→`Highland` (0→1,39482). B1's sole score improvement is
`Monday night` (0.3)→`Monday` (1.0,37034), again involving answer specificity.

C1/C2 both lose one point on 38830, “what time do the phones read?”:
`3:40`→`3:00`. C1 replaces compact source42 with high-norm source24 at output
slot13. Random restoration controls displace the same source42; their mean
score is also lower. This is consistent with the displacement itself mattering,
but does not isolate that cause. A restoration is a swap, not a free addition.

Repeated A1-control score decreases occur on 36773 (5/5 seeds),39279 (3/5),
36502 and38384 (2/5 each). These are candidates for CPU visual/answer review,
not yet independently established text-evidence mechanisms. No new GPU run is
needed merely to review these saved cases.
