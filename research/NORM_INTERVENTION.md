# High-norm decoder states: bounded TextVQA100 experiment

Use `research/notebooks/kaggle_norm_intervention.ipynb`, the existing
`rq1-ih-inputs-v1` input, and the locked original environment. Push new files,
import the notebook, set `RUN_NORM_INTERVENTION_CALIBRATION = True`. Run the
zero-skip native prerequisites and five-question calibration. Only after its
controls pass, set `RUN_FULL_TEXTVQA = True` in the full-run cell. Download both
ZIP and log. Calibration has 155 predictions; full has 3,100. Estimate time from
calibration rather than treating the earlier runtime estimate as a guarantee.

## What changes

All arms retain 96 input carriers and 32 decoder carriers after block16. There
are 31 conditions: full576; native-drop, FPS→FPS, native-input→E parents;
seven interventions/precision arms; and four control families × five seeds.
Native ApET files and the earlier experiment protocols remain untouched.

- A1/A2: replace the highest-norm one/two retained **output slots** of native-drop.
  Fill with the highest-E pool sources not represented in the parent, stable
  source-index tie break. Controls sample one/two retained basis-assigned slots,
  excluding sources whose norm exceeds 10 times the pool median.
- B1: replace the highest-norm retained FPS→FPS slot. Pick the candidate farthest
  from the remaining retained set using incoming-dtype FPS distance arithmetic;
  exclude every original retained source, including the removed one. Controls
  replace one low-norm retained slot by the same rule. This is a single-point
  continuation from the remaining set, **not** an FPS restart on a different pool.
- C1/C2: insert the highest-norm pool source / all >10×median sources missing
  from native-input→E. Displace the lowest-E retained sources, tie by source index.
  Controls insert one random missing non-high-norm source. Already-present
  sources are not inserted again; the actual swap count is recorded. C2 lacks a
  count-matched multi-insertion control and is secondary/descriptive.
- D: native-drop and FPS→FPS with only decoder FPS distance arithmetic promoted
  to FP32. The same initial RNG draw is used. All subsequent native computations
  remain unchanged. Membership and ordering can both change.

### Important positional choice

Native basis assignment copies source states into different carrier slots. A
fresh sorted union of source indices would change positional assignments as well
as contents. Therefore replacement arms **keep the parent's sorted carrier list
and consecutive position handling**, changing only the selected slots' contents.
They do not re-sort by the incoming source's index. Every record distinguishes
carrier positions, old/new direct sources, and original-image lineage. Thirty-two
unique carrier slots are required; native copying is not assumed to imply 32
unique source identities. These are controlled state-replacement experiments.

## Controls and evidence

Each intervention must have exactly the parent's input encoding and pre-prune
layer16 tensor. Native parent source mapping is checked with `torch.equal`.
All unaffected output slots are checked bitwise. The first question in each run
also checks frozen replay and four-token cache behavior for every custom arm.
The 576-token identity control is retained. Native model tests must pass with
zero skips before inference. Full runs require the same source signature as a
passed five-question calibration. Raw predictions, source pairs, norms, states,
lineage, source code, lockfile, logs and scores are archived.

Interpret changes against matched control means, using paired **image** bootstrap
intervals, not independent arm intervals. Report the top-source padding split
as descriptive. Removal is replacement, and restoration is within a restricted
pool: neither proves general necessity/sufficiency. No attention-backend changes
are made; this cannot establish an attention-sink mechanism. Exact historical
answer reproduction may be affected by previously observed numerical variation.
