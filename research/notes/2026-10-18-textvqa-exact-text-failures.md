# TextVQA: compression changes literal text — 2026-10-18

The user's Brolio observation is confirmed in the raw paired predictions and
images. This is a concrete answer-preservation failure pattern worth featuring
in the RQ1 report: **native input ApET can change correctly read printed text
into an incorrect word or related phrase.** The previous consolidation placed
too much emphasis on aggregate E–I relationships and too little on these actual
generation outcomes.

## Observed examples

The saved comparison uses the same model, image, prompt and seed: uncompressed
576 visual tokens versus native input-only ApET at 288, basis count 10, with no
decoder pruning. No new inference was performed for this review.

| ID / question | Printed/reference text | Baseline | Input ApET | Consensus score |
|---|---|---|---|---:|
| 37533 / wine name | BROLIO | Brolio | Bottoli | 1 → 0 |
| 35962 / school name | cookery school | Cookery school | Culinary school | 1 → 0 |
| 36001 / instruction on posters | give up | Give up | Gone | 1 → 0 |
| 38355 / wine name | CHÂTEAU PONTET-CANET | Chateau pontet-canet | Chateau pontet canne | 0.9 → 0 |

The first three originals and actual saved 336×336 model canvases were inspected
in this turn. BROLIO appears on repeated bottle labels; cookery school appears
on the wall and aprons; give up appears on the posters. These wrong strings
are not merely casing, formatting or an equivalent spelling of the shown answer.
For the school-name question, a related generic phrase does not preserve the
printed name.

The fourth original image was also inspected: PONTET-CANET is printed clearly
on its label. It is supporting spelling-loss evidence, with varying reference
spellings and only partial baseline credit. It is not counted as a fourth
full-credit baseline loss. Its actual saved 336-pixel canvas was not available
in the six-case intervention exports, so model-scale legibility is not claimed
from its original alone.

These have low reasoning demands, but that does not make their visual evidence
large or easy at model resolution. Brolio's repeated letters are very small on
the 336-pixel canvas. The full model's saved correct response establishes that
it could answer this particular case under the recorded setup.

## Counts, controls and scope

Rechecking all 200 raw prediction records confirms 100 exact baseline/compressed
pairs, with identical paired questions, gold answers, image hashes, prompts and
seeds. Of 44 full-credit baseline cases, 41 remain full-credit and the three
above are lost: **3/44 = 6.82%**, or 93.18% full-credit preservation. This is a
100-question screening cohort, not a full TextVQA population estimate.

Across all 100 pairs, 23 normalized answers change, five consensus scores decrease
and two increase. The other score decreases are the partial-credit wine name and
the ambiguous Princess/Woman question. That ambiguous question is not promoted
to a clean text-reading example.

Compression also preserves Dominion, Sky mall and Laughter, and changes Um → Umd
on a uniform abbreviation, increasing its score from 0 to 1. Monday night → Monday
also increases the score, with reference-wording effects. Thus the observed
pattern is not universal text destruction, and both losses and gains remain
visible in the review.

## What the existing native records add

For Brolio, state **317** lies near a wine label and has the largest positive
whole-answer singleton-zeroing sensitivity, about **+0.12858 nats/token**.
Its E is about **17.5662**, around the 73rd nonbasis percentile.

Importantly, it is **selected, is not an FPS basis source or replacement slot,
and serves as a native merge target**. The saved merge records show three
incoming source indices: **103, 109 and 331**. With native scaling 1, the
pre-projector state is averaged with those three source features. The answer
therefore becomes Bottoli even though this original source is included in the
retained target set; retaining a source coordinate does not mean its vector
is unchanged.

Other existing records offer different candidates:

- Cookery school states 136/137 are not directly retained original targets;
  they merge into 104/138. State 149 is an FPS basis source carried into output
  slot 397, as well as being a source in the recorded merge graph.
- Give up state 82 has lower-quartile E and positive whole-answer zeroing
  sensitivity; it merges into target 108 rather than being directly retained.

These mappings establish native operations on the measured sources. They do
**not** establish which operation caused the wrong spelling or phrase. The
features are contextualized, multiple states change together, and native
compression also shortens the sequence. The facts motivate a controlled
merge/no-merge experiment; they do not prove that averaging blurred specific
letters or that removing a particular low-E state caused the answer loss.

## Interpretation for the project

We can now describe one observed condition where ApET fails: **questions that
require preserving literal text identity can lose correctness after compression,
even when the baseline returns the printed word correctly**. The compressed
output may still be word-like or semantically related. We did not independently
measure whether the model retained the overall scene meaning, so that broader
claim should not be inferred from these single questions.

This adds actual answer-level evidence to RQ1's representation/proxy analysis.
It is consistent with the proposal's fine-detail concern. It does not yet
explain why a given failure happens, predict failure for unseen inputs, or show
that these examples' text necessarily has low E. Brolio itself cautions against
equating text loss with simply dropping the highest-sensitivity source.

Use Brolio, Cookery school and Give up plus the three preserved TextVQA controls
in the first RQ2 diagnostic pilot. Compare original native input compression
with merging disabled at the same 288-token budget and identical selected set,
FPS sources, source-to-slot basis mapping, prompt and generation policy. If
no-merge restores a literal answer, that would support a merge-specific account
for that fixed configuration; if it does not, the error may require further
selection/basis/context analysis. No additional GPU run is necessary merely to
document the currently observed failures.

## Saved review

The [focused gallery](../results/rq1-textvqa-word-errors-review-v1-2026-10-18/index.html)
contains all five score-loss cases, both score gains and three preserved controls,
with original images, raw answers, reference counts and available saved model
canvases. Adjacent JSON/CSV preserve all 100 pairs and the native-role evidence.
Images are copied unchanged; raw results and native/measurement code are untouched.

Original generation ZIP SHA256:
`ebf1cfc318eeb9e712452fd71efce6ee69516faaed8c7d4b4b8f1d948715902b`.
The archived byte hash and all paired identities were checked again. Native role
records come from the already verified canonical intervention ZIP
`rq1-textvqa-input-pilot-v1-20261016T204945039325Z.zip`, SHA256
`cf37506d9a3c087e60db409de57feb3691c813b2e3dd2cdc086de0e187ff99ea`,
using its verified reference copies in the input-IH synthesis. Per-record hashes
are retained in the new gallery's `native-role-evidence.json`.
