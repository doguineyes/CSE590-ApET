# TextVQA feature and OCR evidence audit — November 6, 2026

Implemented and ran the CPU preparation/analysis on the complete100-image seed run.
No model inference, new dependencies, or native model changes. Original archives are
preserved. Use the `textvqa-evidence-cpu-v1-reviewed` derived directory as the current
review; earlier derived directories are preliminary outputs, not additional data.

## Deliverables

- Official Rosetta0.2 validation OCR downloaded, matched to all100 image IDs, archived
  with URL and SHA256. Source metadata: `textvqa_rosetta_ocr`, `val`,0.2.
-100 image/patch property maps from saved pixels and raw CLIP/native layer16 tensors.
  Tensor loading uses a restricted CPU-to-NumPy reader, without installing Torch;
  every CLIP tensor reproduces the saved native feature fingerprint.
-100-case gallery of OCR candidate overlays on the actual padded model canvas.
-20 deterministic matched-candidate visual audits, independent of model outcomes.
-Per-seed coverage and outcome CSV, per-image properties, summary and sensitivity
  tables. These are derived observations, not changes to any experiment result.

Read `research/TEXTVQA_EVIDENCE_CPU.md` for definitions and reproduction.

## Measurement quality

32 unique OCR candidates;4 repeated/multiple matches;3 multiword candidates;61
unmatched. Unique means one OCR detection, not guaranteed unique semantic evidence.
Only the32 unique candidates enter provisional coverage statistics. All100 remain
in performance/property summaries. Failure to find a box is never scored as zero
coverage. This matching subset can overrepresent easily OCR-readable examples.

Visual inspection of20 matched candidates found box/grid alignment plausible.
Four small-text overlays were also inspected at full size. This is geometry review,
not a claim that every tiny transcript is independently readable. Two unique
candidates have clear caveats:36995 asks for one of several player numbers;35802
has another visible TaB occurrence that OCR did not match. They stay in the primary
automatic32 table; a labeled stricter sensitivity table excludes these two.
Repeated Brolio labels remain ambiguous; no preferred bottle was selected after
looking at model outcomes. The remaining12 unique candidates were not individually
visually audited in this round.

## Property audit (all100; medians of per-image Spearman correlations)

| Comparison | All patches | Fully nonpadding patches |
|---|---:|---:|
| E vs feature norm | −0.513 | −0.210 |
| E vs pixel detail | +0.490 | +0.262 |

E vs nonpadding fraction: median+0.482 across91 nonsquare images (undefined for
9 square images). Excluding FPS basis tokens does not remove the broad signs:
E–norm−0.504, E–detail+0.521. Native layer16 E–norm median+0.112.

At stage1, random kept slots have mean geometric padding fraction27.79%; native
slots7.42%. Accounting for basis source replacement, native direct-source locations
have8.90% padding. Thus native avoids substantial padding but does not eliminate it.
Slot and direct-source statistics are deliberately distinct. Direct-source sets
are unique source locations, not group-size- or attention-weighted information.

These results make padding avoidance and detail selection credible hypotheses to
test. They do not support the simple claim that input E is merely a positive
feature-norm ranking. Padding explains part of the raw correlations: associations
shrink within fully nonpadding patches. No causal attribution of the score gap yet.

## Provisional within-question coverage analysis

Among32 unique candidates,15 questions vary in full-credit correctness across20
random masks. Compare each question's mean coverage in correct versus wrong seeds,
then average questions equally:

| Coverage proxy | Correct-minus-wrong mean | Paired image bootstrap95% | Positive/negative questions |
|---|---:|---:|---:|
| Stage1 positions | +1.56 percentage points | [−6.27,+7.55] |11/4 |
| Final positions | +7.45 percentage points | [+4.15,+10.51] |14/1 |

After excluding the two visually flagged examples:30 candidates,13 varying
questions; stage1+1.97 points [−7.24,+8.69], final+7.88 [+4.00,+11.35],12/1 positive/
negative for final positions. This check preserves the general final-position pattern.
Intervals condition on these20 seeds and the matched development cases, not the
whole TextVQA distribution. Seeds are not independent questions.

Native direct-source coverage exceeds random mean by24.39 points at stage1 and
14.57 points at the final boundary among32 candidates. Native input merge paths
reach nearly all matched-region positions on average (~97.9% when computed across
all39 matched candidates), illustrating why a binary "merged in" metric alone can
be uninformative. Path existence does not establish usable evidence or its weight.

The final-position association is a useful lead, not a claim that layer16 alone
causes success: stage1 and stage2 masks both vary; intermediate states already
contain context, and OCR boxes are incomplete. Do not interpret the stage1 interval
as proof that stage1 evidence does not matter.

## Next decision

No immediate GPU rerun required. Review candidate coverage quality and the property
audit first. The next controlled selector experiment should include nonpadding
random and detail selection as rival explanations, retaining an exact native arm
and matching budgets/compensation. Norm remains a useful comparison but should not
be described as an established explanation. Do not expand the cohort just to make
this provisional pattern stronger.

Validation:8 CPU tests passed (padding/odd dimensions, boundaries, detail definition,
ambiguous/no-fuzzy matching, basis/merge lineage, rank ties, tensor strides/bounds,
and forbidden pickle globals). All100 feature maps generated and byte fingerprints
checked;20 overlays and one property-map figure visually inspected.
