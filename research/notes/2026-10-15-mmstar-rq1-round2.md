# MMStar RQ1 review and small extension — 2026-10-15

The supplied human export matches the exact first-round MMStar manifest and all
30 case IDs. It contains 17 Keep, 12 Exclude and one Ambiguous. The 17 Keeps
include five input-only losses, two preserved-correct, five gains and five
both-wrong cases. All decisions and notes are preserved unchanged.

The seven baseline-correct Keeps form the initial primary subset. The other ten
remain in a qualitative appendix; they are not declared invalid questions.
Math and science had no primary cases, and preserved coverage was thin elsewhere.

With seed 592, sampled 12 baseline-correct candidates on new images, then reviewed
six targeted replacements. Direct inspection of all 18 original images adopted
11 additions and recorded four ambiguous, two excluded and one uncertain case.
Rejected examples include white/gray jersey color, turf/grass distinction, static
driving/parking, vague guitarist spacing, demographic-only options, specialist
art attribution and an anatomical pointer the reviewer could not verify.

The resulting primary subset has **18 questions / 18 images**, with six losses
and 12 preserved answers. Category counts are coarse 3, fine-grained 2, instance 4,
logical 3, math 3 and science/technology 3. The separate ten-question appendix
retains five gains and five both-wrong cases from the human Keep decisions.

The new math loss is independently checkable: five pentagon angles sum to 540°,
so x=102 and angle H=x−5=97°. Other additions cover charts, map labels, a flowering
stage, diagram edges, a soccer scene, makeup activity, hydrant-relative position
and image-coordinate localization. New options and source predictions are intact.

The full native baseline is 521/1,500 = **34.73%**, not the supplied advice's
33.9%. The parser finds 1,308 records with four real choices, 84 with two, 92
with three and 16 whose options are not parsed by the helper. A blanket 25%
guessing comparison is therefore unsuitable. Baseline A predictions remain
1,059/1,500; the primary set's gold distribution is A=10, B=4, C=2, D=2.

The source bird-label typo and two-choice colorful-image example are flagged for
sensitivity checks. Correct multiple-choice output alone does not establish
image grounding or isolate reasoning from perception. This selection is
conditional and unblinded, and assistant decisions are not independent human
validation. Limited question-only and option-order checks are appropriate in
the eventual intervention pilot, alongside no-op/reference repeats.

Artifacts: `research/results/rq1-mmstar-reviewed-v1-s592/index.html`, qualitative
and complete-audit galleries, manifests and an adjacent ~4.42 MiB ZIP. Nine CPU
curation tests passed, covering MMStar and the shared GQA renderer. Image and
source hashes, gallery links/embedded images, ZIP contents and native-source
invariance were verified. No GPU inference, environment or native algorithm change.
No further human review is requested for this round.

See [selection rules and rebuilding instructions](../MMSTAR_RQ1_ROUND2.md).
