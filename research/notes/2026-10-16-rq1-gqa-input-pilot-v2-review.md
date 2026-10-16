# RQ1 GQA input pilot v2 — 2026-10-16

**The full pilot passed: 12 questions on six images, 576 singleton interventions
per question, 6,912 measured patch effects.** The data are complete and usable for
exploratory RQ1 analysis under the stated zeroing/target-rendering protocol.
No repeat of this pilot is needed now. Next: the small MMStar measurement check
and the retained TextVQA generation baseline, followed by cross-dataset analysis.

## Artifacts and verification

- Original 40.8 MiB ZIP:
  `research/results/rq1-gqa-input-pilot-v2-20261016T143422920388Z.zip`.
- Extracted original figures and records:
  [original gallery](../results/rq1-gqa-input-pilot-v2-20261016T143422920388Z/analysis/index.html).
- Separate role-aware CPU review:
  [review gallery](../results/rq1-gqa-input-pilot-v2-20261016T143422920388Z-review/index.html),
  [question table](../results/rq1-gqa-input-pilot-v2-20261016T143422920388Z-review/questions.csv),
  [patch roles](../results/rq1-gqa-input-pilot-v2-20261016T143422920388Z-review/tokens-with-native-roles.csv),
  and [paired-question table](../results/rq1-gqa-input-pilot-v2-20261016T143422920388Z-review/pairs.csv).

All 87 ZIP entries passed CRC validation. Every saved evidence/source digest and
all six cached-feature tensor file hashes were verified. All 12 question files
have 576 unique, finite measured patch scores. The 6,912 exported token rows and
native E values were checked; I arithmetic and all question correlations were
independently recomputed. Original results/ZIP were left unchanged.

Runtime commit `a3842946ccd91c35805c161dca1f23729000d325`, clean Kaggle worktree,
protocol `rq1-input-zero-v2`. The eight monitored native sources match the passed
original-ApET reference. The original dependency lock remains SHA256
`6458cb0ab526857c483b41c19cb87e63179507ebaf6e48f4e9020d18d261f3a2`.

All 12 fresh baseline answers were correct. Fixed target IDs matched generated
prefixes. Repeated references, clone-only no-ops, cached/native forwards and final
references had zero score drift. The maximum generation-vs-teacher-forcing target
score difference was 0.00000472 nats, below 0.001 tolerance.

The first pair repeats all 64 earlier v2 calibration interventions. Their image
features, baseline scores and all repeated I values matched **exactly** across
the two runs. This is useful cross-run evidence for these cases, not a general
claim of CUDA determinism.

Wall time: **4,460.28 seconds = 74.34 minutes**. Peak allocated GPU memory:
**7.43 / 7.27 GiB**. These include instrumentation and teacher-forced full-sequence
forwards; this is not a native generation-speed benchmark.

## Primary result: no consistent positive E–I ordering in this pilot

E is the native pre-projector reconstruction residual; I is the signed change in
mean reference token log probability from zeroing one full-context projected patch
embedding. Larger positive I means a larger decrease in the fixed target's score.

| Question | Target | Archived input-only outcome | Spearman E vs I | Excluding FPS basis sources |
|---|---|---|---:|---:|
| `201636988` cake vs plate color | No | loss | +0.002 | +0.013 |
| `201637316` plate lies on what | Stove | preserved | -0.137 | -0.145 |
| `201713640` medicine-cabinet side | Right | loss | +0.018 | +0.030 |
| `201713599` curtains/trash cans | No | preserved | -0.012 | -0.001 |
| `201428959` refrigerator vs container | Right | loss | -0.073 | -0.078 |
| `201429017` appliance left of refrigerator | Stove | preserved | +0.022 | +0.015 |
| `201882905` device right of curtains | Television | loss | -0.304 | -0.295 |
| `201883221` sinks/mirror/chair | No | preserved | -0.229 | -0.247 |
| `201998268` polished furniture | Table | preserved | -0.211 | -0.215 |
| `201998118` skateboards/cabinets | No | preserved | -0.233 | -0.235 |
| `202218593` paper-towel position | Bottom | preserved | +0.093 | +0.104 |
| `202218829` pan in front of what | Window | preserved | -0.101 | -0.108 |

Eight correlations are negative and four positive. Median per-question rho is
**-0.08684**, range **[-0.30401, +0.09272]**. Excluding ten FPS basis-source
patches per image yields a median **-0.09266**, range [-0.29458, +0.10420]. Thus
near-zero residuals of basis features do not explain away the overall pattern.

The defensible observation is that higher E does not consistently rank patches
by higher positive I in this selected input-stage pilot. This does not rule out
nonlinear relationships, establish an unbiased GQA population result, or prove
that ApET compression is ineffective. Native compression merges/prunes multiple
features jointly; our singleton zeroing proxy is a different intervention.

The review also exports E vs absolute-I correlations for non-basis patches as an
explicitly **auxiliary** influence-magnitude analysis. It does not replace the
signed contribution metric or turn negative effects into answer losses.

## Same image, different questions

All paired questions share exact cached image features and native E values.
The signed-I rank correlations between questions are:

| Image/pair | Between-question sensitivity rho |
|---|---:|
| Cake color / plate on stove | -0.834 |
| Medicine-cabinet side / absence control | +0.513 |
| Refrigerator side / stove identity | +0.573 |
| Television identity / absence control | -0.147 |
| Furniture identity / absence control | +0.319 |
| Paper-towel position / pan-window relation | -0.330 |

There is no single image-only sensitivity ranking that matches all these question
contexts. The evidence establishes question/target-conditioned behavior of this
proxy; wording, answer type, target length and starting confidence also differ,
so it is not an isolated causal estimate of visual task relevance.

### Useful case: paper towels vs pan/window (`n578564`)

For non-basis patch **211** (row 8, column 19), E=10.0653, about the **20th
percentile** among non-basis patches:

- Pan/window question: I=**+0.30548** nats/token. This is its largest positive
  singleton effect, and the original patch is outside the directly retained 278
  original-feature targets of native input compression.
- Paper-towel question: I=**-0.000821** nats/token for the same patch.

Patch **469**, over the paper-towel region in the processed canvas, has E=19.4072
and is directly retained. Its I is +0.03113 for paper-towel position and +0.00919
for pan/window. The 12-question full-run maps show these localized peaks.

Interpretation needs two cautions. First, the full target-prefix probabilities
were 0.98849 for `Bottom` (two target tokens) and 0.27097 for `Window` (one token).
The I scales are affected by target confidence/length as well as the question.
Second, both questions were preserved-correct in archived input-only compression.
Patch 211 may contribute through merging even when its original embedding is not
directly retained. This is a useful low-E/high-I example, not proof that native
compression destroyed the required evidence or caused a wrong answer.

### Cases requiring caution

The refrigerator-side question has strong positive effects on top padded patches
(e.g. patch 21: E=3.6535, I=+0.32563). Padding-region embeddings are contextualized
CLIP representations, so blank pixels do not imply information-free features.
Zeroing the entire projected embedding can affect global representation/position
behavior. Do not interpret every colored square as localized object evidence.

The television question has the strongest negative E–I rho (-0.304), yet its
largest positive singleton effect is patch 215 (I=+0.36821) near the right-side
television region. Most patches have much smaller effects. A single global rho
therefore does not substitute for inspecting sparse peaks and their native roles.
The original automatic exemplar plot does not always label the largest peak;
raw records and role-aware CSVs preserve every patch.

FPS basis-source patches must remain separate: low E is expected for basis
features and native ApET can represent them in replacement slots. Our role table
records basis membership, direct original retention, replacement-slot membership
and selected-mask membership separately. A mask slot and an original source
feature are not interchangeable concepts.

## Next steps

1. Keep this GQA pilot fixed as v2 exploratory evidence; no immediate expansion
   to all 49 questions and no additional GQA rerun is needed.
2. Run the prepared **MMStar 3-question × 32-patch** measurement check. The fixed
   cases are 313 (fine-grained perception, target D), 617 (instance reasoning,
   target A) and 1014 (math, target A). All were reviewed baseline-correct cases;
   fresh baseline/alignment/numeric checks still apply. Their correctness alone
   does not prove visual grounding, especially on multiple-choice questions.
3. Run the existing **TextVQA 100-question baseline + input-only generation** with
   ten-answer consensus scoring. Preserve all 100, including partial-credit and
   baseline-wrong cases. This is not yet TextVQA intervention measurement.
4. After these checks, choose a bounded cross-dataset intervention expansion,
   inspect native-role/contextual-padding controls, and then prepare decoder-layer
   measurement. Do not replace signed I or select favorable examples post hoc
   without explicitly recording the new analysis/protocol decision.

Next MMStar cell in the same live notebook:

```python
mmstar_dir = run_experiment(
    "rq1-mmstar-calibration-v2", "mmstar/manifest.json",
    ["--questions", "3", "--tokens", "32"],
)
```

Then, as a separate run:

```python
textvqa_dir = run_experiment(
    "rq1-textvqa-baseline", "textvqa/manifest.json",
    ["--mode", "textvqa", "--configurations", "baseline", "input-only"],
)
```

These use the already uploaded v1 input bundle, existing v2 inference scripts and
locked environment/checkpoint cache. The CPU review script is new project-owned
analysis only and does not change inference or require a GPU rerun:

```bash
environments/editor/.venv/bin/python scripts/summarize_rq1_pilot.py \
  research/results/rq1-gqa-input-pilot-v2-20261016T143422920388Z \
  --output research/results/rq1-gqa-input-pilot-v2-20261016T143422920388Z-review
```

The review refuses to overwrite an existing output. Keep ZIPs/results separately
from Git, and commit source/notes through the usual workflow.
