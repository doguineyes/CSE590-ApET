# RQ1 input IH full sweep — 2026-10-18

The full sweep passed: **30 questions, 24 images, 17,280 new input-stage IH
effects**, joined to the same archived E/zeroing records. Runtime was **31.1
minutes** on T4 ×2. No measurement fix or further RQ1 GPU run is required now.
The GPU can be stopped. This completes the planned bounded input-IH extension,
not every possible task/layer study in the proposal.

## Preserved evidence and validation

Both supplied files were copied byte-for-byte to `research/results/`; their
Downloads originals remain intact:

- `rq1-input-ih-thirtycases-v1-20261018T140840768523Z.zip`: 10,402,603 bytes;
  SHA256 `724a0e13ad5db7ab025e0f11cb0f96ecee377bdbb38719652cc273f53e68f0ef`.
- Matching `.log`: 17,103 bytes;
  SHA256 `cec0d3331fd345566f4c5f7e2ba6fe7fdda6f4c64fc0daaa56447aef7e87faaa`.

All 209 unique ZIP members pass CRC, all 102 evidence hashes and 24 fresh feature
metadata hashes verify. Runtime commit `967aa230b2df8bd9aec440d4baec0ed2e9682a34`
was clean. Native ApET, measurement sources, original lock, accepted manifests,
reference index and feature/E/basis identities match their pinned references.
No source algorithm, dependency or historical result was changed for this review.

Independent CPU checks validate all raw score arithmetic, retained positions,
sequence geometry, fresh correct baseline answers and fixed target prefixes.
All **90 fresh zeroing audit scores** match the archived scores exactly. The
full run embeds the passed six-case input calibration and its detailed records;
all **192 calibration IH scores and null references repeat exactly**. Retain-all,
null-repeat, retained-repeat, final-reference and unchanged-input controls have
zero recorded drift. Exact controls do not establish unlimited numerical
precision for every tiny effect.

There is no missing companion for this supplied ZIP/log pair. Original feature
tensors remain in the canonical older ZIPs; this run records their matching
metadata rather than uploading redundant tensor contents. Results remain Git
ignored and need a separate backup.

The independent archive review is in
`results/rq1-input-ih-full-review-v1-2026-10-18/`. The combined review adds an
[interactive gallery](../results/rq1-input-ih-synthesis-v1-2026-10-18/synthesis.html),
[per-question CSV](../results/rq1-input-ih-synthesis-v1-2026-10-18/question-synthesis.csv),
[same-image pairs](../results/rq1-input-ih-synthesis-v1-2026-10-18/same-image-pairs.csv),
and [matched depth comparison](../results/rq1-input-ih-synthesis-v1-2026-10-18/depth-comparison.csv).
These are derived outputs, not additional observations. The gallery's 30 canvases,
17,280 data rows and JavaScript syntax were checked; browser rendering was not
previewed during this review.

All six focused CPU archive/synthesis regression tests passed, including raw
preservation, full/cohort denominators, source gates, matched layer input scores,
the sparse Dominion tail and Cookery's basis-source label. `git diff --check`
also passed. These are offline verification checks; pretrained execution was
the supplied Kaggle run.

## Matched measurements

The two primary I columns use the **same first frozen gold tokenizer-piece
probability units**:

```text
I_zero = p(first gold piece | full) − p(first gold piece | zero one state)
I_IH   = p(first gold piece | retain one state) − p(first gold piece | remove all)
```

This is the documented source-informed **1×1 IH adaptation**, not an exact
reproduction of IH's original benchmark/default regions. Input E is native
pre-projector CLIP reconstruction error; interventions act on corresponding
projected LLaVA inputs. The whole-answer zeroing log score remains preserved
separately. Neither I measures intrinsic token importance or generated full-answer
accuracy. Retention also changes sequence context and length, whereas zeroing
preserves shape, so operator and retained-context effects are not isolated.

| Dataset | Questions / images | Median rho(E, zero) | Median rho(E, IH) | Median rho(zero, IH) | Positive E–IH |
|---|---:|---:|---:|---:|---:|
| GQA | 12 / 6 | −0.087 | −0.068 | −0.016 | 6/12 |
| MMStar | 12 / 12 | +0.009 | −0.121 | −0.005 | 4/12 |
| TextVQA | 6 / 6 | −0.031 | −0.136 | −0.028 | 1/6 |

These are medians of per-question correlations, not a correlation on pooled
tokens. Across questions E–IH ranges from **−0.486 to +0.404**, with 11 positive
and 19 negative values. The conclusion is **mixed, question-dependent alignment**,
not universal near-zero correlation or universal negative alignment. This curated
cohort, repeated same-image questions and correlated visual states do not support
population significance claims from sign counts or 17,280 nominal observations.

FPS basis and padding sensitivity analyses matter. Excluding basis sources gives
median E–IH of −0.044 / −0.115 / −0.117 for GQA/MMStar/TextVQA. Restricting further
to photo-only states gives **+0.087 / −0.065 / −0.039**: GQA's median changes sign.
Thus filtering does change some summaries; it still does not reveal consistently
strong positive alignment. Padding states have contextualized features and are
not automatically irrelevant. Correlating E with absolute IH also asks a
different question; its dataset medians are +0.008 / +0.113 / +0.109.

All 12 MMStar questions remain primary. Existing grounding flags (1014, 1030,
1188, 776, 1372) and prior quality flags (604, 617) remain explicit. Excluding
their union leaves only five cases, with median E–IH −0.102; this post-hoc check
does not establish a clean, representative five-question benchmark.

## Strong states missed by calibration

Full sweeps matter even when the 32 measured calibration positions repeat
perfectly. Best retained first-piece probabilities change as follows:

| Target | Best of calibration 32 | Best of all 576 |
|---|---:|---:|
| Stove | 11.10% | 31.09% |
| Brolio | 0.002865% | 1.0584% |
| Dominion | 0.001500% | **20.188%** |

The input Dominion maximum is 20.2%, not the 98.1% recorded at layer 16. Its
full-grid E–IH rho is +0.288, while its median absolute IH effect is only about
0.000000504 probability units: a small median can coexist with a sparse large
tail. Sky mall similarly has only five states with IH greater than 0.1, yet its
best retained first-piece probability is 81.70% (full reference 94.75%). Global
rank statistics should therefore accompany effect sizes and visual examples.

Useful gallery cases:

- **Stove, GQA 201637316, state 191:** IH +0.31022, zeroing −0.00815.
  Retaining a state alone can help relative to the null even when zeroing it
  from the full context slightly increases the target probability. These are
  different counterfactual questions, not contradictory importance labels.
- **Cookery school, TextVQA 35962, state 149:** best retained probability
  19.45%; E about 0.00000709; zeroing +0.15206. This is an **FPS basis source**,
  not an ordinary low-E redundant state. It must not be assigned a nonbasis E
  percentile or treated as evidence that ApET simply discards the information.
- **Sky mall, TextVQA 39323, state 522:** IH +0.81667 but zeroing only +0.00267;
  E is at about the 87th nonbasis percentile. It helps illustrate conditional
  necessity versus retention-based support without deciding either is ground truth.

Each retained state is already globally contextualized by CLIP. A high retained
first-piece probability does not show that one raw image patch is sufficient to
answer. The first piece can also be semantically partial (for example Laughter
starts with `▁L`). No full-answer intervention generation was measured here.

## Same image and representation depth

All six GQA pairs have exactly identical image features and E. Their IH-map
rank agreements are **−0.284, −0.443, +0.655, +0.161, −0.401, −0.234** in fixed
protocol order. Thus the same visual ranking can accompany similar or different
question/target-conditioned maps. Targets and reference confidence also differ;
this is not a controlled causal manipulation of wording alone.

The earlier six-case layer full ZIP is pinned and checked again. All **3,456
archived input zeroing scores** and full target references/features agree exactly
with the current input records. Matched IH results are:

| Target | Input rho(E, IH) | Layer-16 rho(E, IH) | Input/layer IH-map rho |
|---|---:|---:|---:|
| Window | +0.231 | −0.334 | +0.067 |
| Bottom | −0.299 | −0.110 | +0.198 |
| Bedroom | −0.102 | −0.101 | −0.038 |
| Soccer | −0.152 | −0.094 | +0.042 |
| Cookery school | −0.087 | −0.003 | +0.012 |
| Dominion | +0.288 | −0.099 | +0.131 |

There is no consistent improvement of E alignment at layer 16. Only two
boundaries were studied, not a full information-horizon curve. Input E is
image-conditioned. In this runner, visual tokens precede the question in a
causal decoder, so their layer-16 states cannot attend to the later question
either. Different whole-sequence shapes can yield numerical differences; the
observed same-image layer E agreement is nearly exact. This corrects the earlier
statement that processing the prompt makes these visual states question-aware.
Answer sensitivity remains question/target-conditioned. Late nulls retain text states
that have already seen the full image; different null probabilities and
intervention contexts prevent attributing every cross-stage difference to depth
alone. The input null preserves the original visual-position gap and is not an
ordinary question-only prompt either.

## Next work

1. Review a few gallery cases and consolidate the bounded RQ1 result: E has
   inconsistent association with our two answer-sensitivity measures; the maps
   depend on question, counterfactual and boundary. This does not establish E
   is useless, that IH is ground-truth importance, or that compression is bad.
2. Move to a small **RQ2 input-only merge ablation** at the existing 288-token
   budget: native ApET versus the same FPS sources, E-selected positions and
   basis-replacement assignment with only merging disabled. Keep seed, model,
   prompts, precision and output settings fixed. Start with parity/no-op controls
   and a diagnostic generation pilot before GQA-development-800/MMStar-1500.
3. Record source-to-output basis mapping separately from merge destinations.
   Native `llava_arch.py` replaces dedicated selected slots with FPS features
   before merging; absent original source positions do not imply absent basis
   information. Disabling basis replacement as well would confound two factors.
   In combined compression, changing input merge can change downstream decoder
   selection, so input-only is the clean first ablation.

No additional broad RQ1 GPU sweep is required to take these steps. Later optional
checks should address a stated unresolved question, rather than rerunning because
correlations are weak. RQ2 should compare actual generation/preservation too;
gold-answer IH-guided protection would be an oracle diagnostic, not a deployable
question-only compression method. RQ2 implementation is a proposed next task;
it was not added to the native baseline during this analysis.

Reproduce the offline synthesis into a **new** output directory:

```bash
environments/editor/.venv/bin/python -X utf8 scripts/summarize_rq1_input_ih_full.py \
  --output research/results/rq1-input-ih-synthesis-new-review
```

This uses only saved ZIP records and images; no Torch installation, model
inference or tensor deserialization is required on the Mac.
