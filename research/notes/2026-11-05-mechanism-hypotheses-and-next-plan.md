# What is ApET actually selecting? Mechanism hypotheses and next-stage plan — 2026-11-05

This note records the November 5 rereading of the ApET paper and code, the discussion that followed, and the plan for the remaining experiment window (about two to three weeks of testing before report writing). It is a planning note, not a frozen protocol. No new model experiment was run for it. Section 1 lists code facts checked against this repository. Sections 2–3 record the conceptual discussion. Section 4 states the new main question and hypotheses. Section 5 gives the two experiments to run first in detail. Section 6 lists later or conditional experiments as short ideas.

## 1. What the code actually does (checked against this repository)

Sources: `llava/model/llava_arch.py` (`encode_images`, `token_merging`) and `llava/model/modeling_llama_x.py` (`layer_prune`, module-level `token_merging`), `llava/model/utils.py` (`fps`).

| Step | What the code does | Not stated in the paper |
|---|---|---|
| Stage-1 location | Runs on the 576 CLIP penultimate patch features (1024-d) **before** the MLP projector | The paper says "after the vision encoder"; merged averages then pass through the nonlinear projector |
| Basis | 10 tokens by FPS with squared Euclidean distance; the start index is `torch.randint` (seed-dependent) | — |
| Reconstruction | Ridge-like normal equations, `G = BBᵀ + 1e-5·I`, solved in fp32 on **raw, unnormalized** features; `E = ‖x − x̂‖₂` | No feature normalization; stage 2 uses unnormalized layer-16 residual-stream states |
| Keep set | `topk(E, K)` | — |
| Basis placement | Basis features **overwrite the 10 lowest-E slots of the top-K**. Those 10 tokens' own content is discarded, not merged | Basis vectors sit at other tokens' positions |
| Merge targets | Only the K−10 ordinary retained tokens (`errors_indices[:-k]`). **Dropped tokens can never merge into the basis slots** | Answers our open question "do dropped tokens merge into B?": not by construction |
| Merge rule | Each dropped token goes to the most cosine-similar ordinary target; target becomes `(target + Σ sources) / (1 + n)` | Unweighted average, target weight 1: a target absorbing 25 sources keeps about 4% of itself. **No group size is tracked**, so no proportional attention (unlike ToMe) |
| Original basis tokens | Their E is near zero, so they are usually non-retained and are merged into ordinary targets as well | — |
| Positions | Kept tokens stay in original order; position IDs are re-indexed consecutively after each stage (decoding adjusts by the removed count) | At layers 17–32 text tokens move about 64 positions closer to the image (96→32) |

Hardware note: T4 GPUs do not support FlashAttention-2, so our runs use eager/SDPA attention. Adding a per-key logit bias (for example log of group size) is therefore straightforward in our setup.

## 2. Discussion record: why can a linear, geometric rule work inside the LLM?

### 2.1 Recoverability is not usability

The paper's argument (Eq. 1–3) is about **recoverability**: dropped information remains *linearly recoverable* from the kept set S, so I(V;S) stays high. Mutual information only says *some* decoder could recover it. The frozen LLM never computes `x̂ = Σ αᵢbᵢ`.

The partial link: keys and values are linear in the (RMS-normalized) token, so a dropped token's value lies in the span of the kept values. This breaks down because:

- **Softmax gives a convex mix, not a span.** After dropping, one attention head outputs `Σ pⱼvⱼ` with `pⱼ ≥ 0`, `Σ pⱼ = 1`, weights set by `q·k`. Least-squares α are unconstrained (signed, not summing to 1), so many low-E tokens are in the span but **outside the convex hull** of kept values and cannot be produced by any attention weights.
- **Exact effect of dropping token d** on one head's output for one query: `out − out′ = p_d (v_d − out′)`. Damage depends on attention received (`p_d`, query-dependent) and on distance to what the head outputs anyway. E uses neither.
- **Attention mass.** n near-duplicates jointly receive about n times the weight of one. Keeping one copy preserves content but loses weight; ToMe restores it by adding `log n` to that key's logit. ApET's plain average does not, so merged background groups are effectively downweighted. This could hurt (counting, area) or act as denoising.
- **RMSNorm is scale-invariant**: x and 3x look identical to the layer, yet raw-space E treats high-norm tokens as hard to reconstruct.
- **Depth and positions**: MLPs and many softmax layers break linearity; RoPE positions change after re-indexing.
- **Stage 2 is cheap for a different reason** (FastV): by layer 16 the text tokens have already read all visual tokens in layers 1–16.

**Reframe adopted:** the reconstruction step is only a **scoring device**. Its effect on the LLM is entirely through *which* tokens survive (and what survivors look like after averaging). The scientific question is therefore not "how does the LLM decode the reconstruction" but **"what do high-E tokens contain, and is that what the answer needs?"**

### 2.2 Causal mask: image tokens never see the question

In LLaVA-1.5 the sequence is `[system] [576 image tokens] [question] [answer]` with a causal mask, in training and inference. Image-token hidden states are identical for every question at every layer; information flows only from image tokens to text tokens. So both ApET stages are question-independent; text-guided methods get question awareness only from text tokens' attention. A same-image, two-question hidden-state comparison at layer 16 is a cheap pipeline sanity check.

### 2.3 Spatial information

Position reaches the LLM through CLIP's learned position embeddings (inside the features) and through 1D RoPE order. Merging can average tokens from far-apart patches and gives the group one position; pruning leaves gaps and re-indexes. Kept as a secondary concern: measure merge-group spatial spread from saved routing before adding any position experiment.

### 2.4 What earlier results already suggest

- ApET beats pure random strongly on TextVQA (+20.9 at 96→32), weakly on GQA (+2.5), not on MMStar; the advantage comes mainly from **fewer lost answers**.
- E does not rank single-token answer sensitivity (RQ1). Single-token ablation cannot see redundancy, so this fits a **set-level coverage** effect rather than proving E is useless.
- Shuffling merge assignments with each target's load fixed barely changed correctness (19–20 vs. 19/30), and the lowest-E retained quarter absorbs 66.1% of merges. Merge *details* may matter little, but heavy averaging can corrupt a retained token (Brolio→Bottoli).

## 3. Working hypothesis and its rivals

**Working hypothesis (H-detail): ApET is a detail and coverage detector.** FPS picks a few extreme directions; tokens far from their span are unusual, high-detail patches (text, edges, small objects); smooth regions get low E. Text answers live in a few small, detailed patches that random selection often misses, which would explain the task pattern.

Two simpler rivals explain the same pattern and must be ruled in or out:

- **H-pad (padding avoidance).** LLaVA-1.5 pads non-square images to a square with the image-mean colour. Padding patches are perfectly smooth, get low E, and are never kept by ApET, while random spends part of its budget on them. TextVQA images are often non-square.
- **H-norm (norm detection).** Since `E² = ‖x‖² − ‖proj_B x‖²` with only 10 basis vectors, E may mostly track the feature norm ‖x‖₂ (vector length). Large ViTs develop a few high-norm "artifact" tokens (about 2%, roughly 10× norm) on low-information background patches that store global information (Darcet et al., 2024). Direction is unclear: high norm raises E, but FPS chooses far-away points first and may pick these outliers as basis vectors; if they resemble one another, one basis vector spans them all and they get *low* E. This must be measured, not assumed.

These hypotheses are not exclusive; the goal is to estimate how much of the ApET–random gap each explains.

## 4. Main question for the next stage

> **What property of the retained token set predicts whether LLaVA still answers correctly, and does ApET's reconstruction error select for it?**

A **property** P is a number computable for any retained set, whichever method chose it. The question is a mediation question:

```
selection rule (E, random, …)  →  property P of the kept set  →  answer correct?
```

1. **Right arrow:** does P predict correctness, whatever method produced the set?
2. **Left arrow:** do ApET's sets score higher on P than random sets?
3. **Mediation:** does P account for most of the ApET–random gap?
4. **Sufficiency:** does a simple selector that targets P directly, without reconstruction, match ApET **through the same P** (similar coverage, similar losses/gains)?
5. **Necessity:** does manipulating P inside ApET's selection change answers as predicted?

Candidate properties:

| Property P | Computed as | Tests |
|---|---|---|
| Evidence coverage | Fraction of answer-region patches kept (TextVQA: OCR boxes of the answer, mapped to the 24×24 grid) | The model fails when evidence is removed |
| Detail coverage | Share of image patch-detail energy on kept patches | H-detail |
| Padding share | Fraction of kept tokens that are padding | H-pad |
| Norm mass | Share of total feature norm on kept tokens | H-norm |
| Set reconstruction error | Least-squares error of all 576 tokens from the kept set | The paper's information story |

Evidence coverage uses the answer and is analysis-only. The other properties are inference-time computable and could become a method. The ideal result links them: evidence coverage explains correctness, and an inference-time property predicts evidence coverage.

**Merge question (kept separate from selection):** with selection fixed, does merging add useful gist from dropped regions, or does it dilute retained evidence?

## 5. First two experiments (detailed)

### Step 0 — preparation (GPU minutes + CPU)

- **Feature dump** for every TextVQA100 and GQA800 image: preprocessed 336×336 pixel tensor, CLIP penultimate features (576×1024, fp16), and for the native run, stage-1 E, FPS basis indices, and the 96 layer-16 states with their E. All later property computations and selector prototypes then run on CPU.
- **Cohort extension (optional, decide before seeing outcomes):** freeze about 200 additional unused TextVQA development questions; run uncompressed, native ApET and pure-random seeds 1–3 on them, giving TextVQA300.
- **Answer regions (CPU):** match the normalized answer string to OCR word boxes (verify the OCR file and box format for our split); map boxes through LLaVA's pad-to-square and resize to the 24×24 grid; mark questions with no OCR match and analyse them separately. Audit about 20 mappings visually (reuse `scripts/render_rq1_gallery.py`).
- **Patch maps (CPU):** padding fraction per patch (padding pixels are about 0 after normalization), detail energy per patch (mean |∂x|+|∂y| of the normalized pixels within each 14×14 patch), feature norm per patch.

### Experiment 1 (GPU first, no new code): random-seed natural experiment for evidence coverage

**Purpose.** Test the right arrow: does evidence coverage predict correctness when coverage is assigned **at random**? Random selection gives the same question different coverage by chance across seeds, so within-question comparisons are not confounded by question difficulty or by anything ApET-specific. This is the strongest single piece of mechanism evidence available to us.

**What to run.** The existing pure-random pipeline (no basis, no E, no merging; random at both stages; 96→32), seeds 4–20 on TextVQA100 (seeds 1–3 exist). Archive stage-1 and stage-2 selected indices; map stage-2 survivors back to original patches. If Step 0's cohort extension is done, run the same 20 seeds on the new 200 questions after the first 100 are finished.

**Measures.**
- `coverage_s1 = |kept_stage1 ∩ A| / |A|` and `coverage_final` (stage-2 survivors mapped back), where A is the answer region.
- Full-credit correctness and the ten-annotator score.
- For comparison (no new runs): the same coverage for native ApET, counting *kept* and, separately, *merged-in* answer patches (dropped but merged into a kept token).

**Analysis.**
1. Dose–response: P(correct) by coverage bin, pooled over seeds and questions.
2. Within-question test: logistic regression `correct ~ coverage + question fixed effect` (conditional logit), with an image-level bootstrap. Simple companion: for each question whose correctness varies across seeds, compare mean coverage of correct and wrong seeds; sign test across questions.
3. Left arrow (free): paired per question, ApET coverage vs. the random-seed mean; is the difference larger on questions ApET preserves and random loses?

**Expected results and interpretation.**

| Outcome | Interpretation |
|---|---|
| Strong positive slope within question; zero-coverage seeds rarely correct | Evidence coverage is a real mediator candidate; proceed to the Experiment 2 mediation |
| Positive slope, but ApET coverage ≈ random coverage | Coverage matters, but ApET's advantage comes from elsewhere (for example the merged-in path or padding) |
| Flat slope | Answer survival under random pruning does not depend on keeping the answer region (redundancy, priors, or OCR-like global features); H-detail loses its main support |

**Supports.** Links 1 and 2 of the main question.

**Checks.** Seeds 1–3 reproduce the archived predictions; identity (576) control; exact token counts; report how many questions lack an OCR answer match.

**Cost.** 17 seeds × 100 questions on the existing runner; scale from the measured TextVQA100 runtime.

### Experiment 2 (GPU, small code change): selector × compensation grid

**Purpose.** Discriminate H-detail, H-pad and H-norm; test sufficiency of simple selectors; and separate selection from merging.

**Fixed design (all arms).**
- Native 96→32 schedule. **Only the stage-1 ordinary selector varies**; stage 2 always runs the native ApET rule on whatever arrives (a realistic total effect; properties are measured at both stages).
- The same 10 FPS basis tokens (fixed FPS start seed); basis indices excluded from ordinary candidates; native slot rule in every arm: take top-K by score, overwrite the 10 lowest-scoring kept slots with the basis features. Native position handling.
- Compensation factor: **drop** (skip `token_merging`) vs. **native merge**.

**Arms.**

| Ordinary selector | Hypothesis | drop | merge |
|---|---|---|---|
| ApET E | reference | ✓ | ✓ (native) |
| Random (basis fixed), 5 seeds | baseline | ✓ | ✓ |
| Random over non-padding patches, 5 seeds | H-pad | ✓ | ✓ |
| Detail energy top-K | H-detail | ✓ | ✓ |
| Feature norm top-K | H-norm | ✓ | ✓ |
| *Optional:* FPS for all K | feature-space coverage | ✓ | |

**Implementation sketch** (`encode_images`, after the FPS basis is chosen):

```python
score = {
    "apet": errors,                                   # native
    "norm": image_features.norm(dim=-1),
    "detail": patch_detail(pixels),                   # mean |dx|+|dy| per 14x14 patch
    "nonpad_random": rand.masked_fill(pad_mask, float("-inf")),
    "random": rand,
}[selector]
score[:, basis_idx] = float("-inf")                   # basis never chosen as ordinary
errors_indices = torch.topk(score, k=K, dim=1)[1]
# ... unchanged native basis overwrite of errors_indices[:, -k:] ...
if compensation == "merge":
    image_features = self.token_merging(image_features, errors_indices, k)
```

**Measures.** Score, full-credit accuracy, APR, losses/gains vs. uncompressed, paired image-level bootstrap intervals vs. native; for every arm the properties of Section 4 (evidence coverage kept/merged-in, padding share, detail coverage, norm mass); number of selections differing from native.

**Predictions.**

| | E correlates most with (Step 0) | Arm that matches ApET | Evidence coverage ordering |
|---|---|---|---|
| H-detail | detail energy | detail selector; non-pad random recovers only part | detail ≈ ApET > non-pad random > random |
| H-pad | not-padding | non-pad random | non-pad random ≈ ApET > random |
| H-norm | feature norm | norm selector; detail weaker | norm ≈ ApET |

**Merge readouts (same runs).** Merge − drop for each selector; the selection × merge interaction (does merging help weak selectors more?); **dilution**: for kept answer-region tokens, sources absorbed and own share `1/(1+n)`, and whether merge-specific losses concentrate where answer tokens were heavily diluted; **gist**: whether merge-specific gains concentrate on questions whose evidence was dropped but merged in.

**Supports.** Links 2 and 4 of the main question, and RQ2's selection/merge separation.

**Checks.** `apet + merge` reproduces saved native outputs (compare pre-merge features exactly given the known FP16 scatter nondeterminism); exact counts at both stages; identical basis across arms; first fresh runs kept as primaries.

**Cost.** 13 selector configurations × 2 = 26 runs of TextVQA100 (TextVQA300 if extended). If short of time, cut the optional FPS arm and the merge versions of the norm and non-pad arms first.

## 6. Later or conditional experiments (short ideas)

- **A0 — CPU property audit (in parallel with Experiment 1).** Per-image norm histograms: do high-norm artifact tokens exist in LLaVA's CLIP-L/336 penultimate features, and where? Are they in the FPS basis? Spearman ρ of stage-1 E with norm, detail and not-padding, plus top-96 overlaps; padding share in random's kept sets; stage-2 E vs. norm and outlier dimensions at layer 16. Results decide which Experiment 2 arm to watch.
- **Exp 3 — pooled mediation.** `correct ~ method + coverage` (question fixed effects) over Experiments 1–2; report how much the ApET–random coefficient shrinks.
- **Exp 4 — force-in / force-out intervention (about 50 TextVQA questions, drop compensation).** ApET plus all answer patches at a fixed budget (replace lowest-E ordinary tokens) and ApET minus answer patches (replace with next-highest E). Tests necessity.
- **Exp 5 — GQA800 replication** of ApET, random, non-pad random and the best simple selector, drop and merge. Evidence coverage only if object boxes are available for our questions; otherwise padding, detail and norm properties only.
- **Exp 6 — answer turnover vs. baseline margin.** One uncompressed pass saving answer scores; do ApET and random flips both concentrate on low-margin questions? If so, much of the "gain" side of turnover is decision-boundary noise.
- **Conditional on Experiment 2:**
  - *Size-weighted merging* (`log n` logit bias, counts composed across stages; identity check with all n = 1) if merge results suggest attention-mass loss or denoising.
  - *Stage localization* (ApET/random crossed across the two stages) if coverage does not explain the gap.
  - *Local merging* or merge spatial-spread analysis if merge groups prove spatially scattered.
  - *E on RMS-normalized features* if H-norm is supported.

## 7. Schedule and GPU use

| When | Work |
|---|---|
| Nov 5 (tonight) | Feature dump; start Experiment 1 seeds 4–20 on TextVQA100 |
| Nov 6 (before the Friday 8 PM quota reset) | A0 audit and answer-region mapping on CPU; implement the selector hook and drop switch; identity checks; start Experiment 2 |
| Nov 7–13 | Finish Experiment 2; cohort extension if chosen; Experiment 1 analysis |
| Nov 14–20 | Exp 3–5; Exp 6 if time |
| Nov 21–23 | Buffer and confirmation |
| Nov 24 → | Report and Dec 5 presentation |

## 8. What would count as success

- **Strong:** coverage predicts correctness within question (Exp 1), explains most of the gap (Exp 3), a simple selector matches ApET *with similar coverage* (Exp 2), and the intervention moves answers (Exp 4).
- **Informative partial:** one rival (padding or norm) explains part of the gap, with the remainder quantified.
- **Negative:** coverage does not predict correctness; ApET's advantage lies elsewhere. Reported with the arm-activity checks so that the null is interpretable.

Sources: Ma et al., ApET (CVPR 2026); midterm report v9 and the October 29 gap note; code in `llava/model/llava_arch.py`, `llava/model/modeling_llama_x.py`, `llava/model/utils.py`; Darcet et al., *Vision Transformers Need Registers* (ICLR 2024); Sun et al., *Massive Activations in Large Language Models* (2024); Bolya et al., *Token Merging* (ICLR 2023); Chen et al., FastV (ECCV 2024).
