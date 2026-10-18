# RQ1: add input-stage 1×1 IH to the existing panel

This extension adds retain-one sensitivity to **every canonical input-stage
case already measured**: GQA 12, MMStar 12, TextVQA 6. Existing calibrations and
smaller sweeps are nested repetitions, not extra cohorts. The six-question
layer-16 zeroing/IH full grid is already complete and stays unchanged.

No native ApET, environment, dataset selection, target spelling or historical
measurement file changes. New measurement code is project-owned. The
[profile](protocols/rq1_input_ih_v1.json) pins the reference bundle and cohort.

## What to upload tomorrow

1. Commit/push the new code and updated notebooks to `apet-analysis`.
2. Import [kaggle_rq1_input_pilot.ipynb](notebooks/kaggle_rq1_input_pilot.ipynb)
   into a new editable Kaggle notebook. Enable **Internet and T4 ×2**.
3. Upload/attach **[rq1-ih-inputs-v1.zip](results/rq1-ih-inputs-v1.zip)**, **34.1 MiB**,
   as a private Kaggle Dataset. Kaggle may unpack it; both ZIP and extracted
   forms work. The dataset does not need to be public. This new bundle contains
   the old accepted images/manifests plus exact archived E/zeroing records;
   the previous `rq1-inputs-v1.zip` alone is insufficient.
4. In the first code cell set only:

   ```python
   RUN_INPUT_IH_CALIBRATION = True
   RUN_FULL_INPUT_IH = False
   ```

   Keep every historical experiment flag False. `INPUT_ZIP = None` lets the
   notebook detect the dataset. If several copies are attached, set `INPUT_ZIP`
   to the exact ZIP path **or** the extracted folder containing `references/`.
5. Run cells in order. Native tiny-model prerequisites run before pretrained
   loading. Download the new **calibration ZIP and log** for review.

The notebook's default flags are all False to prevent accidental GPU work when
Kaggle imports/runs a version. Import into an editable draft to change them.
No local Python setup or Mac package installation is required.

## Calibration and full sweep

The first two **recorded** questions from each original dataset run are the
calibration panel. They are fixed independently of the new IH outcomes:

| Dataset | Cases |
|---|---|
| GQA | 201636988, 201637316 |
| MMStar | 313, 617 |
| TextVQA | 37533, 35670 |

Each uses the original first 32 positions, shared across same-image questions:
**192 new retention scores**, plus **18 zeroing audit repeats** and controls.
MMStar 617 retains its existing quality flag; calibration success does not
establish visual grounding. No flagged case is silently removed from the cohort.

After reviewing the calibration, run this cell in the same live notebook:

```python
input_ih_full_dir = run_input_ih(
    tokens=576,
    calibration=input_ih_calibration_dir,
)
```

The full run measures **30 × 576 = 17,280 IH effects**, audits 90 zeroing scores,
and joins all 17,280 archived E/zero pairs. Its 192 calibration positions must
repeat exactly and are not additional independent observations. One model load
serves all questions; environments and checkpoint downloads are reused between
commands in the live session. A new Kaggle session reinstalls/reloads as usual.

Use the **new input-IH** calibration, not a layer-16 report. If returning to a
later session, preserve/extract the complete calibration ZIP and set
`IH_CALIBRATION_REPORT` to its `report.json`; the source/model/profile/reference
gate verifies detailed question records too. A report alone is insufficient.
Any relevant measurement change requires a matching calibration. Never pull
Git while a subprocess is actively running.

Runtime needs a new estimate from the input calibration. Layer-16 timing is not
an appropriate direct estimate: input deletion avoids the long visual prefix in
every decoder block. Correlations and sampled maxima should not determine
whether the other frozen cases are retained. Review controls, shapes and runtime.

## Method and exact reuse policy

The new operator slices prepared **projected visual embeddings before block 1**,
retaining all nonvisual embeddings and original retained RoPE position IDs.
The original native decoder builds its own shortened causal mask. Temporary
hooks extend old native RoPE table lookup to the original position extent;
they are removed on errors. No custom decoder suffix, native file edit,
FlashAttention install or different Torch/Transformers environment is used.

This is a **source-informed 1×1 IH adaptation**, following the same released
physical-retention interpretation documented in the
[layer extension](RQ1_LAYER_IH_PILOT.md). It is not a complete reproduction of
IH's original benchmark or its default 2×2 regions.

The two primary I columns share first-gold-token **probability units**:

```text
I_zero_first = exp(archived full first log p) − exp(archived zero-one first log p)
I_IH         = p(first token | retain one) − p(first token | remove all visual inputs)
```

The original zeroing whole-answer mean-log score is also saved in the comparison
CSV. No old sensitivity values are overwritten. E remains native ApET's 1024-D
pre-projector CLIP residual; zeroing and retention act on the corresponding
4096-D projected input embeddings, as in the existing input experiment.

Before joining old scores, each case must pass:

- Exact CLIP/projector hashes, seed, native E/basis, prompt, target IDs, positions
  and expanded sequence length.
- Fresh correct baseline generation with the same frozen target-token prefix.
- Exact full-reference target-token scores, native/cached parity and clone no-op.
- Exact repeats of the first three archived zeroing interventions, selected by
  the old random permutation rather than E or I.
- Retain-all no-op, repeated null, repeated retained case, final reference and
  unchanged input checks. Internal retention controls keep the frozen 0.001
  per-answer-token tolerance; **cross-run archived reuse requires exact scores**.

Any mismatch stops with evidence. We do not loosen the gate, replace an answer,
drop a case, retarget to a prediction or automatically rerun a new baseline.
A few zeroing repeats do not prove the entire old run mathematically; together
with exact features/references and pinned code/runtime they provide a documented
consistency check. Small effects still require numerical caution.

At this input boundary no decoder block has processed image information.
However, the null retains the original image-placeholder positional gap and
is not the ordinary question-only prompt. Each retained CLIP feature has already
been contextualized by the vision encoder; its patch index does not isolate raw
pixels. At layer 16, retained text has also seen the image. Keep these boundary
differences explicit when comparing depths. Neither I is intrinsic token
importance; probability for a first tokenizer piece is not answer accuracy.

## Outputs and provenance

Each timestamped run saves a ZIP and a separate streamed `.log`, including on
handled failure. Download both before closing Kaggle. Outputs include:

- `comparison.csv`: sample/stage/index, native E, old zero-whole, matched
  zero-first and IH-first scores, full/null/retained probabilities, FPS and
  padding labels.
- `questions.csv`, `summary.json`, `index.html`: per-question all/nonbasis
  correlations and probability summaries. Calibration summaries cover only
  sampled positions; sparse strong states may remain unseen.
- `questions/*.json`: raw target-token probabilities, retained position maps,
  strict reuse checks, repeat controls and fresh baseline evidence.
- Used original images, actual 336×336 model canvases and fresh feature metadata.
  Original feature tensors remain in the canonical archived ZIPs; no redundant
  tensor upload is needed and no tensor file is unpickled locally.
- Reference records/reports/index, accepted manifests, native/source snapshots,
  original lock/profile hashes, and full-run prerequisite report/details.

The portable bundle was built by fully auditing the three hash-pinned canonical
source ZIPs with `prepare_rq1_ih_inputs.py`; no inference or curation occurred.
Its reference index SHA256 is
`dd91cca156a090fc6e2c236f16042358e2dbe3b9db5d49d12609a45618d49644`.
ZIP SHA256:
`27c30658b6ec201a67e175dd1f42f1d59d981856254ddc73eb13e9009480f419`.
Results/bundles are ignored by Git, so pushing code does not upload them.

## Validation scope

CPU checks cover bundle integrity, exact reuse/drift rejection, pairing units,
retained geometry, corruption, and complete compatible calibration gating.
Three tiny **native Torch** tests compare physical input retention to an
independent manual decoder oracle, verify original positions and causal target
scoring, and check hook cleanup/reference recovery. The notebook imports Torch
explicitly so these native checks cannot silently skip in Kaggle.

The Intel Mac does not have this native runtime. Native tests and pretrained
calibration must therefore pass on Kaggle before the full run. No new model
inference has been claimed during implementation.
