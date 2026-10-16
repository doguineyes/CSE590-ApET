# TextVQA baseline and input-only ApET — 2026-10-16

The retained 100-question screening run completed successfully. TextVQA provides
enough baseline competence for a small RQ1 study: 44 cases receive full consensus
credit before compression. Keep all 100 cases in this baseline archive; select a
documented diagnostic subset separately for interventions.

## Archive and validation

Original ZIP and log were copied from their explicitly supplied Downloads paths
to `research/results/rq1-textvqa-baseline-20261016T163946789583Z.{zip,log}`.
Downloads originals were left intact. ZIP SHA-256:
`ebf1cfc318eeb9e712452fd71efce6ee69516faaed8c7d4b4b8f1d948715902b`.
Log SHA-256:
`7910db8024f3ca5bcf93f3c3084e737fb9dd5b2f88f86bb1b35ddfa5219e3f1f`.

The ZIP is 28,592,191 bytes with 125 entries; CRC validation passed. All 23
recorded evidence hashes and all 100 image hashes match. The accepted TextVQA
manifest matches the portable input bundle byte for byte. Native source hashes
match the original-LLaVA reference, including its previously documented mask-device
compatibility patch. Runtime commit `a3842946ccd91c35805c161dca1f23729000d325` has
a clean working tree; original dependency lock is unchanged.

All 200 unique question/configuration records passed, with no empty responses.
Questions, all ten references, image identities and fixed canonical targets
match the manifest. Paired prompts and seeds agree. Input traces confirm
576 → 576 for baseline and 576 → 288 for input-only; no decoder pruning was used.
Every consensus score and report summary was independently recomputed on CPU
using the exact archived native scorer. No labels or predictions were rewritten.
The model-type and TypedStorage warnings in the log were non-fatal.

## Results

| Metric | Baseline, 576 tokens | Input-only, 288 tokens |
|---|---:|---:|
| Questions | 100 | 100 |
| Mean ten-answer consensus score | 49.7% | 46.9% |
| Full-credit answers | 44 | 43 |
| Answers with any positive credit | 53 | 50 |
| Full-credit baseline answers preserved | Reference | 41/44 = 93.18% |

The mean-score change is **−2.8 percentage points**. This is fractional-credit
TextVQA scoring, not 49.7 or 46.9 completely correct questions. The paired mean
score ratio is 94.37%, a separate quantity from full-credit preservation.

Of 100 pairs, 93 keep the same score, five decrease and two increase. Normalized
answers change on 23 cases, including 16 with unchanged score. Compression loses
three full-credit answers and gains two (one previously zero-credit and one
previously partial-credit), leaving 43 full-credit answers. Similar totals again
hide which answers change.

Runtime was **157.02 seconds / 2.62 minutes**. Mean recorded generation time was
0.805 s baseline and 0.588 s input-only. Peak allocated GPU memory was
8.134/7.272 GiB and 8.134/7.212 GiB, respectively. Maximum generated length was
11 tokens, below the 16-token cap. These are single-session sequential screening
timings with baseline always first and variable answer lengths, not a controlled
latency benchmark.

## Cases and interpretation

| Question ID | Baseline → input-only | Consensus score | Initial reading |
|---|---|---|---|
| 37533 | Brolio → Bottoli | 1 → 0 | Candidate text-reading loss; BROLIO is visible on repeated wine labels |
| 35962 | Cookery school → Culinary school | 1 → 0 | Candidate name-reading loss; wall/aprons visibly say cookery school, so a generic synonym does not preserve the name |
| 36001 | Give up → Gone | 1 → 0 | Candidate poster-reading loss; give up appears on posters |
| 34764 | Um → Umd | 0 → 1 | Candidate gain in reading a uniform abbreviation; needs visual/input-scale review |
| 37034 | Monday night → Monday | 0.3 → 1 | Score gain includes reference wording effects; not automatically better visual understanding |
| 38355 | Chateau pontet-canet → Chateau pontet canne | 0.9 → 0 | Partial-credit spelling/name loss; reference spellings themselves vary |
| 38104 | Princess → Woman | 0.9 → 0.3 | Awkward question, mixed reference answers; weak primary mechanism example |

The three full-credit losses and the Princess image were visually inspected.
The Princess question is “who' is use this?” on a perfume image with “Princess”
printed on it; annotators disagree between princess, women and unanswerable.
Preserve its official score, but do not treat it as a clean compression mechanism
case. All images still need to be considered at the model's actual 336-pixel
preprocessing scale before asserting fine-text legibility.

These examples motivate RQ1; they do not yet show that low-reconstruction-error
patches caused the answer changes. Zeroing individual full-model patches and
native selection/merging measure different interventions. The 100 retained
examples are a screening subset, not a full TextVQA benchmark.

## Saved review and next steps

The separate [100-case review gallery](../results/rq1-textvqa-baseline-20261016T163946789583Z-review/index.html)
shows images, questions, both answers and all ten references, with score decreases
and increases first. Adjacent `summary.json` and `questions.csv` retain the audit
and paired scores. Original ZIP/log and extracted evidence remain unchanged.

1. Complete the planned 576-patch measurement on the same three MMStar cases;
   no new GitHub pull or dependency installation is needed in the current notebook:

   ```python
   mmstar_fullpatch_dir = run_experiment(
       "rq1-mmstar-threecases-v2", "mmstar/manifest.json",
       ["--questions", "3", "--tokens", "576"],
   )
   ```

2. Prepare a bounded TextVQA intervention pilot from baseline-full-credit cases:
   include the three full-credit losses and predefined preserved controls, rather
   than choosing only failures. Document selection and inspect model-scale text.
   Keep gains/partial-credit cases available for secondary qualitative analysis.
3. Establish exact teacher-forced reference token formatting, no-op/parity controls
   and repeats before expanding TextVQA token measurements. All 44 full-credit
   baseline answers match the normalized frozen canonical reference, but normalized
   agreement does not establish exact tokenizer-prefix alignment. The current CLI
   still deliberately refuses TextVQA token interventions; this baseline does not
   change that implementation boundary.
