# Results archive

Checked **2026-10-16**, including the TextVQA full pilot and twelve-case MMStar sweep. Raw artifacts are kept in
`research/results/`, which is ignored by Git. GitHub therefore does not back up
these result files. Preserve an additional copy of the downloaded exports.

## Verified local holdings

The [dated inventory](results/archive-inventory-2026-10-16.json) records sizes,
SHA-256 hashes and report provenance for **46 top-level artifacts**. All **26
ZIPs** passed a full member CRC check. Extracted folders and separately generated
review galleries are retained alongside the original ZIPs. No predictions,
measurements, original logs or ZIP contents were changed during this audit.

All twelve current RQ1 runs have both their original ZIP and adjacent `.log`:

| Run | Timestamp suffix | Interpretation |
|---|---|---|
| GQA calibration | `20261016T133855495304Z` | Preserve as historical evidence; raw-lowercase target policy superseded |
| GQA target audit | `20261016T140244407717Z` | Evidence for the target-format correction |
| GQA calibration v2 | `20261016T142220402681Z` | Validated corrected measurement |
| GQA input pilot v2 | `20261016T143422920388Z` | 12 questions × 576 interventions |
| MMStar calibration v2 | `20261016T160231333218Z` | 3 questions × 32 interventions |
| TextVQA baseline | `20261016T163946789583Z` | 100 questions × baseline/input-only; generation screening |
| MMStar full-patch check | `20261016T170443901294Z` | Same 3 calibration cases × 576 interventions |
| TextVQA audit v1 | `20261016T180759434962Z` | Failed after 3/6: cached/full score gap; preserve as failed evidence, no interventions |
| TextVQA audit v2 (filename label v1) | `20261016T182747377877Z` | All 6 passed; full-forward controls and first-token agreement have zero drift |
| TextVQA calibration v1 | `20261016T182930625290Z` | 2 cases × 32 interventions passed; ready for frozen six-case full grid |
| TextVQA input pilot v1 | `20261016T204945039325Z` | 6 cases × 576 interventions passed; all 64 calibration effects repeat exactly |
| MMStar twelve-case input sweep v2 | `20261016T215926569196Z` | 12 cases × 576 interventions passed; all 1,728 earlier full-grid effects repeat exactly |

The [input-stage synthesis](notes/2026-10-16-rq1-input-stage-summary-and-next-steps.md)
counts only the canonical GQA12, MMStar12 and TextVQA6 sweeps: 30 questions,
24 images and 17,280 interventions. Calibration and reproduction effects are not
counted again. The combined local gallery/CSV/JSON are retained separately in
`results/rq1-input-stage-summary-2026-10-16/`.

Earlier native artifacts are also present: passed original-ApET smoke, MMStar
24 and its separately rescored v2 exports, MMStar 1,500, GQA pilot, GQA development
800, and GQA repeatability. Keep both original GQA-development ZIPs: the
`-complete.zip` adds the matching lock omitted by the original packaging; its
predictions are unchanged. Candidate/input bundles and both human review JSONs
are preserved. See the dated notes for scientific validation and limitations;
CRC checks alone do not establish scientific validity.

## Gaps and scope limits

- The TextVQA full pilot embeds a newer passed audit report and all six target
  records from `rq1-textvqa-target-audit-v1-20261016T204856423651Z`. Its standalone
  ZIP/log have not been received; the key prerequisite evidence is preserved
  inside the pilot, but its separate console log is not archived locally yet.

- The standalone `rq1-runtime.json` visible in Kaggle has not been located
  locally. Download it if still available. Run ZIPs already retain dependency,
  source and runtime provenance; no inference rerun is needed for this file.
- The October 9 **HF adapter** raw exports were not located under
  `research/results/` or inside its top-level ZIPs: JSON/JSONL pairs for
  `mmstar-baseline-1500`, `mmstar-apet-input-96-1500`,
  `mmstar-apet-input-288-1500`, and `apet-fixed-output`; also
  `apet-comparison.json` and `runtime-smoke.json`. The
  [October 9 summary](notes/2026-10-09-kaggle-apet.md) remains, but is not a
  substitute for those raw files. Recover them from prior downloads/Kaggle
  exports if available; they are separate from the preserved native baselines.
- macOS denied access to listing Downloads, including an escalated attempt.
  Explicitly supplied TextVQA ZIP/log paths were subsequently readable and copied
  successfully; directory listing access is a separate restriction. Other
  Downloads files have not been inventoried. These are **not located**
  artifacts, not evidence that they were deleted. No source files were removed.

## Derived saved-data analysis (2026-10-17)

The [group/pair analysis note](notes/2026-10-17-rq1-saved-data-analysis.md) records
the completed CPU-only review of the canonical 30 questions. Derived CSV/JSON,
source validation summaries and a gallery with unchanged figure copies are in
`results/rq1-saved-analysis-v1-2026-10-17/`. Its plan pins the three original ZIP
hashes; those archives and their logs remain unchanged. This directory is a
derived analysis, not a new GPU run or an additional independent sample, and
does not increase the top-level raw artifact counts above.
The later `results/rq1-padding-aware-gallery-v1-2026-10-17/` adds clearer PNG/PDF
review figures, marker/location CSVs and rendering provenance. It preserves all
original exports and numerical analysis. Existing derived pages have a banner
link and an adjacent HTML backup; see the same analysis note for the legend.

## Workflow for subsequent runs

### October 17 additions

The original validity ZIP/log and the newly supplied layer-calibration ZIP/log
are preserved under their timestamped names. The
[validity review](notes/2026-10-17-rq1-validity-review.md) and
[layer-calibration review](notes/2026-10-17-rq1-layer16-calibration-review.md)
record their hashes and separate derived reviews. In particular,
`rq1-layer16-calibration-v1-20261017T230335470061Z.zip` and `.log` both match the
Downloads originals; the ZIP's 78 members pass CRC and all 39 evidence hashes
and 11 feature-file byte hashes verify. A summary-export encoding error was
repaired locally without changing the source ZIP or rerunning inference.
The October 16 inventory counts above remain a dated snapshot, not a claim
that today's later artifacts were included in those counts.

### Subsequent downloads

1. Download the run's `.zip` **and** `.log` from Kaggle before ending its session.
   Keep the timestamped names. Optionally download the executed notebook and
   standalone runtime report as session context.
2. Tell the assistant the downloaded filenames/paths. On that turn, copy them
   into `research/results/`, compare hashes to avoid overwriting a different
   artifact, validate ZIP members, and update the inventory/review notes.
   Downloads are not watched automatically between conversations. If macOS
   blocks access, place the files directly in `research/results/`.
3. Preserve original results and logs. Write rescoring/analysis into separately
   named outputs; report missing companions explicitly. Keep raw exports apart
   from generated review galleries and maintain a second backup outside this
   ignored directory.

## TextVQA baseline (completed)

**No GitHub pull, environment reinstall or input re-upload is required in the
current live notebook.** The MMStar calibration ZIP contains the TextVQA runner
in its archived `scripts/run_rq1_input.py`, from runtime commit
`a3842946ccd91c35805c161dca1f23729000d325`.

The completed run used the already defined helper:

```python
textvqa_dir = run_experiment(
    "rq1-textvqa-baseline", "textvqa/manifest.json",
    ["--mode", "textvqa", "--configurations", "baseline", "input-only"],
)
```

All 200 generations and their scores were verified. The
[TextVQA review](notes/2026-10-16-rq1-textvqa-baseline-review.md) records 49.7%
baseline and 46.9% input-only mean consensus scores, with 41/44 full-credit
baseline answers preserved. This is baseline/preservation screening; TextVQA
token interventions follow a separate reference-format decision. The
[three-case MMStar full-patch check](notes/2026-10-16-rq1-mmstar-fullpatch-review.md)
is also archived and verified; no repeat of either completed run is needed now.
