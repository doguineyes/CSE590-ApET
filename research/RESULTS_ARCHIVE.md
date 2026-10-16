# Results archive

Checked **2026-10-16**, before the TextVQA baseline. Raw artifacts are kept in
`research/results/`, which is ignored by Git. GitHub therefore does not back up
these result files. Preserve an additional copy of the downloaded exports.

## Verified local holdings

The [dated inventory](results/archive-inventory-2026-10-16.json) records sizes,
SHA-256 hashes and report provenance for **32 top-level artifacts**. All **19
ZIPs** passed a full member CRC check. Extracted folders and separately generated
review galleries are retained alongside the original ZIPs. No predictions,
measurements, original logs or ZIP contents were changed during this audit.

All five current RQ1 runs have both their original ZIP and adjacent `.log`:

| Run | Timestamp suffix | Interpretation |
|---|---|---|
| GQA calibration | `20261016T133855495304Z` | Preserve as historical evidence; raw-lowercase target policy superseded |
| GQA target audit | `20261016T140244407717Z` | Evidence for the target-format correction |
| GQA calibration v2 | `20261016T142220402681Z` | Validated corrected measurement |
| GQA input pilot v2 | `20261016T143422920388Z` | 12 questions × 576 interventions |
| MMStar calibration v2 | `20261016T160231333218Z` | 3 questions × 32 interventions |

Earlier native artifacts are also present: passed original-ApET smoke, MMStar
24 and its separately rescored v2 exports, MMStar 1,500, GQA pilot, GQA development
800, and GQA repeatability. Keep both original GQA-development ZIPs: the
`-complete.zip` adds the matching lock omitted by the original packaging; its
predictions are unchanged. Candidate/input bundles and both human review JSONs
are preserved. See the dated notes for scientific validation and limitations;
CRC checks alone do not establish scientific validity.

## Gaps and scope limits

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
  Other Downloads files have not been inventoried. These are **not located**
  artifacts, not evidence that they were deleted. No source files were removed.

## Workflow for subsequent runs

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

## Next Kaggle run: TextVQA

**No GitHub pull, environment reinstall or input re-upload is required in the
current live notebook.** The MMStar calibration ZIP contains the TextVQA runner
in its archived `scripts/run_rq1_input.py`, from runtime commit
`a3842946ccd91c35805c161dca1f23729000d325`.

Add a code cell using the already defined helper:

```python
textvqa_dir = run_experiment(
    "rq1-textvqa-baseline", "textvqa/manifest.json",
    ["--mode", "textvqa", "--configurations", "baseline", "input-only"],
)
```

This runs all 100 retained questions under baseline and input-only ApET (200
generations), with ten-annotator consensus scoring. Download its ZIP and log.
This is baseline/preservation screening; TextVQA token interventions follow a
separate reference-format decision.
