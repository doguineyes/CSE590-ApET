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

The later `rq1-layer16-ih-calibration-v1-20261018T002129035074Z.zip` and `.log`
(October 17 local time) are also preserved and match their Downloads originals.
Its 92 ZIP members pass CRC, with 49 evidence hashes and 11 feature-file byte
hashes verified. All 384 zeroing effects reproduce the preceding calibration;
192 new IH effects and all controls pass. The
[IH calibration review](notes/2026-10-17-rq1-layer-ih-calibration-review.md)
records hashes, first-token probabilities, limitations and the separate saved-data
review. These repetitions do not increase the canonical input-stage sample count.

The full `rq1-layer16-ih-sixcases-v1-20261018T014830278963Z.zip` and `.log`
are now preserved too. All 93 members pass CRC, 50 evidence hashes and 11
feature-file hashes verify, and all 10,368 scores pass independent arithmetic
and geometry/control checks. Its embedded calibration is the user's rerun
ending in `013751203709Z`, not the earlier `002129035074Z` run; the new report's
content hash and source gate verify. That rerun's separate ZIP/log were not
supplied, but the prerequisite report is preserved and no rerun is required.
The [full review](notes/2026-10-17-rq1-layer-ih-full-review.md) records both raw
hashes, exact historical repeats and strong singleton effects missed by the
32-position sample. Derived HTML/CSV and source snapshots are separate in
`results/rq1-layer16-ih-full-review-v1-2026-10-17/`. Original files remain unchanged.

### October 18 prepared input bundle

`rq1-ih-inputs-v1.zip` (34.1 MiB) is the new portable input/reference bundle for
the [input IH extension](RQ1_INPUT_IH.md). It adds the verified canonical
30-case E/zero records to unchanged accepted images/manifests. All 207 members
pass CRC; reference-index and ZIP hashes are recorded in the
[implementation note](notes/2026-10-18-rq1-input-ih-implementation.md).
This is prepared evidence for a future run, not a new GPU result. The previous
input bundle and all source results remain preserved. Upload the new bundle
separately from GitHub; the old bundle alone cannot supply the historical scores.

### October 18 input IH calibration

`rq1-input-ih-calibration-v1-20261018T135403155797Z.zip` and matching `.log`
are saved and match their Downloads originals. All 121 unique ZIP members pass
CRC; 95 evidence hashes, 5 fresh feature metadata files and all 192 retention
scores verify. All 18 zeroing audit repeats and reference reuse are exact;
retention controls have zero drift. The
[calibration review](notes/2026-10-18-rq1-input-ih-calibration-review.md) records
raw hashes, effect sizes, source-gate acceptance and a same-session full-run
command. Separate derived outputs are in
`results/rq1-input-ih-calibration-review-v1-2026-10-18/`.

### October 18 full input IH sweep

`rq1-input-ih-thirtycases-v1-20261018T140840768523Z.zip` and its matching `.log`
are saved and byte-identical to the supplied Downloads originals. All 209 unique
ZIP members pass CRC; 102 evidence hashes, 24 fresh feature metadata files and
17,280 retention effects verify. All 90 archived zeroing audit scores and 192
calibration IH effects repeat exactly; reference reuse and retention controls
have zero recorded drift. Its embedded prerequisite report and detailed records
match the already archived calibration. Both supplied companions are present.

The [full input-IH review](notes/2026-10-18-rq1-input-ih-full-review.md) records
raw hashes, 31.1-minute runtime, matched E/zero/IH findings, basis/padding checks,
same-image comparisons and the six-case input/layer panel. Derived verification
and synthesis are separate in `results/rq1-input-ih-full-review-v1-2026-10-18/`
and `results/rq1-input-ih-synthesis-v1-2026-10-18/`. No raw source was overwritten,
no inference was repeated locally and the primary cohort is unchanged. The dated
October 16 inventory above does not include these later files.

### October 19 first RQ2 failure

`rq2-input-merge-pilot-v1-20261019T185452294158Z.zip` and matching `.log` are
preserved byte-for-byte from Downloads. All 109 members pass CRC; 108 artifact
hashes and 101 evidence hashes verify. The embedded Kaggle setup passed ten
native prerequisite tests. The first case stopped on separate-execution projected
feature hashes, with zero admitted completed pilot cases. Selection/basis/source
identities match; numeric feature magnitudes were not retained in this archive.
The [failure review](notes/2026-10-19-rq2-first-parity-failure.md) records raw hashes,
exact checks, limitations and the bounded next diagnostic. Derived review records
are separate in `results/rq2-input-merge-failed-review-v1-2026-10-19/`.

### October 19 RQ2 numerical diagnostic

`rq2-native-parity-diagnostic-v1-20261019T193300319406Z.zip` and matching `.log`
are preserved byte-for-byte from Downloads. All 110 members pass CRC; 109 artifact
hashes and 103 evidence hashes verify, including the actual tensor evidence.
The setup passed 17 prerequisite tests; all nine diagnostic passes completed.
Raw native repeats vary, observed native differences first appear after merging
on retained states, and two no-merge repeats are exact. This is diagnostic-only,
not a passed RQ2 panel. The [review note](notes/2026-10-19-rq2-native-parity-diagnostic-review.md)
records numerical/score effects and a proposed versioned capture/replay policy.
Derived JSON/CSV verification is separate in
`results/rq2-native-parity-review-v1-2026-10-19/`. Nothing from this supplied pair
is missing. No tensor loading or inference ran locally.

### October 19 second v1 attempt and v2 correction

`rq2-input-merge-pilot-v1-20261019T195332472383Z.zip` and its matching `.log`
are preserved from Downloads; CRC/artifact hashes verify. It repeats
`features_exact` on the first GQA case, with identical masks/generated IDs and
zero completed pilot questions. The diagnostic had not changed the v1 gate.
The [v2 correction note](notes/2026-10-19-rq2-v2-capture-replay.md) records its raw
hashes and the implemented first-realization/replay policy. Use the separate
[v2 guide](RQ2_INPUT_MERGE_V2.md) for the seven-question calibration next;
old failed evidence remains unchanged.

### Subsequent downloads

The October 19 v2 prerequisite failure was supplied as pasted console output and
is preserved as `results/rq2-v2-prerequisites-20261019T221634315245Z-pasted.log`.
It is not a complete standalone setup log. Fifteen of 16 tests passed; the one
error came from our mutation test editing an inference tensor outside inference
mode. The [test fix note](notes/2026-10-19-rq2-v2-prerequisite-test-fix.md) records
the correction and rerun instructions. No calibration inference result was lost.

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
