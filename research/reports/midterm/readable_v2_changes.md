# Readable midterm revision — October 23, 2026

- [Readable Markdown](midterm_readable_v2.md)
- [Four-page PDF](midterm_readable_v2.pdf)
- [Detailed evidence checklist](evidence_checklist.md)

The feedback's main recommendation was sound: the detailed draft contained
enough evidence but needed a clearer research story. This version focuses on
what was done, the two main findings, and the unresolved component question.

## What changed

- Kept the professor's seven sections and original three research questions.
- Retained two principal scientific tables: RQ1 correlations and staged
  native-versus-random scores, including all original seed SDs and paired
  intervals. Summarized the selected component experiment in prose, preserving
  both native merge/no-merge counts and the basis-fixed random results.
- Replaced equations with plain-language explanations of the two response
  measurements. Removed secondary token-probability examples and detailed
  diagnostic counts. Kept Brolio as a short, qualified case study.
- Made a compact vector diagram and kept risk/milestone summaries concise.
- Verified the tentative December 15 final presentation/report/code date against
  page 6 of the current local Fall 2026 syllabus. The newer portal announcement
  still governs the October 30, 11:59 PM midterm deadline.

## Scientific boundaries retained

RQ1 is largely complete only within the bounded diagnostic scope. Its scores
are intervention-specific first-piece responses, not full-answer accuracy or
intrinsic importance. RQ2's selected component tests and broader whole-procedure
comparison remain separate. TextVQA100/GQA800 retain their screening/development
roles. Mixed effects and zero-containing intervals do not establish universal
rankings or equivalence. Fresh-run variation remains a limitation. RQ3 is
undecided; proposed dates are planning targets, not promised improvements.

## Validation and reproduction

The PDF has **four US Letter pages including references**, 11-point body text,
embedded fonts, linked references and a vector method diagram. All rendered
pages were inspected for legibility, clipping, overlap and table breaks. Tables
were checked against the saved summaries and source hashes. No new GPU run or
experimental change was made. The original detailed Markdown and figure remain
unchanged. Slides have not been generated.

The canonical PDF is also saved at `output/pdf/midterm_readable_v2.pdf`; both
copies are byte-identical. [Verification metadata](materials/readable_v2_verification.json)
records hashes and the checks performed.

The local build uses ReportLab 4.4.3 from a temporary PDF-only dependency folder
and system Times New Roman/Arial fonts, leaving the research environment intact:

```sh
PYTHONPATH=/tmp/apet-midterm-pdf-deps environments/editor/.venv/bin/python research/reports/midterm/build_readable_pdf.py
```

If that temporary folder is gone, install ReportLab 4.4.3 into a separate target
folder and set PYTHONPATH accordingly. The builder also uses Pillow and the font
paths documented in its source. Regenerating a PDF changes its creation metadata
and therefore its byte hash; refresh the verification after any content revision.
