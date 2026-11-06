# Selector grid calibration review — 2026-11-06

Archived ZIP and matching log in research/results, retaining Downloads originals. ZIP CRC integrity verified; copied-file SHA-256 verified.

- Run: selector-grid-textvqa-calibration-v1-20261106T164503679270Z
- Commit: cee1d74c5f053bb293cbca9f803a8884d4654fa4 (clean).
- Status passed: 5 questions, 31 conditions, 155/155 primary predictions.
- Wall time: 199.76 seconds (about 3.3 minutes).
- Kaggle prerequisites: 27 tests passed, zero skips.
- All five question records passed; archived CPU top-96 agreement passed.
- No reported historical output drift for the comparable baseline/native/random-drop reference arms.
- Original native source matches reference; environment lock preserved in archive.

Calibration supports proceeding with the full TextVQA100 run. Five questions are insufficient for ranking selectors or drawing mechanism conclusions. Analyze full-run paired outcomes and seed variability before choosing follow-up conditions; no changes to running measurement code are required based on this calibration.

ZIP SHA-256: 44805d484247ff9b2eca0b05cc5bd4e9a68cd1af5400124df951e64108b5061c
Log SHA-256: 53f47fbb9f61248a59c992fc2710d1ac3282b48020c3b7c40feddb4571f3c390
