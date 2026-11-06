# GQA selector-grid completion — 2026-11-06

Saved all three new ZIP/log pairs from Downloads into research/results, retaining originals and verifying copy SHA-256 equality. Archives ending184239002981Z,190743936821Z,193311388609Z are respectively frozen shards1,2,3. Each passed200 questions/25 images/2800 predictions. ZIP CRC and345 internal file hashes passed for each. Runtime24.98,25.35,25.04 minutes respectively.

Together with earlier174104944642Z shard0, IDs exactly cover the frozen development800:800 distinct questions,100 images,11200 predictions. No duplicate IDs or missing questions. All four reports have identical measurement source hashes, environment lock and condition lists. No rerun is required for completeness.

Descriptive combined accuracy: full57669.00%; native63.75%; random-drop five-seed mean61.375%; nonpad-drop matched five-seed mean64.35%; detail-drop61.125%; no-basis E-drop61.375%. These are descriptive values; no paired confidence intervals were computed during this archive/completeness check. Native has78 baseline-correct losses and36 gains. Detailed machine-readable combined counts are saved in research/results/selector-grid-gqa800-completeness.json.

The TextVQA detail advantage does not appear in the combined GQA point estimates. Nonpad random has a modest point-estimate advantage over native, which should not be called superiority without the paired image-level analysis. GQA merge variants were not included.
