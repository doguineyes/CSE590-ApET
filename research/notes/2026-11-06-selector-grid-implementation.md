# November6: GPU selector comparison while annotation continues

Reviewed the attached discussion and implemented the no-basis grid as a **new research
family**, preserving original ApET, old fixed-basis ablations and environment locks.
The old feature/coverage observations motivate this test; they do not establish the
proposed mechanism. No GPU execution has occurred locally.

## Scientific decisions

- Run TextVQA first; no OCR boxes or reference answers enter any selector. Annotation
  and CPU analysis can proceed independently. FamilyB and answer force-in/out wait.
- Use fresh uncompressed/native and five fresh pure-random-drop reference arms. This
  adds seven reference arms to the24 new proposed arms, giving31 total. Don't pool
  historical repeated seeds as new observations.
- At both stages, research arms insert no basis. E still uses native FPS10 for scoring.
  Averages route to all retained targets; k=0 is never passed into native merge code.
- Corrected detail ignores differences touching geometric padding, including boundaries
  inside a patch. It is not identical to the earlier exploratory detail map. Portable
  offline top96 references were regenerated from the archived100-image pixel tensors.
- Nonpad policy prioritizes fully nonpadding patches, then minimal-padding fallback,
  randomizing ties. The square-image special case exactly equals pure random.
- E is recomputed after block16 on current representations. Stage1 merging can alter
  stage2 E/FPS/keep sets; therefore drop/merge is a total pipeline comparison. Input
  mask equality and decoder membership overlap are explicitly recorded.
- Detail/nonpad comparisons do not perfectly hold padding fixed. E-drop vs native
  bundles basis and merging changes. Positive coverage associations do not prove
  late-layer necessity, and E/norm under ridge is not an exact sin(theta) measure.

## Delivery

Notebook: `research/notebooks/kaggle_selector_grid.ipynb`.
Protocol/readme: `research/SELECTOR_GRID.md` and `research/protocols/selector_grid_v1.json`.

First5 manifest TextVQA questions,31 arms,155 primary predictions; execution gates
and review before all100×31. Gates: actual tiny-native tests with zero skips, original
pure-random operator score agreement, all8 selector/merge modes, exact CPU top96
membership, counts/geometry, replay, cache and identity576. Archived native drift
is recorded instead of requiring lucky byte-identical reruns.

Same notebook offers GQA800 replication in four disjoint200-question image shards.
Fourteen arms each: native/full, E-drop/detail-drop,5 nonpad-drop and5 random-drop.
Source/calibration gates remain active. This is11,200 predictions overall, not a
promise to finish before the quota reset. Calibrate and measure runtime first.

Full result archives preserve masks at both stages, original patch mappings, all
merge routes/counts/own-shares, per-image E/padding/detail maps, generated predictions,
scoring, native/reference drift, source/lock/runtime evidence. Annotation can join
later by case/image IDs. No answer-dependent filtering or method tuning is allowed
inside a live run.

## Local checks

27 tests passed with the Torch-dependent class skipped (3 new native tests await
Kaggle); notebook cells and runner syntax validated; isolated `--help` invocation
works;63 source files hash successfully. CPU checks cover100 offline selector maps,
original random masks, square nonpad equality, shortage fallback, score ties and
padding-border exclusion. Pretrained calibration and GPU native checks remain pending.
