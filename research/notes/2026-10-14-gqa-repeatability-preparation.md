# GQA repeatability preparation — 2026-10-14

Prepared a small follow-up to the project review. This adds only project-owned
orchestration, a notebook and evidence recording; native ApET sources and the
original environment lock remain unchanged.

- Runner: `scripts/repeat_original_gqa.py`.
- Notebook: `research/notebooks/kaggle_original_gqa_repeatability.ipynb`.
- Protocol: [GQA_REPEATABILITY.md](../GQA_REPEATABILITY.md).
- Cases: airplane `2058428` and historically stable lamp control `201882614`.
- Default: 72 recorded generations plus four short warmups, one shared model
  load. Three identical repeats, logging off/on/off, and two configuration-return
  sequences are run for both questions. Optional two fresh model processes reuse
  the environment/downloads and produce 144 recorded trials.
- JSONL flushes after each trial. Reports preserve native error/FPS/selection/merge
  values when tracing is enabled, historical input identities, numerical flags,
  checkpoint/device information and source snapshots including `uv.lock`.
- Execution completion and answer/native-value variation are reported separately.
  Variation does not abort the experiment. Results use new paths and preserve
  partial evidence after failure/interruption.

Verification completed locally without installing Torch or running inference:

1. All eight CPU tests passed: trial/control coverage, reserved-manifest rejection,
   exact case preparation, changed-label/image rejection, pinned-source table
   selection, one-load/seed/prompt orchestration, retained answer variation and
   partial-trial preservation.
2. Data-only preflight against the real development manifest verified both
   historical images (raw/RGB hashes and dimensions), exact labels/questions and
   seeds `334882464` / `413006391`. The matching lock and source snapshots were
   included. Scratch artifacts were temporary.
3. Notebook code cells compile with empty outputs and a False launch flag.
   All eight monitored native file hashes still match the passed reference.

GPU generation has not been verified for this new runner. The next action is to
push the files, enable the notebook flag and run it on Kaggle T4 ×2. Read results
as scoped repeatability evidence; stable answers here do not explain or reproduce
the entire historical pilot/development execution contexts. No full baseline
rerun or algorithm modification is part of this check.
