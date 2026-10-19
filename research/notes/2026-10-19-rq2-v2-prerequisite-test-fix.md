# RQ2 v2 prerequisite mutation-test fix — 2026-10-19

The supplied pasted console output records 16 prerequisite tests in 4.476 seconds:
15 passed and one errored. The actual native capture, selection/basis checks,
fixed-reference scoring, tiny-decoder generation/replay and independent numerical
drift controls passed. Calibration inference had not started.

The error was in our deliberate mutation test: `capture_encoding` produces
inference tensors, and the test attempted an in-place edit outside inference
mode. PyTorch rejected the edit before `validate_snapshot` could test detection.

The correction puts only that deliberate corruption inside
`torch.inference_mode()`. The unchanged fingerprint validator then must raise
`ValueError` for the altered snapshot. Native sources, runtime, capture/replay
implementation, numerical policy and experiment protocol are unchanged.

The supplied attachment is preserved verbatim as
`results/rq2-v2-prerequisites-20261019T221634315245Z-pasted.log`. This is pasted
console evidence, not the complete standalone Kaggle setup log. The actual setup
path in it is `/kaggle/working/rq2-v2-setup-20261019T221634315245Z.log`.

After pushing/pulling the test fix, rerun `verify_rq2_v2_prerequisites()` in the
same notebook. Its subprocess imports the new test file; no kernel restart or
environment reinstall is needed. Only after it passes, run the v2 calibration.
The corrected native test awaits this Kaggle rerun; local checks skip the Torch
class in the Intel Mac reading environment.
