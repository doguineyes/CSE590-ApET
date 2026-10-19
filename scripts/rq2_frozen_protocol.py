"""Versioned RQ2 protocol for one frozen native feature realization per image.

This CPU-only module preserves the accepted v1 cohort and scorers. It changes
the execution contract: fresh native runs are numerical controls, whereas
generation and fixed-answer scoring replay the same captured feature tuple.
"""

import copy
import json
from pathlib import Path

from rq1_protocol import REPO, sha256
from rq2_protocol import (ARMS, PROFILE as V1_PROFILE, case_panel,
                          load_profile as load_v1_profile, require,
                          validate_bundle as validate_v1_bundle)

VERSION = "rq2-input-merge-v2"
PROFILE = REPO / "research/protocols/rq2_input_merge_v2.json"
CALIBRATION_IDS = (
    "gqa:201636988", "gqa:201637316", "mmstar:313",
    "textvqa:37533", "textvqa:35670", "textvqa:35962", "textvqa:36001",
)
REPEAT_CASE_IDS = (
    "gqa:201636988", "textvqa:37533", "textvqa:35670",
    "textvqa:35962", "textvqa:36001",
)

# Semantic identities a full-run calibration gate must compare, rather than
# merely trusting that an arbitrary report says "passed". The runner owns the
# concrete report paths, validation, and archive-hash verification.
REQUIRED_CALIBRATION_FIELDS = (
    "protocol_version", "profile_sha256", "reference_index_sha256",
    "input_protocol_sha256", "measurement_source_sha256",
    "native_source_sha256", "original_lock_sha256", "runtime_identity",
    "checkpoint_identity", "case_ids", "control_results", "status",
)

FROZEN_REALIZATION_POLICY = {
    "bank_key": ["image_sha256", "seed", "arm"],
    "primary_capture": "First unconditional native or scoped no-merge encoding; never select, retry, average, or replace a realization using its answer or reference score.",
    "stored_tuple": "The returned full projected feature tensor and its native boolean mask; keep native selected ordering and basis source-to-slot placement.",
    "reuse": "Reuse that exact tuple for every same-image question, generation, teacher-forced fixed-reference scoring, and fixed-replay controls.",
    "scope": "One run-local feature bank; separate calibration/full runs capture new first realizations and must report their independent numeric variation.",
    "exact_checks": [
        "image, pixel, CLIP patch and pre-merge source identity",
        "FPS basis indices and per-token reconstruction errors",
        "selected source indices, mask and output ordering",
        "basis source-to-slot placement and merge graph",
        "no-merge incoming/outgoing feature identity",
        "same-execution captured tuple and replay tuple identity",
        "same-image question reuse of each frozen tuple",
        "fixed-reference target token IDs, label positions and expanded prompt length",
    ],
    "fresh_native_control": "Record independent fresh native merge captures and their feature, score and generated-answer changes; do not demand byte equality between separate native executions.",
    "prohibited": [
        "broad numeric tolerance replacing an exact structural check",
        "recomputing a primary encoding while switching from generation to scoring",
        "reselecting a realization because its answer or score is favorable",
        "counting control repeats as additional cohort questions",
        "treating a passed diagnostic execution as a passed scientific pilot",
    ],
}


def load_profile(path=PROFILE):
    """Validate v2 against the unchanged, already accepted v1 identities."""
    profile = json.loads(Path(path).read_text())
    old = load_v1_profile()
    require(profile.get("version") == VERSION, "Wrong frozen RQ2 profile version")
    require(profile.get("source_pilot_profile") == str(V1_PROFILE.relative_to(REPO))
            and profile.get("source_pilot_profile_sha256") == sha256(V1_PROFILE),
            "Frozen RQ2 v1 source profile differs")
    unchanged = (
        "input_bundle", "reference_index_sha256", "source_input_ih_profile",
        "source_input_ih_profile_sha256", "flags_profile", "flags_profile_sha256",
        "native_reference_sha256", "lock_sha256", "arms", "generation",
        "evaluation", "case_ids", "cases",
    )
    require(all(profile.get(key) == old[key] for key in unchanged),
            "Frozen RQ2 changes an accepted cohort, arm, runtime, target or scoring policy")
    require(profile.get("frozen_realization") == FROZEN_REALIZATION_POLICY,
            "Frozen RQ2 capture/replay policy differs")
    controls = profile.get("controls", {})
    for field in ("fresh_native_repeats", "fixed_replay_repeats"):
        require(controls.get(field) == list(REPEAT_CASE_IDS),
                "Frozen RQ2 repeat controls differ: " + field)
    require(controls.get("reference_repeats") == old["controls"]["reference_repeats"],
            "Frozen RQ2 drops earlier TextVQA reference controls")
    require(controls.get("fresh_native_captures_per_control") == 2
            and controls.get("fixed_replay_executions_per_control") == 2,
            "Frozen RQ2 repeat counts changed")
    calibration = profile.get("calibration", {})
    require(calibration.get("case_ids") == list(CALIBRATION_IDS)
            and calibration.get("required_identity_fields") == list(REQUIRED_CALIBRATION_FIELDS)
            and calibration.get("full_run_requires_passed_calibration") is True,
            "Frozen RQ2 calibration policy differs")
    require([cid for cid in old["case_ids"] if cid in CALIBRATION_IDS] == list(CALIBRATION_IDS)
            and set(REPEAT_CASE_IDS) <= set(CALIBRATION_IDS),
            "Frozen RQ2 calibration omits repeat controls or canonical ordering")
    require({cid.split(":")[0] for cid in CALIBRATION_IDS} == {"gqa", "mmstar", "textvqa"},
            "Frozen RQ2 calibration must cover all three datasets")
    return profile


def validate_bundle(inputs):
    """Delegate accepted-data checks to v1; only replace its execution profile."""
    profile = load_profile()
    old, index, manifests, records = validate_v1_bundle(inputs)
    require(old["case_ids"] == profile["case_ids"], "Frozen RQ2 accepted bundle cohort differs")
    return profile, index, manifests, records


def choose_panel(records, calibration=False):
    """Choose the predeclared calibration or full cohort, never an arbitrary prefix."""
    require(type(calibration) is bool, "Frozen RQ2 calibration mode must be boolean")
    profile = load_profile()
    panel = case_panel(records)
    allowed = set(CALIBRATION_IDS if calibration else profile["case_ids"])
    return [{**record, "rq2": copy.deepcopy(entry)}
            for record, entry in zip(panel, profile["cases"])
            if record["case"]["case_id"] in allowed]
