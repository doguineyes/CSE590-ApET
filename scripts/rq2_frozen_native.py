"""Capture actual native realizations once, then replay them without recompression.

Native CLIP, FPS, reconstruction, selection, basis assignment and merging execute
at capture. Generation and teacher forcing consume independent exact clones of
that one projected-feature/mask tuple. Independent native realizations are
numerical uncertainty controls, never an exact byte-parity requirement.
"""

from contextlib import contextmanager
import copy

from rq1_native import replaced
from rq2_native import native_input_variant, snapshot_encoded, tensor_fingerprint
from rq2_protocol import ARMS, require, score_response

VERSION = "rq2-frozen-native-v2"


def arm_configuration(arm):
    require(arm in ARMS, "Unknown frozen RQ2 arm: " + str(arm))
    return "baseline" if arm == "baseline576" else "input-only"


@contextmanager
def _runtime_scope(model, arm, seed):
    """Restore native settings, preparation metadata, patch ownership and RNG."""
    import torch
    from smoke_original_apet import configure
    inner = model.get_model()
    fields = ((model, ("visual_token_num", "visual_id")),
              (inner, ("basis_token_num", "layer_list", "image_token_list", "visual_id",
                       "image_token_posi", "prompt_len", "image_tokens", "layer_prune")))
    prior = [(obj, name, name in obj.__dict__, obj.__dict__.get(name))
             for obj, names in fields for name in names]
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            configure(model, arm_configuration(arm))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            yield
    finally:
        for obj, name, owned, value in reversed(prior):
            if owned:
                setattr(obj, name, value)
            elif name in obj.__dict__:
                delattr(obj, name)


def _pixels_fingerprint(pixels):
    import torch
    require(isinstance(pixels, torch.Tensor) and tuple(pixels.shape) == (1, 3, 336, 336),
            "Frozen RQ2 requires one native 336x336 image tensor")
    require(bool(torch.isfinite(pixels).all()), "Non-finite frozen RQ2 image tensor")
    return {**tensor_fingerprint(pixels), "device": str(pixels.device)}


def _snapshot_fingerprints(snapshot):
    values = {"encoded": snapshot_encoded(snapshot["encoded"]),
              "encoded_devices": [str(value.device) for value in snapshot["encoded"]]}
    for name in ("pre_merge", "post_merge"):
        if name in snapshot:
            values[name] = {**tensor_fingerprint(snapshot[name]),
                            "device": str(snapshot[name].device)}
    return values


def validate_snapshot(snapshot, arm, pixels):
    """An exact self-check of one realization, not parity across fresh captures."""
    arm_configuration(arm)
    capture = snapshot.get("capture", {})
    require(capture.get("version") == VERSION and capture.get("arm") == arm,
            "Frozen snapshot belongs to a different version or arm")
    require(capture.get("pixels") == _pixels_fingerprint(pixels),
            "Frozen snapshot image pixels or input device changed")
    actual = _snapshot_fingerprints(snapshot)
    require(actual == capture.get("fingerprints"),
            "Frozen snapshot mutated or its feature/mask device changed")
    require(actual["encoded"]["kept_tokens"] == ARMS[arm]["input_tokens"],
            "Frozen snapshot mask budget differs from its arm")
    require(actual["encoded"] == capture["encoding"]["encoded"],
            "Frozen snapshot differs from its native capture observation")
    return actual


def capture_encoding(model, pixels, seed, arm):
    """Return (JSON observation, tensor snapshot) from exactly one native encode.

    The no-merge arm bypasses only the existing merge call, after unchanged native
    selection and basis placement. Baseline explicitly uses the native disabled
    compression branch. Native-method identity is enforced by the observer.
    """
    import torch
    variant = "no-merge" if arm == "native-input288-no-merge" else "native"
    pixel_identity = _pixels_fingerprint(pixels)
    with _runtime_scope(model, arm, seed), torch.inference_mode():
        with native_input_variant(model, variant, capture_tensors=True) as observed:
            output = model.encode_images(pixels)
        snapshots = observed.pop("snapshots")
        require(len(observed["calls"]) == len(snapshots) == 1,
                "Frozen capture expected exactly one native image encoding")
        observation, snapshot = observed["calls"][0], snapshots[0]
        require(snapshot_encoded(output) == observation["encoded"] ==
                snapshot_encoded(snapshot["encoded"]),
                "Native output, capture observation and frozen clone differ")
        require(all(torch.equal(left, right) for left, right in zip(output, snapshot["encoded"])),
                "Frozen native clone differs from the actual returned tuple")
        require(_pixels_fingerprint(pixels) == pixel_identity,
                "Native capture mutated image pixels")
        snapshot["capture"] = {"version": VERSION, "arm": arm, "seed": seed,
            "pixels": pixel_identity, "encoding": copy.deepcopy(observation),
            "fingerprints": _snapshot_fingerprints(snapshot),
            "policy": "One actual native realization; exact clone replay; no cross-realization feature equality claim"}
        validate_snapshot(snapshot, arm, pixels)
    return observation, snapshot


class _ReplayAudit:
    """Observe returned clones and actual preparation calls for one consumer."""
    def __init__(self, snapshot, arm, pixels):
        self.snapshot, self.arm, self.pixels = snapshot, arm, pixels
        self.calls, self.returned = 0, []
        self.expected = validate_snapshot(snapshot, arm, pixels)["encoded"]

    def observe(self, producer, *args, **kwargs):
        supplied = args[0] if args else kwargs.get("images")
        require(_pixels_fingerprint(supplied) == self.snapshot["capture"]["pixels"],
                "Replayed native preparation received different image pixels or device")
        self.calls += 1
        require(self.calls == 1, "Frozen replay consumed more than one image prefill")
        values = producer(*args, **kwargs)
        require(snapshot_encoded(values) == self.expected,
                "Returned replay features or mask differ from frozen realization")
        require([str(value.device) for value in values] ==
                self.snapshot["capture"]["fingerprints"]["encoded_devices"],
                "Returned replay feature/mask device changed")
        # The recipient may mutate a returned clone. Check it after the forward.
        self.returned.append(values)
        return values

    def finish(self):
        require(self.calls == 1, "Frozen replay did not consume exactly one image prefill")
        for values in self.returned:
            require(snapshot_encoded(values) == self.expected,
                    "Native preparation mutated returned replay features or mask")
        validate_snapshot(self.snapshot, self.arm, self.pixels)
        return {"status": "passed", "image_prefill_calls": self.calls,
                "returned_features_and_mask_exact": True,
                "frozen_snapshot_unchanged": True,
                "pixel_and_device_identity_exact": True}


@contextmanager
def _generation_replay(model, pixels, arm, snapshot):
    audit = _ReplayAudit(snapshot, arm, pixels)

    def cached(*args, **kwargs):
        return tuple(value.clone() for value in snapshot["encoded"])

    def replay(*args, **kwargs):
        return audit.observe(cached, *args, **kwargs)

    with replaced(model, "encode_images", replay):
        yield audit
    audit.finish()


def _score_replayed(tokenizer, model, case, pixels, arm, snapshot, target):
    """Use existing fixed scoring, observe its internal cached-image replay."""
    from run_rq2_input_merge import score_fixed_reference
    audit = _ReplayAudit(snapshot, arm, pixels)
    native_prepare = model.prepare_inputs_labels_for_multimodal_x

    def prepare(*args, **kwargs):
        # score_fixed_reference has installed its cached encoder by this point.
        cached_encoder = model.encode_images

        def observed(*encode_args, **encode_kwargs):
            return audit.observe(cached_encoder, *encode_args, **encode_kwargs)

        with replaced(model, "encode_images", observed):
            return native_prepare(*args, **kwargs)

    with replaced(model, "prepare_inputs_labels_for_multimodal_x", prepare):
        score = score_fixed_reference(tokenizer, model, case, pixels,
                                      snapshot["encoded"], arm, target)
    return score, audit.finish()


def generate_from_snapshot(tokenizer, model, processor, image, case, pixels,
                           arm, snapshot, seed, target):
    """Generate and score using the same native captured tuple, without CLIP/FPS.

    The existing native generation/scoring paths are retained. Each consumes one
    separately cloned prefill. Exact reuse checks apply within this realization;
    they do not assert that a fresh native merge would reproduce the same bytes.
    """
    from rq1_native import observe_generation_tokens
    from smoke_original_apet import generate_native_answer
    require(snapshot.get("capture", {}).get("seed") == seed,
            "Frozen snapshot seed differs from question seed")
    validate_snapshot(snapshot, arm, pixels)
    generation, tokens = {}, {}
    try:
        with _runtime_scope(model, arm, seed):
            with _generation_replay(model, pixels, arm, snapshot) as audit:
                with observe_generation_tokens(model, tokenizer, tokens):
                    generate_native_answer(tokenizer, model, processor, image,
                        case["generation_question"], arm_configuration(arm), seed=seed,
                        max_new_tokens=16, require_nonempty=False, record=generation)
            generation_control = audit.finish()
            score, scoring_control = _score_replayed(
                tokenizer, model, case, pixels, arm, snapshot, target)
    finally:
        # Also enforce immutability if a decoder/scorer throws an exception.
        validate_snapshot(snapshot, arm, pixels)
    return {"case_id": case["case_id"], "arm": arm,
            "response": generation["response"],
            "scoring": score_response(case, generation["response"]),
            "generation": generation, "generation_tokens": tokens,
            "encoding": copy.deepcopy(snapshot["capture"]["encoding"]),
            "reference_score": score,
            "frozen_replay": {"version": VERSION, "generation": generation_control,
                "reference_scoring": scoring_control,
                "fingerprints": copy.deepcopy(snapshot["capture"]["fingerprints"]),
                "note": "Generation and scoring share one native realization; timings include instrumentation"}}


def compare_realizations(reference_observation, reference_snapshot,
                         observed_observation, observed_snapshot,
                         reference_score=None, observed_score=None):
    """Numeric repeat evidence; source/selector changes remain explicit failures.

    This compares independent realizations of the same arm. It records native
    merge drift without turning its magnitude into a new acceptance tolerance.
    """
    from rq2_parity_diagnostic import compare_snapshots, score_comparison
    require(reference_snapshot["capture"]["arm"] == observed_snapshot["capture"]["arm"],
            "Independent-realization comparison requires the same arm")
    require(reference_snapshot["capture"]["pixels"] == observed_snapshot["capture"]["pixels"],
            "Independent-realization image identity differs")
    fields = ("source_clip", "source_patches", "pre_merge", "basis_slot_to_source", "input",
              "input_merge", "merge_argument_indices", "merge_basis_tokens", "merge_scaling")
    checks = {key + "_exact": reference_observation.get(key) == observed_observation.get(key)
              for key in fields}
    checks["mask_exact"] = (reference_observation["encoded"]["mask"] ==
                            observed_observation["encoded"]["mask"])
    require(all(checks.values()), "Independent native source/selection identity changed")
    result = {"identity_controls": {"status": "passed", **checks},
              "numeric_differences": compare_snapshots(reference_snapshot, observed_snapshot),
              "interpretation": "Independent native realizations; numeric differences are uncertainty evidence, not failed exact clone replay"}
    if reference_score is not None or observed_score is not None:
        require(reference_score is not None and observed_score is not None,
                "Independent score comparison requires both fixed-reference scores")
        result["reference_score_difference"] = score_comparison(reference_score, observed_score)
    return result
