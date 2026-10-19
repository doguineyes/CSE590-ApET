"""Observe native input ApET, or disable only its token-merging call.

The pinned ``LlavaMetaForCausalLM.encode_images`` still computes FPS, E, ranked
selection, basis replacement and projection. No-merge returns the *incoming*
merge tensor unchanged; in particular, it retains the basis replacement that
native encode_images has already performed. All patches are instance-scoped and
restore on exceptions. Hashing/copying instrumentation invalidates speed claims.
"""

from contextlib import contextmanager, ExitStack
import hashlib

from original_diagnostics import capture_merge, capture_scores, trace_returns
from rq1_native import replaced

SCHEMA_VERSION = "rq2-native-input-v1"
VARIANTS = ("native", "no-merge")


def tensor_fingerprint(tensor):
    """Hash raw CPU bytes, including FP16/BF16, without numeric conversion."""
    import torch
    cpu = tensor.detach().contiguous().cpu()
    return {"shape": list(cpu.shape), "dtype": str(cpu.dtype),
            "sha256": hashlib.sha256(cpu.view(torch.uint8).numpy().tobytes()).hexdigest()}


def snapshot_encoded(output):
    """JSON-only feature/mask snapshot for unchanged-native parity controls."""
    import torch
    if not isinstance(output, tuple) or len(output) != 2:
        raise ValueError("Native encode_images must return (projected_features, mask)")
    features, mask = output
    if features.ndim != 3 or mask.ndim != 2 or features.shape[:2] != mask.shape:
        raise ValueError("Native projected-feature and mask geometry differ")
    if features.shape[0] != 1 or mask.dtype != torch.bool:
        raise ValueError("RQ2 pilot requires a single-image boolean selection mask")
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite native projected features")
    return {"features": tensor_fingerprint(features), "mask": tensor_fingerprint(mask),
            "selected_indices_sorted": mask[0].nonzero().flatten().cpu().tolist(),
            "kept_tokens": int(mask.sum().item())}


def assert_native_parity(reference, observed, reference_token_ids, observed_token_ids):
    """Exact feature/mask and generated-ID gate; decoded-text equality is weaker."""
    if reference_token_ids is None or observed_token_ids is None:
        raise ValueError("Native parity requires generated token IDs, not text alone")
    checks = {"features_exact": reference["features"] == observed["features"],
              "mask_exact": reference["mask"] == observed["mask"],
              "selected_positions_exact": reference["selected_indices_sorted"] == observed["selected_indices_sorted"],
              "generated_ids_exact": reference_token_ids == observed_token_ids}
    if not all(checks.values()):
        raise RuntimeError("Instrumented native parity failed: " +
                           ", ".join(key for key, passed in checks.items() if not passed))
    return {"status": "passed", **checks}


def assert_selection_equal(native_call, no_merge_call):
    """Prove that no-merge changed neither selection nor basis replacement."""
    keys = ("source_clip", "source_patches", "pre_merge", "basis_slot_to_source")
    checks = {key + "_exact": native_call.get(key) == no_merge_call.get(key) for key in keys}
    for key in ("errors", "basis_indices", "selected_indices", "selected_indices_sorted"):
        checks[key + "_exact"] = native_call["input"][key] == no_merge_call["input"][key]
    checks["mask_exact"] = native_call["encoded"]["mask"] == no_merge_call["encoded"]["mask"]
    checks["no_merge_identity_exact"] = no_merge_call["pre_merge"] == no_merge_call["post_merge"]
    if not native_call["input"].get("enabled") or not no_merge_call["input"].get("enabled"):
        raise ValueError("Selection comparison requires two compressed input passes")
    if not all(checks.values()):
        raise RuntimeError("Native/no-merge selection control failed: " +
                           ", ".join(key for key, passed in checks.items() if not passed))
    return {"status": "passed", **checks}


def _validate_call(call, variant):
    stage = call["input"]
    encoded = call["encoded"]
    if stage["source_tokens"] != 576 or encoded["features"]["shape"][1] != 576:
        raise RuntimeError("RQ2 pilot expects exactly 576 native source patch states")
    if "source_clip" not in call or call["source_clip"]["shape"][1] != 577:
        raise RuntimeError("Native source CLIP features were not captured")
    if not stage.get("enabled"):
        if variant != "native" or encoded["kept_tokens"] != 576 or call.get("merge_calls", 0):
            raise RuntimeError("Unexpected no-merge baseline or native merge call")
        call["basis_slot_to_source"] = []
        return
    if call.get("merge_calls") != 1:
        raise RuntimeError("Compressed native encode_images did not invoke one merge call")
    ranked = stage["selected_indices"]
    basis = stage["basis_indices"]
    slots = sorted(ranked[-len(basis):])
    if len(set(ranked)) != len(ranked) or sorted(ranked) != encoded["selected_indices_sorted"]:
        raise RuntimeError("Native reconstruction selection differs from returned mask")
    if encoded["kept_tokens"] != len(ranked):
        raise RuntimeError("Native output budget differs from ranked E selection")
    # Native boolean assignment visits slots in ascending coordinate order, not
    # the descending-E ordering of errors_indices[:, -k:]. Sources stay FPS-ordered.
    compact = {source_position: output_position
               for output_position, source_position in enumerate(sorted(ranked))}
    call["basis_slot_to_source"] = [
        {"source_index": source, "assignment_slot": slot, "compact_output_position": compact[slot]}
        for slot, source in zip(slots, basis)]
    if variant == "native":
        merge = call.get("input_merge")
        if not merge or merge["basis_assignment_slots"] != slots:
            raise RuntimeError("Native merge assignment diagnostics missing or inconsistent")
    elif call["pre_merge"] != call["post_merge"]:
        raise RuntimeError("Disabled merge modified its incoming tensor")


@contextmanager
def native_input_variant(model, variant="native", *, capture_tensors=False):
    """Yield per-encode observations during an existing native generation call.

Caller configures input budget/FPS seed and resets that seed before each paired
generation. Decoder compression must be disabled. This context neither replays
CLIP nor freezes selectors: exact paired-source/selection checks must pass before
    answer differences can be attributed to merging. Optional ``snapshots`` hold
    detached clones for evidence or teacher forcing and must be removed before
    JSON serialization. No-merge intentionally omits
merge destinations because that operation was not executed in this branch.
"""
    from llava.model.llava_arch import LlavaMetaForCausalLM
    if variant not in VARIANTS:
        raise ValueError("Unknown RQ2 input variant: " + str(variant))
    native_encode = model.encode_images
    native_merge = model.token_merging
    if getattr(native_encode, "__func__", None) is not LlavaMetaForCausalLM.encode_images:
        raise ValueError("RQ2 requires the unchanged pinned native encode_images method")
    if getattr(native_merge, "__func__", None) is not LlavaMetaForCausalLM.token_merging:
        raise ValueError("RQ2 requires the unchanged pinned native token_merging method")
    if getattr(model.get_model(), "layer_list", []):
        raise ValueError("RQ2 input-only pilot requires decoder compression disabled")
    tower = model.get_vision_tower()
    native_vision = tower.forward
    data = {"schema": SCHEMA_VERSION, "variant": variant, "calls": [], "status": "running",
            "timing_note": "Feature hashing and Python tracing affect timings; this is not a speed benchmark"}
    if capture_tensors:
        data["snapshots"] = []
    current = []
    current_snapshot = []

    def active():
        if len(current) != 1:
            raise RuntimeError("RQ2 observation requires one active native image encoding")
        return current[0]

    def vision(*args, **kwargs):
        output = native_vision(*args, **kwargs)
        call = active()
        if "source_clip" in call:
            raise RuntimeError("Native encode_images invoked vision tower more than once")
        call["source_clip"] = tensor_fingerprint(output[0])
        call["source_patches"] = tensor_fingerprint(output[0][:, 1:])
        return output

    def merge(features, errors_indices, k, scaling=1):
        call = active()
        call["merge_calls"] += 1
        call["native_merge_executed"] = variant == "native"
        call["pre_merge"] = tensor_fingerprint(features)
        if capture_tensors:
            current_snapshot[0]["pre_merge"] = features.detach().clone()
        call["merge_argument_indices"] = errors_indices.detach().cpu().tolist()
        call["merge_basis_tokens"] = int(k)
        call["merge_scaling"] = scaling
        # Basis vectors were already assigned in native encode_images above us.
        output = native_merge(features, errors_indices, k, scaling=scaling) if variant == "native" else features
        call["post_merge"] = tensor_fingerprint(output)
        if capture_tensors:
            current_snapshot[0]["post_merge"] = output.detach().clone()
        return output

    def encode(*args, **kwargs):
        if kwargs.get("selection_method", "fps") != "fps" or len(args) > 1 and args[1] != "fps":
            raise ValueError("RQ2 pilot requires native FPS basis selection")
        if current:
            raise RuntimeError("Nested native image encoding is unsupported")
        call = {"variant": variant, "merge_calls": 0, "native_merge_executed": False}
        current.append(call)
        data["calls"].append(call)
        if capture_tensors:
            snapshot = {}
            current_snapshot.append(snapshot)
            data["snapshots"].append(snapshot)
        try:
            output = native_encode(*args, **kwargs)
            call["encoded"] = snapshot_encoded(output)
            if capture_tensors:
                current_snapshot[0]["encoded"] = tuple(item.detach().clone() for item in output)
            _validate_call(call, variant)
            call["status"] = "passed"
            return output
        except BaseException:
            call["status"] = "failed"
            raise
        finally:
            current.pop()
            if capture_tensors:
                current_snapshot.pop()

    handlers = {
        LlavaMetaForCausalLM.encode_images.__code__: lambda local: active().update(input=capture_scores(local, True)),
        LlavaMetaForCausalLM.token_merging.__code__: lambda local: active().update(input_merge=capture_merge(local, True)),
    }
    try:
        with ExitStack() as stack:
            stack.enter_context(replaced(tower, "forward", vision))
            stack.enter_context(replaced(model, "encode_images", encode))
            stack.enter_context(replaced(model, "token_merging", merge))
            stack.enter_context(trace_returns(handlers))
            yield data
        if not data["calls"]:
            raise RuntimeError("RQ2 context observed no native image encoding")
        data["status"] = "passed"
    except BaseException:
        data["status"] = "failed"
        raise
