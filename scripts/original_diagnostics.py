"""Observe values already computed by pinned native ApET, without editing its math.

Python return tracing reads native locals. It never assigns locals, replaces
functions, reruns FPS, or retains GPU tensors. GPU-to-CPU copies affect timings.
Use only in standalone diagnostic subprocesses, not alongside a debugger trace.
"""

from contextlib import contextmanager
import hashlib
import json
import math
import sys

SCHEMA_VERSION = "apet-native-locals-v1"


@contextmanager
def trace_returns(handlers):
    previous = sys.gettrace()
    if previous is not None:
        raise RuntimeError("Native diagnostics require a subprocess without an active Python debugger/trace")

    def local(frame, event, result):
        if event == "return" and result is not None:
            handlers[frame.f_code](frame.f_locals)
        return local

    def dispatch(frame, event, result):
        return local if event == "call" and frame.f_code in handlers else None

    sys.settrace(dispatch)
    try:
        yield
    finally:
        sys.settrace(previous)


def values(tensor):
    return tensor.detach().cpu().tolist()


def capture_scores(local, input_stage):
    if input_stage and "errors" not in local:
        return {"enabled": False, "source_tokens": local["N"]}
    errors = values(local["errors"])
    selected = values(local["errors_indices"])
    if input_stage:
        errors, selected = errors[0], selected[0]
    if not all(math.isfinite(error) for error in errors):
        raise RuntimeError("Non-finite native reconstruction errors")
    fps = values(local["fps_indices" if input_stage else "fps_idx"])
    record = {
        "enabled": True, "source_tokens": len(errors), "feature_dimension": local["seed_features"].shape[-1],
        "errors": errors, "basis_indices": [row[0] for row in fps[0]],
        "selected_indices": selected, "selected_indices_sorted": sorted(selected),
        "error_min": min(errors), "error_max": max(errors), "error_mean": sum(errors) / len(errors),
    }
    if not input_stage:
        record["layer"] = local["rank_layer"]
    return record


def capture_merge(local, input_stage):
    chosen = values(local["errors_indices"])
    mask = values(local["index_mask"])
    counts = values(local["merge_count"])
    nearest = values(local["nearest_token_indices"])
    if input_stage:
        chosen, mask, counts, nearest = chosen[0], mask[0], counts[0], nearest[0]
    k = local["k"]
    targets = chosen[:-k]
    dropped = [index for index, retained in enumerate(mask) if not retained]
    if len(dropped) != len(nearest) or sum(counts) != len(dropped):
        raise RuntimeError("Observed native merge assignments do not cover dropped tokens")
    basis_slots = chosen[-k:]
    return {"source_tokens": len(mask), "basis_slots": basis_slots,
            "basis_assignment_slots": sorted(basis_slots) if input_stage else basis_slots,
            "target_indices": targets,
            "dropped_indices": dropped, "assigned_target_indices": [targets[index] for index in nearest],
            "merge_counts": counts, "scaling": local["scaling"]}


@contextmanager
def observe_native_values():
    from llava.model.llava_arch import LlavaMetaForCausalLM
    from llava.model.modeling_llama_x import LlamaModel, token_merging

    data = {"schema": SCHEMA_VERSION, "decoder": [], "decoder_merge": []}
    handlers = {
        LlavaMetaForCausalLM.encode_images.__code__: lambda local: data.update(input=capture_scores(local, True)),
        LlavaMetaForCausalLM.token_merging.__code__: lambda local: data.update(input_merge=capture_merge(local, True)),
        LlamaModel.layer_prune.__code__: lambda local: data["decoder"].append(capture_scores(local, False)),
        token_merging.__code__: lambda local: data["decoder_merge"].append(capture_merge(local, False)),
    }
    with trace_returns(handlers):
        yield data


def validate_diagnostics(settings, data):
    input_stage = data.get("input", {})
    if input_stage.get("source_tokens") != 576:
        raise RuntimeError("Native input diagnostic was not observed")
    if settings["input_tokens"] < 576:
        if len(input_stage.get("selected_indices", [])) != settings["input_tokens"] or "input_merge" not in data:
            raise RuntimeError("Compressed input diagnostics incomplete")
        input_stage["basis_slot_to_source"] = list(zip(data["input_merge"]["basis_assignment_slots"], input_stage["basis_indices"]))
    elif input_stage.get("enabled"):
        raise RuntimeError("Baseline unexpectedly recorded input reconstruction")
    if [stage["layer"] for stage in data["decoder"]] != settings["layers"]:
        raise RuntimeError("Decoder reconstruction diagnostics incomplete")
    if len(data["decoder_merge"]) != len(settings["layers"]):
        raise RuntimeError("Decoder merge diagnostics incomplete")
    for stage, merge, keep in zip(data["decoder"], data["decoder_merge"], settings["decoder_tokens"]):
        if len(stage["selected_indices"]) != keep:
            raise RuntimeError("Decoder diagnostic budget differs")
        stage["basis_slot_to_source"] = list(zip(merge["basis_assignment_slots"], stage["basis_indices"]))
    data["fingerprint"] = hashlib.sha256(json.dumps(data, sort_keys=True, allow_nan=False).encode()).hexdigest()


def compare_diagnostics(before, after, atol=1e-4, rtol=1e-5):
    if len(before["decoder"]) != len(after["decoder"]) or len(before["decoder_merge"]) != len(after["decoder_merge"]):
        raise ValueError("Same-image decoder observation counts differ")

    def compare_stage(first, second, merge_first, merge_second):
        if first.get("enabled") != second.get("enabled"):
            raise ValueError("Same-image compression stages differ")
        if not first.get("enabled"):
            return {"enabled": False}
        if len(first["errors"]) != len(second["errors"]):
            raise ValueError("Same-image stage token counts differ")
        return {"enabled": True,
                "selected_set_equal": first["selected_indices_sorted"] == second["selected_indices_sorted"],
                "basis_indices_equal": first["basis_indices"] == second["basis_indices"],
                "basis_slots_equal": merge_first["basis_assignment_slots"] == merge_second["basis_assignment_slots"],
                "merge_assignments_equal": merge_first == merge_second,
                "errors_max_abs_difference": max(abs(a - b) for a, b in zip(first["errors"], second["errors"])),
                "errors_close": all(math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
                                    for a, b in zip(first["errors"], second["errors"]))}

    return {"input": compare_stage(before["input"], after["input"], before.get("input_merge"), after.get("input_merge")),
            "decoder": [compare_stage(a, b, ma, mb) for a, b, ma, mb in
                        zip(before["decoder"], after["decoder"], before["decoder_merge"], after["decoder_merge"])],
            "tolerance": {"atol": atol, "rtol": rtol}}
