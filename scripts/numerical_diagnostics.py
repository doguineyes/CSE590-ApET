"""Read native feature tensors and generation scores without changing their math.

Snapshots are plain NumPy arrays, deduplicated by dtype/shape/value hash. CPU
copies synchronize execution and affect timing. Use only in a diagnostic process.
"""

from contextlib import contextmanager
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys

from original_diagnostics import capture_merge, capture_scores


POLICIES = {
    "current": {"workspace": None, "deterministic": False},
    "workspace-only": {"workspace": ":4096:8", "deterministic": False},
    "deterministic": {"workspace": ":4096:8", "deterministic": True},
}
STAGE_ORDER = ["vision_patches", "merge_before", "scatter_initial", "scatter_sources", "scatter_targets",
               "scatter_after_add", "merge_after", "projected_selected", "first_step_scores"]


def policy_environment(name, inherited):
    """Return a child-process environment; never modify the notebook's kernel."""
    result = dict(inherited)
    workspace = POLICIES[name]["workspace"]
    if workspace is None:
        result.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        result["CUBLAS_WORKSPACE_CONFIG"] = workspace
    return result


class TensorStore:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "tensors").mkdir(parents=True, exist_ok=True)
        self.file_hashes = {}

    def save_array(self, array, device="cpu"):
        import numpy as np

        array = np.ascontiguousarray(array)
        raw = array.tobytes()
        header = json.dumps({"dtype": array.dtype.str, "shape": list(array.shape)}, sort_keys=True).encode()
        identity = hashlib.sha256(header + b"\n" + raw).hexdigest()
        relative = Path("tensors") / (identity + ".npy")
        path = self.root / relative
        if not path.exists():
            with path.open("xb") as stream:
                np.save(stream, array, allow_pickle=False)
        if identity not in self.file_hashes:
            self.file_hashes[identity] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"file": str(relative), "content_id": identity, "values_sha256": hashlib.sha256(raw).hexdigest(),
                "file_sha256": self.file_hashes[identity], "dtype": array.dtype.str,
                "shape": list(array.shape), "device": str(device), "bytes": len(raw),
                "nonfinite_values": int(np.count_nonzero(~np.isfinite(array)))}

    def save_tensor(self, tensor):
        return self.save_array(tensor.detach().cpu().contiguous().numpy(), tensor.device)

    def load(self, record):
        import numpy as np

        relative = Path(record["file"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Tensor files must be within the run directory")
        path = self.root / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["file_sha256"]:
            raise ValueError("Saved tensor file hash differs")
        array = np.load(path, allow_pickle=False)
        if (list(array.shape) != record["shape"] or array.dtype.str != record["dtype"]
                or hashlib.sha256(array.tobytes()).hexdigest() != record["values_sha256"]):
            raise ValueError("Saved tensor values/shape/dtype differ")
        return array

    def compare(self, first, second):
        import numpy as np

        if first["shape"] != second["shape"] or first["dtype"] != second["dtype"]:
            return {"exact_equal": False, "comparable": False, "reason": "shape/dtype differ"}
        if first["values_sha256"] == second["values_sha256"]:
            return {"exact_equal": True, "comparable": True, "max_abs_difference": 0.0,
                    "different_values": 0, "nonfinite_disagreements": 0}
        a, b = self.load(first), self.load(second)
        finite = np.isfinite(a) & np.isfinite(b)
        equal = a == b
        nonfinite = int(np.count_nonzero(~finite & ~equal))
        maximum = float(np.max(np.abs(a[finite].astype(np.float64) - b[finite].astype(np.float64)))) if finite.any() else 0.0
        return {"exact_equal": bool(equal.all()), "comparable": True,
                "max_abs_difference": None if nonfinite else maximum,
                "different_values": int(np.count_nonzero(~equal)), "nonfinite_disagreements": nonfinite}


def score_summary(scores, sequence_ids, generated_count, tokenizer):
    """Describe the actual generation's first processed score vector."""
    import numpy as np

    vector = np.asarray(scores).reshape(-1)
    if np.isnan(vector).any() or generated_count < 1 or len(sequence_ids) < generated_count:
        raise ValueError("Invalid first-step scores or generated sequence")
    ordered = np.argsort(-vector)[:10].tolist()

    def number(value):
        value = float(value)
        return value if math.isfinite(value) else str(value)

    candidates = {}
    for word in ["Yes", "No", "yes", "no", "Left", "Right"]:
        ids = tokenizer.encode(word, add_special_tokens=False)
        candidates[word] = {"token_ids": ids, "single_token": len(ids) == 1,
                            "first_token_score": number(vector[ids[0]]) if ids else None}
    margin = None
    if candidates["Yes"]["single_token"] and candidates["No"]["single_token"]:
        a, b = candidates["Yes"]["token_ids"][0], candidates["No"]["token_ids"][0]
        if np.isfinite(vector[a]) and np.isfinite(vector[b]):
            margin = float(vector[a]) - float(vector[b])
    return {"sequence_ids": list(sequence_ids), "generated_ids": list(sequence_ids[-generated_count:]),
            "first_generated_id": int(sequence_ids[-generated_count]),
            "score_argmax_id": int(ordered[0]),
            "top_tokens": [{"id": int(i), "text": tokenizer.decode([i]), "score": number(vector[i])} for i in ordered],
            "top1_minus_top2": number(float(vector[ordered[0]]) - float(vector[ordered[1]])),
            "yes_minus_no_first_token_score": margin, "candidates": candidates,
            "note": "Scores after generation logits processors; not whole-answer probabilities. "
                    "Candidate scores refer to each encoded label's first token."}


@contextmanager
def observe_numerical_values(model, tokenizer, store):
    from llava.model.llava_arch import LlavaMetaForCausalLM

    if sys.gettrace() is not None:
        raise RuntimeError("Numerical observations need a subprocess without debugger tracing")
    data = {"stages": {}, "diagnostics": {"schema": "apet-native-locals-v1", "decoder": [], "decoder_merge": []}}
    encode_code = LlavaMetaForCausalLM.encode_images.__code__
    merge_code = LlavaMetaForCausalLM.token_merging.__code__
    source, first_line = inspect.getsourcelines(LlavaMetaForCausalLM.token_merging)
    scatter_line = next(first_line + i for i, line in enumerate(source)
                        if line.strip().startswith("merged_features.scatter_add_("))
    count_line = next(first_line + i for i, line in enumerate(source)
                      if line.strip().startswith("merge_count.scatter_add_("))

    def save(name, tensor):
        if name in data["stages"]:
            raise RuntimeError(f"Unexpected repeated feature stage: {name}")
        data["stages"][name] = store.save_tensor(tensor)

    def vision_hook(module, inputs, output):
        # Native encode_images removes the leading CLS token.
        save("vision_patches", output[0][:, 1:])

    def local(frame, event, result):
        if frame.f_code is merge_code and event == "line":
            if frame.f_lineno == scatter_line:
                for name, variable in [("scatter_initial", "merged_features"),
                                       ("scatter_sources", "non_retained_tokens"),
                                       ("scatter_targets", "nearest_token_indices")]:
                    save(name, frame.f_locals[variable])
                data["scatter_input"] = {"initial": data["stages"]["scatter_initial"],
                    "sources": data["stages"]["scatter_sources"], "targets": data["stages"]["scatter_targets"], "dim": 1}
            elif frame.f_lineno == count_line:
                save("scatter_after_add", frame.f_locals["merged_features"])
        if event == "return" and result is not None:
            if frame.f_code is merge_code:
                save("merge_after", result)
                data["diagnostics"]["input_merge"] = capture_merge(frame.f_locals, True)
            else:
                data["diagnostics"]["input"] = capture_scores(frame.f_locals, True)
                features, mask = result
                save("projected_selected", features[mask])
        return local

    def dispatch(frame, event, result):
        if event == "call" and frame.f_code is merge_code:
            save("merge_before", frame.f_locals["image_features"])
            data["merge_input"] = {
                "features": data["stages"]["merge_before"],
                "selected_indices": store.save_tensor(frame.f_locals["errors_indices"]),
                "basis_tokens": frame.f_locals["k"], "scaling": frame.f_locals.get("scaling", 1),
            }
            return local
        if event == "call" and frame.f_code is encode_code:
            return local
        return None

    original = model.generate
    had_instance_generate = "generate" in model.__dict__
    instance_generate = model.__dict__.get("generate")
    hook = None

    def generated(*args, **kwargs):
        output = original(*args, **kwargs)
        if not output.scores:
            raise RuntimeError("Generation did not return first-step scores")
        first = output.scores[0][0].detach().float().cpu().numpy()
        data["stages"]["first_step_scores"] = store.save_array(first, output.scores[0].device)
        data["generation_scores"] = score_summary(first, output.sequences[0].detach().cpu().tolist(),
                                                   len(output.scores), tokenizer)
        if data["generation_scores"]["first_generated_id"] != data["generation_scores"]["score_argmax_id"]:
            raise RuntimeError("First generated token differs from the captured greedy score argmax")
        return output

    try:
        hook = model.get_vision_tower().register_forward_hook(vision_hook)
        model.generate = generated
        sys.settrace(dispatch)
        yield data
    finally:
        sys.settrace(None)
        if hook is not None:
            hook.remove()
        if had_instance_generate:
            model.generate = instance_generate
        elif "generate" in model.__dict__:
            delattr(model, "generate")
