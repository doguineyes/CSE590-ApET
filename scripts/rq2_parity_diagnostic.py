"""Bounded numerical evidence after an exact RQ2 parity failure.

This diagnostic never changes native code, tolerances or the scientific gate.
Raw native encoding is observed only after it returns. The existing heavier
wrapper additionally captures the already computed native merge intermediates.
The nine passes and fixed-answer scores are numerical controls, not benchmark
predictions. Tensor loading/computation belongs to the original Torch runtime.
"""

from contextlib import contextmanager
import math

from rq1_protocol import sha256, write_json
from rq2_protocol import require

VERSION = "rq2-native-parity-diagnostic-v1"
PASS_PLAN = (("raw-native-1", None), ("raw-native-2", None),
             ("raw-native-3", None), ("observed-native-1", "native"),
             ("observed-native-2", "native"), ("observed-native-3", "native"),
             ("observed-no-merge-1", "no-merge"),
             ("observed-no-merge-2", "no-merge"),
             ("generation-native", "generation-native"))


def selection_comparison(reference, observed, expected_budget=288):
    """Compare coordinate sets without treating a mismatch as accepted parity."""
    for values in (reference, observed):
        require(isinstance(values, list) and len(values) == len(set(values))
                and all(type(x) is int and 0 <= x < 576 for x in values),
                "Invalid diagnostic selected-position list")
    left, right = set(reference), set(observed)
    return {"exact": reference == observed, "same_coordinates": left == right,
            "reference_count": len(reference), "observed_count": len(observed),
            "reference_expected_budget": len(reference) == expected_budget,
            "observed_expected_budget": len(observed) == expected_budget,
            "reference_only": sorted(left-right), "observed_only": sorted(right-left)}


def score_comparison(reference, observed):
    """Fixed-token log-probability deltas; no claim of answer correctness."""
    left, right = reference["token_logprobs"], observed["token_logprobs"]
    require(len(left) == len(right) > 0 and all(math.isfinite(x) for x in left+right),
            "Invalid diagnostic reference-score geometry or values")
    delta = [b-a for a, b in zip(left, right)]
    return {"exact": left == right, "token_logprob_deltas_observed_minus_reference": delta,
            "max_abs_token_logprob_delta": max(abs(x) for x in delta),
            "mean_logprob_delta": sum(delta)/len(delta),
            "first_token_probability_delta": math.exp(right[0])-math.exp(left[0])}


def tensor_delta(reference, observed):
    """Actual numeric drift in detached CPU float64, never hash-only distance.

    A token is a coordinate on the penultimate axis; differences anywhere in its
    complete feature vector count once. Relative L2 has a null value when the
    reference norm is zero and drift is nonzero (avoids an infinite JSON value).
    """
    import torch
    require(reference.shape == observed.shape and reference.ndim == 3
            and reference.numel() > 0, "Incompatible diagnostic tensor geometry")
    a = reference.detach().to(device="cpu", dtype=torch.float64)
    b = observed.detach().to(device="cpu", dtype=torch.float64)
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    require(finite, "Non-finite diagnostic features")
    diff = b-a
    changed = a != b
    norm_reference = float(torch.linalg.vector_norm(a).item())
    norm_delta = float(torch.linalg.vector_norm(diff).item())
    return {"shape": list(a.shape), "reference_dtype": str(reference.dtype),
            "observed_dtype": str(observed.dtype), "both_finite": finite,
            "exact": reference.dtype == observed.dtype and
                     bool(torch.equal(reference.detach().cpu(), observed.detach().cpu())),
            "max_abs": float(diff.abs().max().item()),
            "mean_abs": float(diff.abs().mean().item()),
            "rms": float(diff.square().mean().sqrt().item()),
            "reference_l2": norm_reference, "delta_l2": norm_delta,
            "relative_l2": norm_delta/norm_reference if norm_reference else
                           (0.0 if norm_delta == 0 else None),
            "relative_l2_defined": bool(norm_reference or norm_delta == 0),
            "changed_elements": int(changed.sum().item()),
            "total_elements": a.numel(),
            "changed_tokens": int(changed.any(dim=-1).sum().item()),
            "total_tokens": a.shape[0]*a.shape[1],
            "calculation": "detached CPU float64; observed minus reference"}


def compare_snapshots(reference, observed):
    """Full and selected-coordinate numeric evidence, including mask drift."""
    left_mask, right_mask = reference["encoded"][1], observed["encoded"][1]
    left_indices = left_mask[0].nonzero().flatten().tolist()
    right_indices = right_mask[0].nonzero().flatten().tolist()
    comparison = {"selection": selection_comparison(left_indices, right_indices),
                  "stages": {}}
    for name in ("pre_merge", "post_merge", "projected"):
        left = reference["encoded"][0] if name == "projected" else reference.get(name)
        right = observed["encoded"][0] if name == "projected" else observed.get(name)
        if left is None or right is None:
            continue
        stage = {"full576": tensor_delta(left, right)}
        # A changed mask must not silently realign two unrelated compact arrays.
        # Compare the same original coordinates under both masks separately.
        if left_indices:
            stage["reference_selected_coordinates"] = tensor_delta(
                left[:, left_indices], right[:, left_indices])
        excluded = [index for index in range(576) if index not in set(left_indices)]
        if excluded:
            stage["reference_excluded_coordinates"] = tensor_delta(
                left[:, excluded], right[:, excluded])
        if right_indices != left_indices and right_indices:
            stage["observed_selected_coordinates"] = tensor_delta(
                left[:, right_indices], right[:, right_indices])
        comparison["stages"][name] = stage
    return comparison


def _cpu_snapshot(snapshot):
    return {name: tuple(value.detach().cpu().clone() for value in tensor)
            if name == "encoded" else tensor.detach().cpu().clone()
            for name, tensor in snapshot.items()}


def _reset(model, seed):
    import torch
    from smoke_original_apet import configure
    configure(model, "input-only")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@contextmanager
def _preserve_tensors(path, tensors, report, diagnostic, summary_path):
    """Save the completed capture prefix even if a later pass or score fails."""
    import torch
    try:
        yield
    except BaseException as error:
        diagnostic.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        torch.save(tensors, path)
        relative = str(path.relative_to(summary_path.parent))
        diagnostic.update(tensor_file=relative, tensor_sha256=sha256(path))
        report["feature_sha256"][relative] = diagnostic["tensor_sha256"]
        write_json(summary_path, diagnostic)
        write_json(summary_path.parent / "report.json", report)


def capture_pass(model, pixels, seed, variant):
    """One raw or existing instrumented native pass, with all patches restored."""
    import torch
    from rq2_native import native_input_variant, snapshot_encoded
    _reset(model, seed)
    if variant is None:
        with torch.inference_mode():
            output = model.encode_images(pixels)
        return {"encoded": tuple(t.detach().clone() for t in output)}, {
            "observation": "Unwrapped native encode; output inspected after return",
            "encoded": snapshot_encoded(output)}
    with native_input_variant(model, variant, capture_tensors=True) as observed:
        with torch.inference_mode():
            model.encode_images(pixels)
    snapshots = observed.pop("snapshots")
    require(len(snapshots) == len(observed["calls"]) == 1,
            "Diagnostic requires one native image call per pass")
    return snapshots[0], observed["calls"][0]


def run_diagnostic(args, report, profile, manifests, records, loaded):
    """Run only the first frozen case; successful execution is not pilot parity."""
    import torch
    from PIL import Image
    from rq1_native import encode_image
    from rq2_native import tensor_fingerprint
    from run_rq1_input import save_image
    # Lazy import: the runner dispatches here, while retaining its scoring helper.
    from run_rq2_input_merge import run_arm, score_fixed_reference

    require(records and records[0]["case"]["case_id"] == profile["case_ids"][0],
            "Diagnostic first frozen question differs")
    item = records[0]
    case, archived = item["case"], item["reference"]
    cid, seed = case["case_id"], archived["seed"]
    tokenizer, model, processor = loaded
    report.update(diagnostic_only=True, scientific_pilot_passed=False,
                  expected_case_ids=[cid], questions=[], question_sha256={}, feature_sha256={},
                  progress={"completed_passes": 0, "expected_passes": len(PASS_PLAN)})
    args.manifest = args.inputs / cid.split(":")[0] / "manifest.json"
    image = save_image(args, manifests[cid.split(":")[0]], case)
    encoded = encode_image(model, processor, image, seed)
    for key in ("clip_sha256", "projected_sha256", "seed"):
        require(encoded[key] == archived[key], "Archived baseline source changed: " + key)
    canvas = args.output / "images" / (case["image_sha256"] + "-canvas.png")
    Image.fromarray(encoded["canvas"]).save(canvas)
    diagnostic = {"version": VERSION, "status": "running", "case_id": cid,
        "seed": seed, "archived_baseline_source_exact": True,
        "scientific_pilot_passed": False, "passes": {}, "comparisons": {},
        "scope": "Eight single-image encoding repeats plus one observed native generation; diagnostic controls, no benchmark predictions",
        "limitations": [
            "Raw native passes expose only returned projected features and masks; no upstream source or merge intermediates are observed in those passes.",
            "Heavy observation adds copies, hashes and tracing; repeat groups cannot alone establish the cause of any numerical difference.",
            "No-merge differences from native are intentional. Diagnostic completion neither relaxes exact parity nor authorizes a numerical tolerance.",
            "Teacher-forced gold scores diagnose numerical propagation. The single generation reproduces the original observation context; raw-native generation equality is not tested here."]}
    report["diagnostic"] = diagnostic
    tensors = {"baseline": {"clip": encoded["clip"].detach().cpu(),
                            "projected": encoded["projected"].detach().cpu(),
                            "pixels": encoded["pixels"].detach().cpu()}, "passes": {}}
    feature_dir = args.output / "features"
    feature_dir.mkdir(exist_ok=True)
    summary_path = args.output / "parity-diagnostic.json"
    tensor_path = feature_dir / "parity-diagnostic.pt"
    with _preserve_tensors(tensor_path, tensors, report, diagnostic, summary_path):
        for name, variant in PASS_PLAN:
            print(f"{cid}: diagnostic {name}", flush=True)
            generation_row = None
            if variant == "generation-native":
                _reset(model, seed)
                generation_row, snapshot = run_arm(tokenizer, model, processor, image,
                    case, encoded["pixels"], "native-input288", archived["measurement_target"], seed)
                observation = generation_row["encoding"]
            else:
                snapshot, observation = capture_pass(model, encoded["pixels"], seed, variant)
            budget = int(snapshot["encoded"][1].sum().item())
            row = {"variant": variant or "unwrapped-native", "observation": observation,
                   "actual_budget": budget, "expected_budget": 288,
                   "budget_matches": budget == 288}
            if budget == 288:
                score = generation_row["reference_score"] if generation_row else score_fixed_reference(tokenizer, model, case, encoded["pixels"],
                    snapshot["encoded"], "native-input288-no-merge" if variant == "no-merge"
                    else "native-input288", archived["measurement_target"])
                require(score["target_token_ids"] == archived["target_token_ids"]
                        and score["prompt"] == archived["prompt"], "Frozen diagnostic target/prompt changed")
                row["reference_score"] = score
            else:
                row["reference_score_skipped"] = "Unexpected budget; frozen 288-token scorer would be structurally invalid"
            if generation_row:
                row["generation"] = generation_row["generation"]
                row["generation_tokens"] = generation_row["generation_tokens"]
                row["response"] = generation_row["response"]
                row["interpretation"] = "One context-reproduction control, not a passed RQ2 prediction"
            if "input" in observation:
                prior_stage = item["feature"]["diagnostics"]["input"]
                row["archived_reconstruction_checks"] = {
                    key: observation["input"].get(key) == prior_stage.get(key)
                    for key in ("errors", "basis_indices")}
            tensors["passes"][name] = _cpu_snapshot(snapshot)
            diagnostic["passes"][name] = row
            report["progress"]["completed_passes"] += 1
            write_json(summary_path, diagnostic)
            write_json(args.output / "report.json", report)
            del snapshot
        pairs = [("raw-native-1", "raw-native-2"), ("raw-native-1", "raw-native-3"),
                 ("raw-native-1", "observed-native-1"),
                 ("observed-native-1", "observed-native-2"),
                 ("observed-native-1", "observed-native-3"),
                 ("observed-native-1", "observed-no-merge-1"),
                 ("observed-no-merge-1", "observed-no-merge-2"),
                 ("raw-native-1", "generation-native"),
                 ("observed-native-1", "generation-native")]
        for first, second in pairs:
            comparison = compare_snapshots(tensors["passes"][first], tensors["passes"][second])
            left, right = diagnostic["passes"][first], diagnostic["passes"][second]
            if "reference_score" in left and "reference_score" in right:
                comparison["fixed_reference_score"] = score_comparison(
                    left["reference_score"], right["reference_score"])
            if "input" in left["observation"] and "input" in right["observation"]:
                comparison["heavy_exact_checks"] = {
                    key: left["observation"].get(key) == right["observation"].get(key)
                    for key in ("source_clip", "source_patches", "basis_slot_to_source")}
                comparison["heavy_exact_checks"].update({key: left["observation"]["input"].get(key)
                    == right["observation"]["input"].get(key)
                    for key in ("errors", "basis_indices", "selected_indices")})
            comparison["intentional_merge_ablation"] = second == "observed-no-merge-1" and first == "observed-native-1"
            diagnostic["comparisons"][first + "__" + second] = comparison
    diagnostic.update(status="passed", completion_meaning="Diagnostic executed; exact scientific parity is not asserted",
                      tensor_inventory={name: {key: [tensor_fingerprint(v) for v in value]
                          if key == "encoded" else tensor_fingerprint(value)
                          for key, value in snapshot.items()}
                          for name, snapshot in tensors["passes"].items()})
    write_json(summary_path, diagnostic)
    report["diagnostic_sha256"] = sha256(summary_path)
    write_json(args.output / "report.json", report)
