"""Budget-aware capture/replay and E-free random masks; pinned native files untouched.

Pure random executes the native full576 vision/projector branch, then selects
original projected states with a boolean mask. Projector work is not optimized:
this is an answer-quality calibration, not a compression-overhead benchmark.
"""
from contextlib import contextmanager
import copy

from rq1_native import replaced
from rq2_native import native_input_variant, snapshot_encoded, tensor_fingerprint
from rq2_protocol import require, score_response
from rq2_random_protocol import VERSION, specification, random_selection


@contextmanager
def runtime_scope(model, budget, seed):
    """Temporarily configure a new input budget using the existing native helper."""
    import torch
    import smoke_original_apet as smoke
    require(type(budget) is int and budget in (64, 288, 576), "Unsupported input budget")
    name = f"{VERSION}-input{budget}"
    require(name not in smoke.CONFIGURATIONS, "Nested calibration runtime scope")
    fields = ((model, ("visual_token_num", "visual_id")),
              (model.get_model(), ("basis_token_num", "layer_list", "image_token_list", "visual_id",
                                   "image_token_posi", "prompt_len", "image_tokens", "layer_prune")))
    prior = [(obj, key, key in obj.__dict__, obj.__dict__.get(key))
             for obj, names in fields for key in names]
    smoke.CONFIGURATIONS[name] = {"input_tokens": budget, "layers": [], "decoder_tokens": []}
    try:
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            smoke.configure(model, name)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            yield name
    finally:
        smoke.CONFIGURATIONS.pop(name)
        for obj, key, owned, value in reversed(prior):
            if owned:
                setattr(obj, key, value)
            elif key in obj.__dict__:
                delattr(obj, key)


def pixels_fingerprint(pixels):
    import torch
    require(isinstance(pixels, torch.Tensor) and tuple(pixels.shape) == (1, 3, 336, 336)
            and bool(torch.isfinite(pixels).all()), "Expected finite single-image 336x336 pixels")
    return {**tensor_fingerprint(pixels), "device": str(pixels.device)}


def fingerprints(snapshot):
    result = {"encoded": snapshot_encoded(snapshot["encoded"]),
              "devices": [str(t.device) for t in snapshot["encoded"]]}
    for key in ("pre_merge", "post_merge"):
        if key in snapshot:
            result[key] = {**tensor_fingerprint(snapshot[key]), "device": str(snapshot[key].device)}
    return result


def seal(snapshot, observation, condition, pixels, seed):
    snapshot["capture"] = {"version": VERSION, "condition": condition, "seed": seed,
        "pixels": pixels_fingerprint(pixels), "encoding": copy.deepcopy(observation),
        "fingerprints": fingerprints(snapshot)}
    validate_snapshot(snapshot, pixels, condition)
    return observation, snapshot


def validate_snapshot(snapshot, pixels, condition=None):
    capture = snapshot["capture"]
    condition = capture["condition"] if condition is None else condition
    spec = specification(condition)
    require(capture["version"] == VERSION and capture["condition"] == condition,
            "Snapshot version/condition differs")
    require(capture["pixels"] == pixels_fingerprint(pixels), "Snapshot image identity differs")
    current = fingerprints(snapshot)
    require(current == capture["fingerprints"], "Frozen snapshot mutated or changed device")
    require(current["encoded"] == capture["encoding"]["encoded"]
            and current["encoded"]["kept_tokens"] == spec["budget"], "Snapshot output budget differs")
    return current


def capture_condition(model, pixels, seed, condition, image_sha256):
    import torch
    spec = specification(condition)
    # Random retention invokes only the uncompressed native branch. E/FPS is not
    # run then ignored, unlike our older basis-fixed component ablation.
    encode_budget = 576 if spec["method"] == "pure-random" else spec["budget"]
    variant = "no-merge" if spec["method"] == "native" and not spec["merge"] else "native"
    pixel_id = pixels_fingerprint(pixels)
    with runtime_scope(model, encode_budget, seed), torch.inference_mode():
        with native_input_variant(model, variant, capture_tensors=True) as data:
            output = model.encode_images(pixels)
        require(len(data["calls"]) == len(data["snapshots"]) == 1, "Expected one native encoding")
        observation, snapshot = data["calls"][0], data["snapshots"][0]
        require(all(torch.equal(a, b) for a, b in zip(output, snapshot["encoded"])),
                "Capture differs from actual returned native tuple")
        if spec["method"] == "pure-random":
            require(observation["input"]["enabled"] is False
                    and observation["merge_calls"] == 0 and observation["basis_slot_to_source"] == []
                    and "errors" not in observation["input"], "Pure random executed native compression")
            selected, selection = random_selection(image_sha256, spec["random_seed"], spec["budget"])
            features = snapshot["encoded"][0]
            mask = torch.zeros_like(snapshot["encoded"][1])
            mask[:, selected] = True
            snapshot["encoded"] = (features, mask)
            observation = {"source_clip": observation["source_clip"],
                "source_patches": observation["source_patches"],
                "uncompressed_projected": observation["encoded"]["features"],
                "uncompressed_native_observation": observation,
                "random_selection": selection, "encoded": snapshot_encoded(snapshot["encoded"]),
                "controls": {"no_fps_E_basis_or_merge": True,
                             "original_projected_vectors_unchanged": True}}
        observation["experiment"] = {"condition": condition, **spec,
            "effective_basis_tokens": 10 if spec["method"] == "native" else 0,
            "timing_note": "Full576 projection and capture/replay instrumentation; no speed benchmark"}
        require(pixels_fingerprint(pixels) == pixel_id, "Capture mutated pixels")
        return seal(snapshot, observation, condition, pixels, seed)


@contextmanager
def replay_snapshot(model, pixels, snapshot):
    """One exact native multimodal prefill, with independent returned clones."""
    audit = {"status": "running", "image_prefill_calls": 0}
    expected = validate_snapshot(snapshot, pixels)["encoded"]
    returned = []

    def replay(supplied, *args, **kwargs):
        require(pixels_fingerprint(supplied) == snapshot["capture"]["pixels"], "Replay pixels differ")
        audit["image_prefill_calls"] += 1
        require(audit["image_prefill_calls"] == 1, "Replay received more than one image prefill")
        output = tuple(t.clone() for t in snapshot["encoded"])
        require(snapshot_encoded(output) == expected, "Replay clone differs")
        returned.append(output)
        return output

    with replaced(model, "encode_images", replay):
        yield audit
    require(audit["image_prefill_calls"] == 1, "Replay did not consume exactly one image prefill")
    require(all(snapshot_encoded(v) == expected for v in returned), "Consumer mutated replay tuple")
    validate_snapshot(snapshot, pixels)
    audit.update(status="passed", frozen_snapshot_unchanged=True,
                 returned_features_and_mask_exact=True, pixel_and_device_identity_exact=True)


def score_reference(tokenizer, model, case, pixels, snapshot, target):
    import torch
    from rq1_native import reference_tokens, score_embeddings
    prompt, ids, labels, prefix_length = reference_tokens(tokenizer, case, target)
    device = model.get_input_embeddings().weight.device
    ids, labels = ids.to(device), labels.to(device)
    budget = specification(snapshot["capture"]["condition"])["budget"]
    with replay_snapshot(model, pixels, snapshot) as audit, torch.inference_mode():
        _, positions, attention, _, embeddings, expanded_labels = model.prepare_inputs_labels_for_multimodal_x(
            ids, torch.arange(ids.shape[1], device=device).unsqueeze(0),
            torch.ones_like(ids, dtype=torch.bool), None, labels, pixels, image_sizes=None)
    target_ids = labels[labels != -100].tolist()
    start = prefix_length - 1 + budget
    require(embeddings.shape[1] == ids.shape[1] - 1 + budget
            and expanded_labels[0, start:].tolist() == target_ids
            and bool((expanded_labels[0, :start] == -100).all()) and bool(attention.all()),
            "Fixed-reference expansion/target mapping differs")
    require(torch.equal(positions[0], torch.arange(embeddings.shape[1], device=positions.device)),
            "Native compact position policy changed")
    result = score_embeddings(model, {"embeddings": embeddings, "position_ids": positions,
                                     "attention_mask": attention, "labels": expanded_labels})
    return {"target": target, "target_token_ids": target_ids, "prompt": prompt,
        "expanded_sequence_length": embeddings.shape[1], **result,
        "definition": "Fixed-gold mean log probability without EOS; secondary diagnostic"}, audit


def generate_condition(tokenizer, model, processor, image, case, pixels, condition, snapshot, seed, target):
    from rq1_native import observe_generation_tokens
    from smoke_original_apet import generate_native_answer
    spec = specification(condition)
    require(snapshot["capture"]["seed"] == seed, "Snapshot seed differs")
    validate_snapshot(snapshot, pixels, condition)
    generation, tokens = {}, {}
    try:
        with runtime_scope(model, spec["budget"], seed) as name:
            with replay_snapshot(model, pixels, snapshot) as gen_control:
                with observe_generation_tokens(model, tokenizer, tokens):
                    generate_native_answer(tokenizer, model, processor, image, case["generation_question"],
                        name, seed=seed, max_new_tokens=16, require_nonempty=False, record=generation)
            generation["effective_basis_tokens"] = 10 if spec["method"] == "native" else 0
            # Native configure retains its unused basis parameter in the full/
            # replay paths; effective basis use is recorded separately above.
            score, score_control = score_reference(tokenizer, model, case, pixels, snapshot, target)
    finally:
        validate_snapshot(snapshot, pixels, condition)
    return {"case_id": case["case_id"], "condition": condition, **spec,
        "response": generation["response"], "scoring": score_response(case, generation["response"]),
        "generation": generation, "generation_tokens": tokens, "reference_score": score,
        "encoding": copy.deepcopy(snapshot["capture"]["encoding"]),
        "frozen_replay": {"generation": gen_control, "reference_scoring": score_control,
                          "fingerprints": copy.deepcopy(snapshot["capture"]["fingerprints"])}}
