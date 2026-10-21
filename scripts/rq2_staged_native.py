"""Scoped staged evaluation. Native encode/prune math and forward/cache stay intact.

Pure random changes only the input mask and the layer-prune gather. It does not
run FPS, E, basis replacement or merging. Native arms call the original methods.
Captured RNG continuation is replayed too, so skipping an already captured input
FPS call does not change the RNG seen by native decoder FPS.
"""
from contextlib import contextmanager, ExitStack
import copy

from rq1_native import replaced
from rq2_native import native_input_variant, snapshot_encoded, tensor_fingerprint
from rq2_protocol import require, score_response
from rq2_random_native import pixels_fingerprint, fingerprints
from rq2_staged_protocol import VERSION, LAYER, specification, random_plan


def rng_state():
    import torch
    return {"cpu": torch.random.get_rng_state().clone(),
            "cuda": [s.clone() for s in torch.cuda.get_rng_state_all()]}


def set_rng(state):
    import torch
    torch.random.set_rng_state(state["cpu"])
    torch.cuda.set_rng_state_all(state["cuda"])


def tree_fingerprint(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {**tensor_fingerprint(value), "device": str(value.device)}
    if isinstance(value, dict):
        return {k: tree_fingerprint(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tree_fingerprint(v) for v in value]
    require(value is None or isinstance(value, (str, int, bool)), "Unexpected snapshot metadata")
    return value


def clone_tree(value, *, cpu=False):
    import torch
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu() if cpu else value.detach()
        return value.clone()
    if isinstance(value, dict):
        return {k: clone_tree(v, cpu=cpu) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(clone_tree(v, cpu=cpu) for v in value)
    if isinstance(value, list):
        return [clone_tree(v, cpu=cpu) for v in value]
    return copy.deepcopy(value)


@contextmanager
def runtime_scope(model, condition, seed, *, capture_input=False):
    import torch
    import smoke_original_apet as smoke
    spec = specification(condition)
    require(not model.training, "Staged execution requires eval mode")
    require(spec["layer"] is None or len(model.model.layers) > LAYER, "Model must include blocks16 and17")
    name = VERSION + "-" + condition + ("-encode" if capture_input else "")
    require(name not in smoke.CONFIGURATIONS, "Nested staged runtime configuration")
    budget = 576 if capture_input and spec["method"] == "pure-random" else spec["input_tokens"]
    layers = [] if capture_input or spec["layer"] is None else [LAYER]
    smoke.CONFIGURATIONS[name] = {"input_tokens": budget, "layers": layers,
                                  "decoder_tokens": [spec["decoder_tokens"]] if layers else []}
    fields = ((model, ("visual_token_num", "visual_id", "encode_images")), (model.model, (
        "basis_token_num", "layer_list", "image_token_list", "visual_id", "image_token_posi",
        "prompt_len", "image_tokens", "layer_prune")))
    prior = [(obj, key, key in obj.__dict__, obj.__dict__.get(key)) for obj, names in fields for key in names]
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


def validate_input(snapshot, pixels, condition=None):
    capture = snapshot["capture"]
    condition = capture["condition"] if condition is None else condition
    require(capture["version"] == VERSION and capture["condition"] == condition
            and capture["pixels"] == pixels_fingerprint(pixels), "Input snapshot condition/image changed")
    require(capture["fingerprints"] == fingerprints(snapshot)
            and capture["rng_after_input"] == tree_fingerprint(snapshot["rng_after_input"]),
            "Frozen input/RNG snapshot mutated")
    require(snapshot_encoded(snapshot["encoded"])["kept_tokens"] == specification(condition)["input_tokens"],
            "Input snapshot budget differs")
    return capture


def capture_input(model, pixels, seed, condition, image_sha256):
    import torch
    spec = specification(condition)
    with runtime_scope(model, condition, seed, capture_input=True), torch.inference_mode():
        with native_input_variant(model, "native", capture_tensors=True) as data:
            output = model.encode_images(pixels)
        require(len(data["calls"]) == len(data["snapshots"]) == 1, "Expected one input capture")
        observed, snapshot = data["calls"][0], data["snapshots"][0]
        require(all(torch.equal(a, b) for a, b in zip(output, snapshot["encoded"])), "Input observer altered native tuple")
        snapshot["rng_after_input"] = rng_state()
        if spec["method"] == "pure-random":
            require(not observed["input"]["enabled"] and observed["merge_calls"] == 0,
                    "Pure random input executed native compression")
            plan = random_plan(image_sha256, spec["random_seed"], spec["input_tokens"], spec["decoder_tokens"])
            mask = torch.zeros_like(snapshot["encoded"][1])
            mask[:, plan["input_original_positions"]] = True
            snapshot["encoded"] = (snapshot["encoded"][0], mask)
            observed = {"source_clip": observed["source_clip"], "source_patches": observed["source_patches"],
                "full_projected": observed["encoded"]["features"], "random_plan": plan,
                "encoded": snapshot_encoded(snapshot["encoded"]),
                "pure_input": {"fps": False, "E": False, "basis_replacement": False, "merging": False}}
    snapshot["capture"] = {"version": VERSION, "condition": condition, "seed": seed,
        "pixels": pixels_fingerprint(pixels), "observation": copy.deepcopy(observed),
        "fingerprints": fingerprints(snapshot), "rng_after_input": tree_fingerprint(snapshot["rng_after_input"])}
    validate_input(snapshot, pixels, condition)
    return observed, snapshot


@contextmanager
def replay_input(model, pixels, snapshot):
    expected = validate_input(snapshot, pixels)
    audit = {"calls": 0}
    returned = []
    def encode(supplied, *args, **kwargs):
        require(pixels_fingerprint(supplied) == expected["pixels"], "Replayed pixels differ")
        audit["calls"] += 1
        require(audit["calls"] == 1, "Multiple image prefills")
        output = clone_tree(snapshot["encoded"])
        require(snapshot_encoded(output) == expected["fingerprints"]["encoded"], "Input clone changed")
        # Continue native FPS RNG exactly where original input encoding left it.
        set_rng(snapshot["rng_after_input"])
        returned.append(output)
        return output
    with replaced(model, "encode_images", encode):
        yield audit
    require(audit["calls"] == 1 and all(snapshot_encoded(v) == expected["fingerprints"]["encoded"] for v in returned),
            "Input replay missing or consumer mutated tuple")
    validate_input(snapshot, pixels)
    audit.update(status="passed", features_mask_and_rng_continuation_exact=True)


def random_prune(inner, supplied, selected):
    """Single-image, unpadded gather with the native compact position policy."""
    import torch
    features = supplied["features"]
    positions, attention, labels = (supplied[k] for k in ("position_ids", "attention_mask", "labels"))
    start = inner.image_token_posi[0]
    count = min(inner.image_token_list[0], inner.image_tokens[0])
    require(features.ndim == 3 and features.shape[0] == 1 and start >= 0
            and start + count <= features.shape[1], "Unsupported visual span")
    require(getattr(inner.config, "tokenizer_padding_side", "right") == "right"
            and (attention is None or attention.ndim == 2 and bool(attention.all())),
            "Staged calibration requires native unpadded single-image generation")
    require(selected == sorted(set(selected)) and all(0 <= i < count for i in selected), "Invalid decoder gather")
    indices = list(range(start)) + [start + i for i in selected] + list(range(start + count, features.shape[1]))
    def gather(value):
        return None if value is None else value.index_select(1, torch.tensor(indices, device=value.device))
    output = gather(features)
    new_positions = None if positions is None else torch.arange(
        len(indices), dtype=positions.dtype, device=positions.device).unsqueeze(0)
    return new_positions, gather(attention), output, gather(labels)


def prune_geometry(inner, supplied, output, spec):
    import torch
    features, result = supplied["features"], output[2]
    start, first, last = inner.image_token_posi[0], spec["input_tokens"], spec["decoder_tokens"]
    require(result.shape == (1, features.shape[1] - first + last, features.shape[2])
            and bool(torch.isfinite(result).all()), "Decoder output shape/values differ")
    require(torch.equal(result[:, :start], features[:, :start])
            and torch.equal(result[:, start + last:], features[:, start + first:]),
            "Decoder compression changed nonvisual states")
    if output[0] is not None:
        require(torch.equal(output[0][0], torch.arange(result.shape[1], device=output[0].device)),
                "Decoder compact positions differ")
    for before, after in ((supplied["labels"], output[3]), (supplied["attention_mask"], output[1])):
        require((before is None) == (after is None), "Native optional mask/label contract differs")
        if before is not None:
            require(after.shape == result.shape[:2] and torch.equal(after[:, :start], before[:, :start])
                    and torch.equal(after[:, start + last:], before[:, start + first:]), "Nonvisual mask/labels changed")
    return {"status": "passed", "image_start": start, "visual_before": first, "visual_after": last,
            "sequence_before": features.shape[1], "sequence_after": result.shape[1],
            "nonvisual_states_unchanged": True, "native_compact_position_policy": True}


@contextmanager
def decoder_scope(model, condition, image_sha256, *, replay=None):
    """Observe original prune unchanged, or select random states at that boundary."""
    import torch
    from llava.model import modeling_llama_x as module
    from original_diagnostics import trace_returns, capture_scores, capture_merge
    spec, inner = specification(condition), model.model
    native_prune, native_merge = inner.layer_prune, module.token_merging
    require(getattr(native_prune, "__func__", None) is module.LlamaModel.layer_prune,
            "Native layer_prune method changed")
    data = {"condition": condition, "calls": 0, "snapshot": None, "mode": "replay" if replay else "capture"}
    locals_data = {"scores": [], "merge": []}
    handlers = {module.LlamaModel.layer_prune.__code__: lambda local: locals_data["scores"].append(capture_scores(local, False)),
                native_merge.__code__: lambda local: locals_data["merge"].append(capture_merge(local, False))}
    def merge(*args, **kwargs):
        require(data["snapshot"] is not None, "Merge outside decoder prefill")
        data["snapshot"]["pre_merge"] = args[0].detach().clone()
        output = native_merge(*args, **kwargs)
        data["snapshot"]["post_merge"] = output.detach().clone()
        return output
    def prune(*args, **kwargs):
        require(not args, "Expected native keyword layer-prune call")
        data["calls"] += 1
        require(data["calls"] == 1 and kwargs["cur_num"] == 0 and kwargs["rank_layer"] == LAYER
                and spec["layer"] == LAYER, "Unexpected decoder pruning boundary/count")
        supplied = {k: clone_tree(kwargs[k]) for k in ("features", "position_ids", "attention_mask", "labels")}
        before_rng = tree_fingerprint(rng_state())
        if replay is not None:
            require(replay["condition"] == condition and replay["before"] == tree_fingerprint(supplied)
                    and replay["rng_before"] == before_rng, "Frozen decoder replay source/RNG differs")
            require(replay["after"] == tree_fingerprint(replay["output"])
                    and replay["rng_after"] == tree_fingerprint(replay["rng_after_state"]), "Frozen decoder snapshot mutated")
            output = clone_tree(replay["output"])
            set_rng(replay["rng_after_state"])
            data["snapshot"] = replay
            data["observation"] = {"mode": "frozen_decoder_replay", "source_and_rng_exact": True}
        else:
            snapshot = {"condition": condition, "before": tree_fingerprint(supplied), "rng_before": before_rng,
                        "source_visual": supplied["features"][:, inner.image_token_posi[0]:inner.image_token_posi[0] + spec["input_tokens"]].clone()}
            data["snapshot"] = snapshot
            if spec["method"] == "native":
                output = native_prune(**kwargs)  # Exact original method; no new E or merge implementation.
                require(len(locals_data["scores"]) == len(locals_data["merge"]) == 1, "Native decoder diagnostics missing")
                scores, graph = locals_data["scores"][0], locals_data["merge"][0]
                require(scores["source_tokens"] == spec["input_tokens"] and len(scores["selected_indices"]) == spec["decoder_tokens"]
                        and len(scores["basis_indices"]) == 10 and sum(graph["merge_counts"]) == spec["input_tokens"] - spec["decoder_tokens"],
                        "Native layer16 E/basis/merge counts differ")
                scores["basis_slot_to_source"] = list(zip(graph["basis_assignment_slots"], scores["basis_indices"]))
                data["observation"] = {"method": "unchanged_native_layer_prune", "E": scores, "merge": graph,
                    "pre_merge": tensor_fingerprint(snapshot["pre_merge"]), "post_merge": tensor_fingerprint(snapshot["post_merge"])}
            else:
                plan = random_plan(image_sha256, spec["random_seed"], spec["input_tokens"], spec["decoder_tokens"])
                output = random_prune(inner, kwargs, plan["decoder_compact_positions"])
                require(not locals_data["scores"] and not locals_data["merge"], "Pure random decoder executed E/merge")
                require(torch.equal(output[2][:, inner.image_token_posi[0]:inner.image_token_posi[0] + spec["decoder_tokens"]],
                    supplied["features"][:, inner.image_token_posi[0] + torch.tensor(plan["decoder_compact_positions"], device=supplied["features"].device)]),
                    "Random layer16 vectors differ from original contextual states")
                data["observation"] = {"method": "pure_random_gather", "random_plan": plan,
                    "fps": False, "E": False, "basis_replacement": False, "merging": False,
                    "original_contextual_vectors_exact": True}
            snapshot["output"] = clone_tree(output)
            snapshot["after"] = tree_fingerprint(output)
            snapshot["rng_after_state"] = rng_state()
            snapshot["rng_after"] = tree_fingerprint(snapshot["rng_after_state"])
        geometry = prune_geometry(inner, supplied, output, spec)
        data["observation"].update(geometry=geometry, source=tree_fingerprint(supplied), output=tree_fingerprint(output),
                                   rng_before=before_rng, rng_after=tree_fingerprint(rng_state()))
        return output
    with ExitStack() as stack:
        stack.enter_context(replaced(inner, "layer_prune", prune))
        if spec["method"] == "native" and replay is None:
            stack.enter_context(replaced(module, "token_merging", merge))
        stack.enter_context(trace_returns(handlers))
        yield data
    require(data["calls"] == (0 if spec["layer"] is None else 1), "Missing/unexpected decoder prefill")
    data["status"] = "passed"


@contextmanager
def observe_cache(model):
    """Record first and seventeenth block inputs, including native cache lengths."""
    import torch
    events, handles = [], []
    def hook(index):
        def before(module, args, kwargs):
            hidden = args[0] if args else kwargs["hidden_states"]
            positions = kwargs["position_ids"]
            if isinstance(positions, list):
                positions = torch.stack(positions)
            cache = kwargs.get("past_key_value")
            events.append({"block": index + 1, "sequence_tokens": hidden.shape[1],
                "position_first": int(positions.flatten()[0].item()), "position_last": int(positions.flatten()[-1].item()),
                "cache_tokens_before": int(cache.get_seq_length(index)) if cache is not None else 0})
        return before
    try:
        for index in (0, LAYER):
            handles.append(model.model.layers[index].register_forward_pre_hook(hook(index), with_kwargs=True))
        yield events
    finally:
        for handle in handles:
            handle.remove()


def validate_cache(events, spec, expanded, generated_tokens):
    require(type(generated_tokens) is int and generated_tokens > 0, "Expected scored generation tokens")
    last = spec["decoder_tokens"] if spec["layer"] else spec["input_tokens"]
    expected_lengths = {1: expanded, LAYER + 1: expanded - spec["input_tokens"] + last}
    for block, length in expected_lengths.items():
        rows = [r for r in events if r["block"] == block]
        require(len(rows) == generated_tokens and rows[0] == {"block": block, "sequence_tokens": length,
            "position_first": 0, "position_last": length - 1, "cache_tokens_before": 0},
            f"Prefill/cache differs at block{block}: expected length={length}, forwards={generated_tokens}; observed={rows[:3]!r}")
        for step, row in enumerate(rows[1:]):
            position = length + step
            require(row == {"block": block, "sequence_tokens": 1, "position_first": position,
                "position_last": position, "cache_tokens_before": position},
                f"Cached decode differs at block{block}: expected position/cache={position}; observed={row!r}")
    return {"status": "passed", "events": events, "cached_decode_steps": generated_tokens - 1}


def generate_condition(tokenizer, model, processor, image, case, pixels, condition, input_snapshot, seed,
                       *, decoder_replay=None, probe=False):
    from rq1_native import observe_generation_tokens
    from smoke_original_apet import generate_native_answer
    spec = specification(condition)
    require(input_snapshot["capture"]["seed"] == seed, "Input snapshot seed differs")
    validate_input(input_snapshot, pixels, condition)
    generation, tokens = {}, {}
    with runtime_scope(model, condition, seed) as name, replay_input(model, pixels, input_snapshot) as input_audit:
        with ExitStack() as stack:
            layer = stack.enter_context(decoder_scope(model, condition, case["image_sha256"], replay=decoder_replay))
            cache = stack.enter_context(observe_cache(model))
            stack.enter_context(observe_generation_tokens(model, tokenizer, tokens))
            generate_native_answer(tokenizer, model, processor, image, case["generation_question"], name,
                seed=seed, max_new_tokens=4 if probe else 16, min_new_tokens=4 if probe else 0,
                require_nonempty=False, record=generation)
    cache_check = validate_cache(cache, spec, generation["expanded_prompt_tokens"], generation["generated_tokens"])
    if probe:
        require(generation["generated_tokens"] == 4, "Cached-decoding probe must generate four tokens")
    row = {"case_id": case["case_id"], "image_sha256": case["image_sha256"], "condition": condition,
        **spec, "seed": seed, "response": generation["response"], "scoring": score_response(case, generation["response"]),
        "generation": generation, "generation_tokens": tokens, "input_replay": input_audit,
        "cache": cache_check, "input_encoding": input_snapshot["capture"]["observation"],
        "decoder": layer["observation"] if layer and layer["calls"] else None,
        "control_only": bool(probe or decoder_replay is not None or condition.startswith("pure-random-576"))}
    return row, layer["snapshot"] if layer else None


def generate_raw_native(tokenizer, model, processor, image, case, condition, seed):
    """Fresh full native encoding+decoder, no feature replay or prune replacement.

    This is a separate numerical/answer control, never a replacement primary.
    Existing trace/token/cache observers leave the original return values intact.
    """
    from rq1_native import observe_generation_tokens
    from smoke_original_apet import generate_native_answer
    spec = specification(condition)
    require(spec["method"] == "native", "Raw staged control must be native")
    generation, tokens = {}, {}
    with runtime_scope(model, condition, seed) as name, observe_cache(model) as cache:
        with observe_generation_tokens(model, tokenizer, tokens):
            generate_native_answer(tokenizer, model, processor, image, case["generation_question"], name,
                seed=seed, max_new_tokens=16, require_nonempty=False, record=generation)
    return {"case_id": case["case_id"], "condition": condition, **spec, "seed": seed,
        "response": generation["response"], "scoring": score_response(case, generation["response"]),
        "generation": generation, "generation_tokens": tokens, "control_only": True,
        "cache": validate_cache(cache, spec, generation["expanded_prompt_tokens"], generation["generated_tokens"]),
        "implementation": "Fresh original native encode_images and layer_prune; no feature replay or replacement"}
