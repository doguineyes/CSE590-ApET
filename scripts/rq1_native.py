"""RQ1 measurement instrumentation. Native ApET source/math remains unchanged.

E: native FP32 reconstruction residual on CLIP's 576 patch features (before MLP).
I: mean teacher-forced reference log p, full context minus one zeroed projected
patch embedding. Singleton zeroing preserves order, masks and position IDs; it is
a sensitivity proxy, not a token-deletion or ApET-compressed accuracy estimate.
"""

from contextlib import contextmanager
import hashlib
import time

from original_diagnostics import observe_native_values, validate_diagnostics
from smoke_original_apet import configure
from rq1_protocol import target_audit_candidates


@contextmanager
def replaced(obj, name, value):
    """Restore the exact preexisting instance attribute/bound method on exceptions."""
    owned = name in obj.__dict__
    previous = obj.__dict__.get(name)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if owned:
            setattr(obj, name, previous)
        else:
            delattr(obj, name)


def tensor_sha256(tensor):
    return hashlib.sha256(tensor.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


def synchronize():
    import torch
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def score_from_logits(logits, labels):
    """Causal shift, target suffix only; works when lm_head and labels differ in device."""
    import torch
    positions = (labels[0] != -100).nonzero().flatten()
    if positions.numel() == 0 or int(positions.min()) == 0:
        raise ValueError("Reference target is empty or lacks a preceding logit")
    ids = labels[0, positions].to(logits.device)
    values = logits[0, (positions - 1).to(logits.device)].float().log_softmax(-1)
    values = values.gather(1, ids[:, None]).squeeze(1)
    if not torch.isfinite(values).all():
        raise RuntimeError("Non-finite reference answer log probability")
    result = values.detach().cpu().tolist()
    return {"token_logprobs": result, "sum_logprob": sum(result),
            "mean_logprob": sum(result) / len(result)}


def score_embeddings(model, prepared, embeddings=None):
    import torch
    with torch.inference_mode():
        output = model(
            inputs_embeds=prepared["embeddings"] if embeddings is None else embeddings,
            position_ids=prepared["position_ids"], attention_mask=prepared["attention_mask"],
            use_cache=False, return_dict=True,
        )
        return score_from_logits(output.logits, prepared["labels"])


def zero_patch(embeddings, image_start, token_index=None):
    """None is a clone-only no-op control; zero exactly one complete feature vector."""
    if token_index is not None and not 0 <= token_index < 576:
        raise ValueError("Visual token index outside 0..575")
    result = embeddings.clone()
    if token_index is not None:
        result[:, image_start + token_index, :] = 0
    return result


def max_score_delta(first, second):
    if len(first["token_logprobs"]) != len(second["token_logprobs"]):
        raise ValueError("Reference token counts differ")
    return max(abs(x-y) for x, y in zip(first["token_logprobs"], second["token_logprobs"]))


@contextmanager
def observe_generation_tokens(model, tokenizer, record):
    """Record existing native generation outputs without changing generation."""
    original = model.generate

    def observed(*args, **kwargs):
        output = original(*args, **kwargs)
        count = len(output.scores)
        if not count:
            raise RuntimeError("Native generation returned no scored tokens")
        # GenerationMixin may prefix a dummy BOS when using inputs_embeds.
        ids = output.sequences[0, -count:].detach().cpu().tolist()
        logprobs = [float(scores[0].float().log_softmax(-1)[token].item())
                    for scores, token in zip(output.scores, ids)]
        values, indices = output.scores[0][0].float().log_softmax(-1).topk(10)
        record.update({"generated_token_ids": ids,
                       "generated_token_strings": tokenizer.convert_ids_to_tokens(ids),
                       "generated_token_logprobs": logprobs,
                       "first_step_top10": [{"token_id": int(index),
                           "token_string": tokenizer.convert_ids_to_tokens(int(index)),
                           "logprob": float(value)}
                           for index, value in zip(indices.detach().cpu().tolist(), values.detach().cpu().tolist())],
                       "score_source": "Existing native generate output.scores (processed generation scores)"})
        return output

    with replaced(model, "generate", observed):
        yield


def audit_target_forms(tokenizer, model, case, encoded, generation, generation_tokens):
    """Compare literal gold casing/tokenization with the actual generated prefix.

    No perturbations, no automatic target replacement, no manifest edits. This
    audit checks semantic/format alignment that no-op/numeric controls cannot.
    """
    import math
    import torch

    variants = []
    generated_ids = generation_tokens["generated_token_ids"]
    for candidate in target_audit_candidates(case, generation["response"]):
        diagnostic_case = {**case, "canonical_target": candidate["text"]}
        prepared = prepare_question(tokenizer, model, diagnostic_case, encoded)
        with torch.inference_mode():
            output = model(inputs_embeds=prepared["embeddings"],
                position_ids=prepared["position_ids"], attention_mask=prepared["attention_mask"],
                use_cache=False, return_dict=True)
            score = score_from_logits(output.logits, prepared["labels"])
            next_logprobs = output.logits[0, prepared["target_start"]-1].float().log_softmax(-1)
            first_generated_logprob = float(next_logprobs[generated_ids[0]].item())
            top_value, top_id = next_logprobs.max(0)
            first_top = {"token_id": int(top_id.item()), "logprob": float(top_value.item()),
                         "token_string": tokenizer.convert_ids_to_tokens(int(top_id.item()))}
        del output, next_logprobs
        repeated = score_embeddings(model, prepared)
        ids = prepared["target_token_ids"]
        variants.append({**candidate, "target_token_ids": ids,
            "target_token_strings": tokenizer.convert_ids_to_tokens(ids), "score": score,
            "reference_sequence_probability_no_eos": math.exp(score["sum_logprob"]),
            "reference_repeat_max_delta": max_score_delta(score, repeated),
            "first_target_matches_first_generated_token": ids[0] == generated_ids[0],
            "target_ids_match_generated_prefix": ids == generated_ids[:len(ids)],
            "teacher_forced_first_step_top1": first_top,
            "first_generated_token_logprob_in_teacher_forcing": first_generated_logprob,
            "first_generated_token_logprob_delta_vs_generate": abs(first_generated_logprob -
                generation_tokens["generated_token_logprobs"][0]),
        })
    return {"case_id": case["case_id"], "question": case["question"],
            "canonical_target": case["canonical_target"], "target_rule": case["target_rule"],
            "image_id": case["image_id"], "seed": encoded["seed"],
            "clip_sha256": encoded["clip_sha256"], "projected_sha256": encoded["projected_sha256"],
            "generation": generation, "generation_tokens": generation_tokens, "variants": variants,
            "note": "Target-format diagnosis only; generated text is not adopted as a reference. "
                    "Probability uses exact token sequences, excludes EOS, and is not semantic answer probability."}


def encode_image(model, processor, image, seed):
    """One full image encoding, then native error calculation on identical CLIP values.

    The baseline branch does not compute E. A temporary vision-output replay feeds
    exactly its CLIP values into native input-only encode_images. FPS, solve,
    selection and merge all execute unchanged; the merged output is not used as
    the sensitivity reference. This replay avoids another expensive CLIP call.
    """
    import torch
    from llava.mm_utils import process_images

    configure(model, "baseline")
    device = model.get_input_embeddings().weight.device
    pixels, _ = process_images([image.convert("RGB")], processor, model.config)
    if not isinstance(pixels, torch.Tensor) or tuple(pixels.shape) != (1, 3, 336, 336):
        raise ValueError("Pilot expects a single 336x336 LLaVA-1.5 image")
    pixels = pixels.to(device=device, dtype=torch.float16)
    tower = model.get_vision_tower()
    capture = {}

    def observe(module, args, output):
        capture["clip"] = output[0].detach().clone()  # discard large attention tensors

    hook = tower.register_forward_hook(observe)
    try:
        with torch.inference_mode():
            projected, mask = model.encode_images(pixels)
    finally:
        hook.remove()
    clip = capture["clip"]
    if tuple(clip.shape) != (1, 577, 1024) or projected.shape[1] != 576 or not mask.all():
        raise ValueError("Unexpected native CLIP/projector geometry")
    replay_calls = []

    def replay(*args, **kwargs):
        replay_calls.append(1)
        return clip.clone(), None

    try:
        settings = configure(model, "input-only")
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        with replaced(tower, "forward", replay), observe_native_values() as data, torch.inference_mode():
            model.encode_images(pixels)
        validate_diagnostics(settings, data)
        if len(replay_calls) != 1:
            raise RuntimeError("Native error pass did not consume exactly one cached CLIP output")
    finally:
        configure(model, "baseline")
    # Save the actual preprocessed canvas for correctly aligned 24x24 heatmaps.
    mean = torch.tensor(processor.image_mean).view(3, 1, 1)
    std = torch.tensor(processor.image_std).view(3, 1, 1)
    canvas = (pixels[0].float().cpu() * std + mean).clamp(0, 1)
    return {"pixels": pixels, "clip": clip, "projected": projected.detach(),
            "mask": mask, "diagnostics": data, "seed": seed,
            "clip_sha256": tensor_sha256(clip), "projected_sha256": tensor_sha256(projected),
            "canvas": (canvas.permute(1, 2, 0) * 255).round().byte().numpy()}


def reference_tokens(tokenizer, case):
    import torch
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    conversation = conv_templates["vicuna_v1"].copy()
    conversation.append_message(conversation.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + case["generation_question"])
    conversation.append_message(conversation.roles[1], None)
    prompt = conversation.get_prompt()
    if case.get("archived_prompt") and prompt != case["archived_prompt"]:
        raise ValueError("Prompt differs from archived baseline")
    prefix = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
    full = tokenizer_image_token(prompt + " " + case["canonical_target"], tokenizer,
                                 IMAGE_TOKEN_INDEX, return_tensors="pt")
    if full.numel() <= prefix.numel() or not torch.equal(full[:len(prefix)], prefix):
        raise ValueError("Answer tokenization changes prompt prefix; inspect boundary before measurement")
    if int((full == IMAGE_TOKEN_INDEX).sum()) != 1:
        raise ValueError("Expected one image placeholder")
    labels = torch.full_like(full, -100)
    labels[len(prefix):] = full[len(prefix):]
    return prompt, full.unsqueeze(0), labels.unsqueeze(0), len(prefix)


def prepare_question(tokenizer, model, case, encoded):
    import torch
    from llava.constants import IMAGE_TOKEN_INDEX

    configure(model, "baseline")
    prompt, ids, labels, prefix_length = reference_tokens(tokenizer, case)
    length = ids.shape[1] - 1 + 576
    limit = min(getattr(model.config, "tokenizer_model_max_length", None) or 2048,
                model.config.max_position_embeddings)
    if length > limit:
        raise ValueError("Reference answer would exceed context limit; refusing truncation")
    device = model.get_input_embeddings().weight.device
    ids, labels = ids.to(device), labels.to(device)
    pids = torch.arange(ids.shape[1], device=device).unsqueeze(0)
    mask = torch.ones_like(ids, dtype=torch.bool)
    image_start = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero().item())
    encode = lambda *args, **kwargs: (encoded["projected"].clone(), encoded["mask"].clone())
    with replaced(model, "encode_images", encode), torch.inference_mode():
        _, positions, attention, _, embeddings, expanded_labels = model.prepare_inputs_labels_for_multimodal_x(
            ids, pids, mask, None, labels, encoded["pixels"], image_sizes=None)
    target_ids = labels[labels != -100].tolist()
    target_start = prefix_length - 1 + 576
    if (embeddings.shape[1] != length or expanded_labels.shape != (1, length)
            or expanded_labels[0, target_start:].tolist() != target_ids
            or (expanded_labels[0, :target_start] != -100).any()
            or not attention.all()
            or not torch.equal(positions[0], torch.arange(length, device=positions.device))):
        raise RuntimeError("Native multimodal target/position mapping failed")
    if not torch.equal(embeddings[0, image_start:image_start+576], encoded["projected"][0].to(embeddings.device)):
        raise RuntimeError("Mapped visual block differs from full-context projected patches")
    return {"embeddings": embeddings, "position_ids": positions, "attention_mask": attention,
            "labels": expanded_labels, "input_ids": ids, "original_positions": pids,
            "original_attention": mask, "image_start": image_start, "prompt": prompt,
            "target_token_ids": target_ids, "target_start": target_start}


def native_parity(model, prepared, encoded):
    """Full native multimodal forward vs cached-embedding forward, same CLIP values."""
    import torch
    replay = lambda *args, **kwargs: (encoded["clip"].clone(), None)
    with replaced(model.get_vision_tower(), "forward", replay), torch.inference_mode():
        output = model(input_ids=prepared["input_ids"], position_ids=prepared["original_positions"],
                       attention_mask=prepared["original_attention"], images=encoded["pixels"],
                       use_cache=False, return_dict=True)
        return score_from_logits(output.logits, prepared["labels"])


def measure_question(tokenizer, model, case, encoded, order, tolerance, progress):
    """Yield progress after controls/each token so interruptions retain expensive work."""
    import torch
    started = time.perf_counter()
    prepared = prepare_question(tokenizer, model, case, encoded)
    record = {"case_id": case["case_id"], "question_id": case["question_id"],
              "image_id": case["image_id"], "status": "running", "question": case["question"],
              "prompt": prepared["prompt"], "canonical_target": case["canonical_target"],
              "target_rule": case["target_rule"], "target_token_ids": prepared["target_token_ids"],
              "target_expanded_start": prepared["target_start"], "image_expanded_start": prepared["image_start"],
              "expanded_sequence_length": prepared["embeddings"].shape[1],
              "archived_predictions": case["predictions"], "task_metadata": case["task_metadata"],
              "outcomes": case["outcomes"], "seed": encoded["seed"],
              "clip_sha256": encoded["clip_sha256"], "projected_sha256": encoded["projected_sha256"],
              "token_execution_order": order, "tokens": [], "controls": {},
              "score_definition": "Mean reference answer token log probability, no EOS; I = S_full - S_zero"}
    # Stable reference repeats and a clone-only no-op precede expensive interventions.
    reference = score_embeddings(model, prepared)
    controls = record["controls"]
    controls["reference"] = reference
    controls["reference_repeat"] = score_embeddings(model, prepared)
    no_op = zero_patch(prepared["embeddings"], prepared["image_start"])
    if not torch.equal(no_op, prepared["embeddings"]):
        raise RuntimeError("No-op changed input embeddings")
    controls["noop"] = score_embeddings(model, prepared, no_op)
    del no_op
    controls["native_multimodal"] = native_parity(model, prepared, encoded)
    controls["max_pre_token_logprob_delta"] = max(max_score_delta(reference, controls[key])
        for key in ["reference_repeat", "noop", "native_multimodal"])
    controls["tolerance_per_answer_token"] = tolerance
    controls["passed"] = controls["max_pre_token_logprob_delta"] <= tolerance
    progress(record)
    if not controls["passed"]:
        raise RuntimeError("Reference/no-op/native-parity control exceeded tolerance; token loop not run")
    for token_index in order:
        synchronize()
        tick = time.perf_counter()
        changed = zero_patch(prepared["embeddings"], prepared["image_start"], token_index)
        score = score_embeddings(model, prepared, changed)
        del changed
        synchronize()
        record["tokens"].append({
            "token_index": token_index, "patch_row": token_index // 24, "patch_col": token_index % 24,
            "reconstruction_error": encoded["diagnostics"]["input"]["errors"][token_index],
            "intervention_score": reference["mean_logprob"] - score["mean_logprob"],
            "perturbed_score": score, "seconds": time.perf_counter() - tick,
        })
        progress(record)
    controls["reference_after"] = score_embeddings(model, prepared)
    controls["post_token_logprob_delta"] = max_score_delta(reference, controls["reference_after"])
    controls["passed"] = controls["passed"] and controls["post_token_logprob_delta"] <= tolerance
    record["seconds"] = time.perf_counter() - started
    record["status"] = "passed" if controls["passed"] else "unstable_reference"
    progress(record)
    return record
