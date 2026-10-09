"""Shared HF LLaVA loading and single-image generation for research runners."""

import time


GIB = 1024 ** 3


def load_llava(model_path, report):
    import torch
    from transformers import AutoConfig, AutoProcessor, LlavaForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("Enable a GPU accelerator for the pretrained model test")
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    if config.model_type != "llava":
        raise ValueError("Expected an HF LLaVA checkpoint")
    max_memory = {i: max(0, torch.cuda.mem_get_info(i)[0] - 2 * GIB)
                  for i in range(torch.cuda.device_count())}
    max_memory["cpu"] = 0
    model = LlavaForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float16,
        device_map="auto", max_memory=max_memory, attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    report["device_map"] = {name: str(device) for name, device in model.hf_device_map.items()}
    if any(str(device) in {"cpu", "disk"} for device in model.hf_device_map.values()):
        raise RuntimeError("Weights did not fit on the GPUs; inspect free VRAM/device_map")
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    processor.patch_size = config.vision_config.patch_size
    processor.vision_feature_select_strategy = config.vision_feature_select_strategy
    processor.num_additional_image_tokens = 1  # CLIP has a CLS token.
    processor.tokenizer.padding_side = "left"
    return model, processor


def generate_answer(model, processor, image, question, max_new_tokens, adapter=None, compression_seed=590):
    import torch

    prompt = f"USER: <image>\n{question} ASSISTANT:"
    inputs = processor(text=prompt, images=image.convert("RGB"), return_tensors="pt")
    image_tokens = (inputs.input_ids == model.config.image_token_index).sum().item()
    if image_tokens != model.config.image_seq_length:
        raise RuntimeError(f"Image token mismatch: {image_tokens} vs {model.config.image_seq_length}")
    inputs = inputs.to(model.get_input_embeddings().weight.device, torch.float16)
    original_image_tokens = image_tokens
    if adapter is not None:
        inputs = adapter.prepare_inputs(inputs, compression_seed)
        image_tokens = (inputs["input_ids"] == model.config.image_token_index).sum().item()
    input_length = inputs["input_ids"].shape[1]
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
        torch.cuda.synchronize(i)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                   do_sample=False, use_cache=True)
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
    elapsed = time.perf_counter() - started
    response = processor.decode(generated[0, input_length:],
                                skip_special_tokens=True).strip()
    if not response:
        raise RuntimeError("Generation returned an empty response")
    result = {
        "question": question, "response": response, "image_tokens": image_tokens,
        "original_image_tokens": original_image_tokens, "prompt_tokens": input_length,
        "generated_tokens": generated.shape[1] - input_length,
        "generation_seconds": round(elapsed, 3),
        "peak_allocated_gib": {str(i): round(torch.cuda.max_memory_allocated(i) / GIB, 3)
                               for i in range(torch.cuda.device_count())},
    }
    if adapter is not None:
        result["compression"] = adapter.statistics()
    return result
