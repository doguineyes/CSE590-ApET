"""Scoped projector hooks for HF LLaVA 4.48.2, one image and batch size one.

The core runs before projection. Projection still uses the full spatial grid,
then its output is selected in spatial order, as in the original LLaVA code.
Image placeholders are shortened before generation, so normal HF cache/position
handling sees the actual compressed sequence length. No decoder patch is used.
"""

import importlib.metadata

from ..core import compress_tokens
from ..legacy_reference import source_hashes, verify_result


class HFInputCompressionAdapter:
    def __init__(self, model, config, verify_legacy=True):
        if importlib.metadata.version("transformers") != "4.48.2":
            raise ValueError("This first HF adapter requires Transformers 4.48.2")
        if model.config.model_type != "llava" or model.config.vision_feature_select_strategy != "default":
            raise ValueError("Adapter supports HF LLaVA with default CLIP patch selection")
        if config.keep_tokens > model.config.image_seq_length:
            raise ValueError("keep_tokens exceeds the model's visual token count")
        self.model, self.config = model, config
        self.verify_legacy = verify_legacy
        self.legacy_check = None
        self.handles = []
        self.result = None
        self.calls = 0
        self.seed = None

    def __enter__(self):
        if self.handles:
            raise RuntimeError("Adapter is already active")
        self.handles = [
            self.model.multi_modal_projector.register_forward_pre_hook(self._before_projection),
            self.model.multi_modal_projector.register_forward_hook(self._after_projection),
        ]
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.result = None

    def prepare_inputs(self, inputs, seed):
        import torch

        if not self.handles:
            raise RuntimeError("Use the adapter inside a with block")
        ids = inputs["input_ids"]
        if ids.shape[0] != 1 or inputs["pixel_values"].shape[0] != 1:
            raise ValueError("First adapter supports batch size one and exactly one image")
        positions = torch.where(ids[0] == self.model.config.image_token_index)[0]
        if positions.numel() != self.model.config.image_seq_length:
            raise ValueError("Expected one full expanded image placeholder block")
        if not torch.all(positions[1:] - positions[:-1] == 1).item():
            raise ValueError("Image placeholders must form a contiguous block")
        self.result, self.calls, self.seed = None, 0, seed
        if self.config.keep_tokens == positions.numel():
            return inputs
        keep = torch.ones(ids.shape[1], dtype=torch.bool, device=ids.device)
        keep[positions[self.config.keep_tokens:]] = False
        shortened = dict(inputs)
        shortened["input_ids"] = ids[:, keep]
        shortened["attention_mask"] = inputs["attention_mask"][:, keep]
        return shortened

    def _before_projection(self, module, args):
        import torch

        if self.seed is None or self.calls:
            raise RuntimeError("Unexpected repeated image projection; cached decoding must reuse image features")
        features = args[0]
        if features.shape[:2] != (1, self.model.config.image_seq_length):
            raise ValueError(f"Unexpected vision feature shape: {tuple(features.shape)}")
        generator = torch.Generator(device=features.device).manual_seed(self.seed)
        self.result = compress_tokens(features, self.config, generator)
        self.calls += 1
        if self.verify_legacy and self.legacy_check is None:
            self.legacy_check = {**verify_result(features, self.config, self.result),
                                 "source_hashes": source_hashes()}
        # Real checkpoint is FP16. The cast also permits tiny FP32 CPU models
        # in adapter tests, while the core preserves legacy FP16 arithmetic.
        return (self.result.full_features.to(dtype=features.dtype), *args[1:])

    def _after_projection(self, module, args, output):
        if self.result is None:
            raise RuntimeError("Projector ran without compression state")
        if self.config.keep_tokens == output.shape[1]:
            return output
        indices = self.result.kept_indices.to(output.device)
        return output.gather(1, indices.unsqueeze(-1).expand(-1, -1, output.shape[-1]))

    def statistics(self):
        if self.calls != 1 or self.result is None:
            raise RuntimeError("Expected exactly one compressed image projection per generation")
        result = self.result
        return {
            "stage": "vision input only", "keep_tokens": self.config.keep_tokens,
            "basis_tokens": self.config.basis_tokens, "seed": self.seed,
            "kept_indices": result.kept_indices[0].tolist(),
            "basis_indices": result.basis_indices[0].tolist() if result.basis_indices is not None else [],
            "basis_slots": result.basis_slots[0].tolist() if result.basis_slots is not None else [],
            "mean_reconstruction_error": result.errors.mean().item() if result.errors is not None else None,
            "legacy_check": self.legacy_check,
        }
