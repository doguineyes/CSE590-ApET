"""Preflight and stage-observation checks requiring no Torch or downloads."""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from smoke_original_apet import checkpoint_plan, select_files, validate_native_config, validate_trace


class OriginalSmokeTests(unittest.TestCase):
    def test_rejects_converted_hf_checkpoint_before_loading(self):
        with self.assertRaisesRegex(ValueError, "different format"):
            validate_native_config({"architectures": ["LlavaForConditionalGeneration"],
                                    "text_config": {}, "model_type": "llava"})

    def test_local_missing_shard_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps({
                "architectures": ["LlavaLlamaForCausalLM"], "mm_vision_tower": "openai/clip",
            }))
            (root / "pytorch_model.bin.index.json").write_text(json.dumps({
                "weight_map": {"model.embed_tokens.weight": "pytorch_model-00001.bin"},
            }))
            with self.assertRaisesRegex(ValueError, "Missing checkpoint shard"):
                checkpoint_plan(directory, "unused", root, native=True)

    def test_selects_one_format_without_extra_model_downloads(self):
        names = ["config.json", "preprocessor_config.json", "model.safetensors", "pytorch_model.bin",
                 "tf_model.h5", "flax_model.msgpack", "mm_projector.bin", "tokenizer.model", "subdir/other.json"]
        selected = {file.rfilename for file in select_files(
            [SimpleNamespace(rfilename=name) for name in names])}
        self.assertIn("model.safetensors", selected)
        self.assertIn("preprocessor_config.json", selected)
        self.assertNotIn("pytorch_model.bin", selected)
        self.assertNotIn("mm_projector.bin", selected)
        self.assertNotIn("tf_model.h5", selected)
        self.assertNotIn("subdir/other.json", selected)

    def test_missing_decoder_pruning_cannot_pass_combined_smoke(self):
        with self.assertRaisesRegex(RuntimeError, "did not occur"):
            validate_trace("combined", {"input": {"before": 576, "after": 288}})

    def test_wrong_visual_budget_cannot_pass(self):
        with self.assertRaisesRegex(RuntimeError, "Unexpected input"):
            validate_trace("input-only", {"input": {"before": 576, "after": 576}})

    def test_observed_combined_reduction_and_baseline(self):
        validate_trace("baseline", {"input": {"before": 576, "after": 576}})
        validate_trace("combined", {"input": {"before": 576, "after": 288}, "decoder": [
            {"layer": 16, "sequence_before": 350, "sequence_after": 158,
             "visual_before": 288, "visual_after": 96},
        ]})


if __name__ == "__main__":
    unittest.main()
