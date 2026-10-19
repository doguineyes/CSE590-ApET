"""Tiny native ApET checks; run in the original environment, no checkpoints.

These exercise the actual repository encode_images/token_merging on synthetic
CLIP features. The Intel Mac reading environment intentionally skips Torch.
Pretrained answer parity is a separate gate in the Kaggle pilot runner.
"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))
from rq2_native import (assert_native_parity, assert_selection_equal,
                        native_input_variant, snapshot_encoded, tensor_fingerprint)


class GateValidationTests(unittest.TestCase):
    """The provenance gates themselves run without the model runtime."""

    def test_parity_requires_features_mask_and_token_ids_all_to_match(self):
        import copy
        record = {"features": {"shape": [1, 576, 32], "dtype": "torch.float16", "sha256": "features"},
                  "mask": {"shape": [1, 576], "dtype": "torch.bool", "sha256": "mask"},
                  "selected_indices_sorted": [2, 4, 7], "kept_tokens": 3}
        self.assertEqual(assert_native_parity(record, record, [6, 9], [6, 9])["status"], "passed")
        for field in ("features", "mask"):
            changed = copy.deepcopy(record)
            changed[field]["sha256"] += "changed"
            with self.assertRaises(RuntimeError):
                assert_native_parity(record, changed, [6, 9], [6, 9])
        with self.assertRaisesRegex(RuntimeError, "generated_ids_exact"):
            assert_native_parity(record, record, [6, 9], [6, 10])
        with self.assertRaises(ValueError):
            assert_native_parity(record, record, None, None)

    def test_selection_gate_fails_when_fps_or_basis_assignment_changes(self):
        import copy
        original = {"source_clip": "clip", "source_patches": "patches", "pre_merge": "basis-replaced",
                    "post_merge": "averaged", "basis_slot_to_source": [{"assignment_slot": 4, "source_index": 7}],
                    "input": {"enabled": True, "errors": [0.1, 0.2], "basis_indices": [7],
                              "selected_indices": [2, 4], "selected_indices_sorted": [2, 4]},
                    "encoded": {"mask": "same-mask"}}
        no_merge = copy.deepcopy(original)
        no_merge["post_merge"] = no_merge["pre_merge"]
        self.assertEqual(assert_selection_equal(original, no_merge)["status"], "passed")
        changed = copy.deepcopy(no_merge)
        changed["basis_slot_to_source"][0]["source_index"] = 8
        with self.assertRaisesRegex(RuntimeError, "basis_slot_to_source_exact"):
            assert_selection_equal(original, changed)
        changed = copy.deepcopy(no_merge)
        changed["input"]["basis_indices"] = [8]
        with self.assertRaisesRegex(RuntimeError, "basis_indices_exact"):
            assert_selection_equal(original, changed)


class NativeRQ2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("Run native RQ2 checks in the original Kaggle environment with Torch")
        os.environ["MPLBACKEND"] = "Agg"
        cls.torch = torch

    def fixture(self, keep=288):
        from llava.model.llava_arch import LlavaMetaForCausalLM
        torch = self.torch

        class Vision(torch.nn.Module):
            def __init__(self, features):
                super().__init__()
                self.features = features

            def forward(self, images):
                return self.features.clone(), None

        class Harness:
            encode_images = LlavaMetaForCausalLM.encode_images
            token_merging = LlavaMetaForCausalLM.token_merging

            def __init__(self, features):
                self.keep = keep
                self.tower = Vision(features)
                self.model = SimpleNamespace(basis_token_num=10, layer_list=[],
                    get_vision_tower=lambda: self.tower, mm_projector=lambda values: values)

            def get_model(self):
                return self.model

            def get_vision_tower(self):
                return self.tower

            def get_visual_token_num(self):
                return self.keep

        torch.manual_seed(590)
        source = torch.randn(1, 577, 32).half()
        return Harness(source), source

    def encode(self, model, variant):
        self.torch.manual_seed(590)
        with native_input_variant(model, variant, capture_tensors=True) as observed:
            output = model.encode_images(None)
        return output, observed

    def test_native_wrapper_is_exact_and_records_actual_merge_destinations(self):
        torch = self.torch
        model, _ = self.fixture()
        torch.manual_seed(590)
        expected = model.encode_images(None)
        actual, data = self.encode(model, "native")
        self.assertEqual(data["status"], "passed")
        call = data["calls"][0]
        parity = assert_native_parity(snapshot_encoded(expected), call["encoded"], [4, 8], [4, 8])
        self.assertEqual(parity["status"], "passed")
        self.assertTrue(torch.equal(expected[0], actual[0]))
        self.assertTrue(torch.equal(expected[1], actual[1]))
        self.assertEqual(call["merge_calls"], 1)
        self.assertEqual(len(call["input"]["errors"]), 576)
        merge = call["input_merge"]
        self.assertEqual(sum(merge["merge_counts"]), 288)
        self.assertEqual(len(merge["dropped_indices"]), 288)
        self.assertEqual(len(merge["target_indices"]), 278)
        self.assertEqual(set(merge["assigned_target_indices"]) - set(merge["target_indices"]), set())
        self.assertEqual(set(merge["target_indices"]) & set(merge["basis_slots"]), set())

    def test_no_merge_keeps_exact_fps_selection_and_basis_placement(self):
        torch = self.torch
        model, source = self.fixture()
        native_output, native = self.encode(model, "native")
        changed_output, changed = self.encode(model, "no-merge")
        control = assert_selection_equal(native["calls"][0], changed["calls"][0])
        self.assertEqual(control["status"], "passed")
        self.assertTrue(torch.equal(native_output[1], changed_output[1]))
        self.assertNotIn("input_merge", changed["calls"][0])
        # Independent oracle: raw source patches with only the unchanged native
        # FPS basis-assignment mapping applied. No copy of selection/solve math.
        expected = source[:, 1:].clone()
        call = changed["calls"][0]
        mapping = call["basis_slot_to_source"]
        self.assertEqual(len(mapping), 10)
        self.assertEqual([x["assignment_slot"] for x in mapping],
                         sorted(call["input"]["selected_indices"][-10:]))
        self.assertEqual([x["source_index"] for x in mapping], call["input"]["basis_indices"])
        for entry in mapping:
            expected[0, entry["assignment_slot"]] = source[0, 1 + entry["source_index"]]
            self.assertEqual(call["encoded"]["selected_indices_sorted"][entry["compact_output_position"]],
                             entry["assignment_slot"])
        self.assertTrue(torch.equal(changed_output[0], expected))
        snapshot = changed["snapshots"][0]
        self.assertTrue(torch.equal(snapshot["pre_merge"], snapshot["post_merge"]))
        self.assertTrue(torch.equal(snapshot["encoded"][0], expected))
        # Only native non-basis merge targets may differ from that oracle.
        native_snapshot = native["snapshots"][0]
        targets = set(native["calls"][0]["input_merge"]["target_indices"])
        others = [index for index in range(576) if index not in targets]
        self.assertTrue(torch.equal(native_snapshot["post_merge"][:, others], expected[:, others]))
        self.assertFalse(torch.equal(native_snapshot["post_merge"], expected))
        self.assertTrue(torch.equal(source, model.tower.features))

    def test_all_instance_patches_and_tracing_restore_on_failure(self):
        import sys
        model, _ = self.fixture()
        before_model = set(model.__dict__)
        before_tower = set(model.tower.__dict__)
        before_trace = sys.gettrace()
        with self.assertRaisesRegex(RuntimeError, "intentional failure"):
            with native_input_variant(model, "no-merge") as data:
                model.encode_images(None)
                raise RuntimeError("intentional failure")
        self.assertEqual(data["status"], "failed")
        self.assertEqual(set(model.__dict__), before_model)
        self.assertEqual(set(model.tower.__dict__), before_tower)
        self.assertIs(sys.gettrace(), before_trace)
        original, _ = self.encode(model, "native")
        repeated, _ = self.encode(model, "native")
        self.assertEqual(snapshot_encoded(original), snapshot_encoded(repeated))

    def test_disabled_decoder_and_native_method_identity_are_enforced(self):
        model, _ = self.fixture()
        model.model.layer_list = [16]
        with self.assertRaisesRegex(ValueError, "decoder compression disabled"):
            with native_input_variant(model):
                pass
        model.model.layer_list = []
        model.token_merging = lambda *args, **kwargs: args[0]
        with self.assertRaisesRegex(ValueError, "unchanged pinned native token_merging"):
            with native_input_variant(model):
                pass

    def test_parity_gate_rejects_unchanged_text_with_changed_ids_or_features(self):
        model, _ = self.fixture()
        output, data = self.encode(model, "native")
        expected = snapshot_encoded(output)
        with self.assertRaisesRegex(RuntimeError, "generated_ids_exact"):
            assert_native_parity(expected, data["calls"][0]["encoded"], [4, 8], [4, 9])
        altered = {**expected, "features": {**expected["features"], "sha256": "bad"}}
        with self.assertRaisesRegex(RuntimeError, "features_exact"):
            assert_native_parity(expected, altered, [4, 8], [4, 8])
        with self.assertRaisesRegex(ValueError, "generated token IDs"):
            assert_native_parity(expected, expected, None, None)

    def test_selection_gate_rejects_a_changed_basis_assignment(self):
        import copy
        model, _ = self.fixture()
        _, native = self.encode(model, "native")
        _, changed = self.encode(model, "no-merge")
        corrupted = copy.deepcopy(changed["calls"][0])
        corrupted["basis_slot_to_source"][0]["source_index"] += 1
        with self.assertRaisesRegex(RuntimeError, "basis_slot_to_source_exact"):
            assert_selection_equal(native["calls"][0], corrupted)

    def test_uncompressed_native_branch_is_observed_but_not_a_merge_ablation(self):
        model, _ = self.fixture(keep=576)
        _, data = self.encode(model, "native")
        call = data["calls"][0]
        self.assertEqual(call["merge_calls"], 0)
        self.assertFalse(call["input"]["enabled"])
        self.assertEqual(call["encoded"]["kept_tokens"], 576)
        self.assertEqual(call["basis_slot_to_source"], [])
        with self.assertRaisesRegex(RuntimeError, "Unexpected no-merge baseline"):
            self.encode(model, "no-merge")

    def test_fixed_reference_scoring_matches_explicit_tiny_native_sequence(self):
        """Real decoder/labels, independent manual image-placeholder expansion."""
        torch = self.torch
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.model.language_model.llava_llama_x import LlavaConfig, LlavaLlamaForCausalLM_X
        from rq1_native import max_score_delta, reference_tokens, score_from_logits
        from run_rq2_input_merge import score_fixed_reference
        from smoke_original_apet import configure

        class CharacterTokenizer:
            bos_token_id = 1

            def __call__(self, text):
                return SimpleNamespace(input_ids=[1] + [3 + ord(char) % 120 for char in text])

        class Vision(torch.nn.Module):
            def __init__(self, features):
                super().__init__()
                self.features = features

            def forward(self, images):
                return self.features.clone(), None

        class FloatProjector(torch.nn.Module):
            def forward(self, features):
                return features.float()

        config = LlavaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            max_position_embeddings=2048, bos_token_id=1, eos_token_id=2, pad_token_id=0)
        config._attn_implementation = "sdpa"
        config.tokenizer_model_max_length = 2048
        torch.manual_seed(590)
        model = LlavaLlamaForCausalLM_X(config, visual_token_num=576, selected_indices=[]).eval()
        model.model.vision_tower = Vision(torch.randn(1, 577, 32))
        model.model.mm_projector = FloatProjector()
        tokenizer = CharacterTokenizer()
        case = {"case_id": "gqa:synthetic", "canonical_target": "red",
                "generation_question": "What color?"}
        pixels = torch.zeros(1, 3, 336, 336)
        prompt, ids, labels, _ = reference_tokens(tokenizer, case, "Red")
        marker = int((ids[0] == IMAGE_TOKEN_INDEX).nonzero().item())
        for arm, keep in (("baseline576", 576), ("native-input288", 288), ("native-input288-no-merge", 288)):
            configure(model, "baseline" if keep == 576 else "input-only")
            torch.manual_seed(590)
            variant = "no-merge" if arm == "native-input288-no-merge" else "native"
            with native_input_variant(model, variant, capture_tensors=True) as data:
                model.encode_images(pixels)
            encoded = data["snapshots"][0]["encoded"]
            actual = score_fixed_reference(tokenizer, model, case, pixels, encoded, arm, "Red")
            # Independent expansion puts the same retained states between the
            # literal text segments, with IGNORE_INDEX labels for every state.
            visual = encoded[0][0][encoded[1][0]]
            embed = model.get_input_embeddings()
            manual = torch.cat((embed(ids[0, :marker]), visual,
                                embed(ids[0, marker+1:])), dim=0).unsqueeze(0)
            manual_labels = torch.cat((labels[0, :marker],
                torch.full((keep,), -100, dtype=torch.long), labels[0, marker+1:])).unsqueeze(0)
            positions = torch.arange(manual.shape[1]).unsqueeze(0)
            attention = torch.ones_like(positions, dtype=torch.bool)
            with torch.inference_mode():
                output = model(inputs_embeds=manual, position_ids=positions,
                    attention_mask=attention, use_cache=False, return_dict=True)
                expected = score_from_logits(output.logits, manual_labels)
            self.assertEqual(actual["prompt"], prompt)
            self.assertEqual(actual["expanded_sequence_length"], ids.shape[1] - 1 + keep)
            self.assertEqual(actual["target_token_ids"], labels[labels != -100].tolist())
            self.assertEqual(max_score_delta(actual, expected), 0)
            self.assertEqual(model.model.image_tokens, [keep])


if __name__ == "__main__":
    unittest.main()
