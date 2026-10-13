"""Check original image compression across projection devices, without weights.

Run with the original Kaggle interpreter. The native functions are extracted by
AST so no vision checkpoint, model loading, or duplicate compression math is used.
"""

from pathlib import Path
from types import SimpleNamespace
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


class OriginalProjectionDeviceTests(unittest.TestCase):
    def test_mask_indexes_projected_features_across_devices(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Run this check in the original Kaggle environment with Torch")
        if not torch.cuda.is_available():
            self.skipTest("CUDA required for the original device mismatch")
        from apet_compression.legacy_reference import original_functions

        functions = original_functions()

        class Harness:
            encode_images = functions["encode_images"]
            token_merging = functions["token_merging"]

            def __init__(self, features, count, destination):
                self.count = count
                cls = features.new_zeros(1, 1, features.shape[-1])

                def project(values):
                    # Observe the same invocation; GPU scatter reductions can
                    # differ slightly between independent compression runs.
                    self.projector_input = values.detach().clone()
                    return values.to(destination)

                self.model = SimpleNamespace(
                    basis_token_num=10,
                    get_vision_tower=lambda: lambda images: (torch.cat((cls, features), dim=1), None),
                    mm_projector=project,
                )

            def get_model(self):
                return self.model

            def get_visual_token_num(self):
                return self.count

        torch.manual_seed(590)
        features = torch.randn(1, 576, 32, device="cuda:0", dtype=torch.float16)
        destinations = ["cpu"]
        if torch.cuda.device_count() >= 2:
            destinations.append("cuda:1")
        for count in (576, 288):
            # Selection precedes projection and must survive a device transfer.
            torch.manual_seed(590)
            _, reference_mask = Harness(features.clone(), count, "cuda:0").encode_images(None)
            for destination in destinations:
                with self.subTest(keep_tokens=count, destination=destination):
                    torch.manual_seed(590)
                    harness = Harness(features.clone(), count, destination)
                    projected, mask = harness.encode_images(None)
                    self.assertEqual(mask.device, projected.device)
                    self.assertEqual(projected[0][mask[0]].shape, (count, 32))
                    self.assertEqual(int(mask.sum().item()), count)
                    torch.testing.assert_close(mask.cpu(), reference_mask.cpu(), rtol=0, atol=0)
                    torch.testing.assert_close(projected.cpu(), harness.projector_input.cpu(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
