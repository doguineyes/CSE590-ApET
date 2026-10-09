"""Execute the original math functions for comparison, without legacy imports.

AST extraction deliberately reads the checked-in source instead of maintaining
another copy of the algorithm. This is a validation helper, not the core.
"""

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[1]
SOURCES = ["llava/model/utils.py", "llava/model/llava_arch.py"]


def source_hashes():
    return {name: hashlib.sha256((REPO / name).read_bytes()).hexdigest() for name in SOURCES}


def original_functions(fixed_basis=None):
    import torch
    import torch.nn.functional as F

    namespace = {"torch": torch, "F": F}
    utils = ast.parse((REPO / SOURCES[0]).read_text())
    architecture = ast.parse((REPO / SOURCES[1]).read_text())
    cls = next(node for node in architecture.body
               if isinstance(node, ast.ClassDef) and node.name == "LlavaMetaForCausalLM")
    nodes = [next(node for node in utils.body if isinstance(node, ast.FunctionDef) and node.name == "fps")]
    nodes.extend(node for node in cls.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"token_merging", "encode_images"})
    for node in nodes:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<original ApET math>", "exec"), namespace)
    if fixed_basis is not None:
        namespace["fps"] = lambda features, count: fixed_basis
    return namespace


def original_input_compression(features, config, fixed_basis=None):
    import torch

    functions = original_functions(fixed_basis)

    class Harness:
        token_merging = functions["token_merging"]
        encode_images = functions["encode_images"]

        def __init__(self):
            self.model = SimpleNamespace(basis_token_num=config.basis_tokens)
            cls_token = features.new_zeros(features.shape[0], 1, features.shape[2])
            self.model.get_vision_tower = lambda: lambda images: (torch.cat((cls_token, features), dim=1), None)
            self.model.mm_projector = lambda values: values

        def get_model(self):
            return self.model

        def get_visual_token_num(self):
            return config.keep_tokens

    return Harness().encode_images(None)


def verify_result(features, config, result):
    import torch

    if not config.merge or config.epsilon != 1e-5:
        raise ValueError("Legacy parity requires merging and epsilon=1e-5")
    # FPS has separate RNG/selection parity tests. Fixing its output here avoids
    # touching global RNG during a real-model diagnostic.
    original, mask = original_input_compression(features, config, result.basis_indices)
    torch.testing.assert_close(result.kept_mask, mask, rtol=0, atol=0)
    torch.testing.assert_close(result.full_features, original, rtol=5e-3, atol=5e-3)
    return {"status": "passed", "max_abs_difference": (result.full_features - original).abs().max().item(),
            "rtol": 5e-3, "atol": 5e-3,
            "scope": "Original input compression on identical vision features and FPS indices; "
                     "same installed PyTorch runtime, not full legacy-model equivalence."}
