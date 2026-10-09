# Extracted LLaVA input-stage ApET

`core.py` exposes `CompressionConfig`, `compress_tokens`, and `fps` using only
PyTorch. It imports no model code. `adapters/hf_llava.py` contains the integration
for the pinned HF LLaVA runtime. `legacy_reference.py` is an optional validation
helper that extracts the original functions from the checked-in source via AST;
it does not import the legacy LLaVA package or its dependencies.

From the repository root, the API is:

```python
from apet_compression import CompressionConfig, compress_tokens

result = compress_tokens(vision_features, CompressionConfig(keep_tokens=96, basis_tokens=10))
compressed_tokens = result.tokens  # [batch, 96, feature_dim]
```

This is an importable package in the checkout. The existing uv environment still
uses `package=false`; this first step does not create a separately published or
editable-installed distribution. The experiment entry point adds the checkout
root to its import path before using the package.

The extraction preserves the FPS implementation, FP32 regularized solve with
epsilon 1e-5, largest-error top-k, FP16 conversion, original basis-slot placement,
cosine merging, and final spatial ordering from `llava/model/llava_arch.py`.
The original input stage places FPS basis vectors in sorted positions corresponding
to the last M error-top-k slots; these are not necessarily the FPS source positions.
It merges discarded tokens into the K-M non-basis destinations. These choices are
preserved for fidelity, rather than redesigned during extraction. The caller's
input tensor is not mutated. Retaining all tokens is an exact identity operation.

Only FPS input-stage compression is integrated. Decoder-layer pruning, Qwen,
DPC selection, multiple images, batched HF generation, and training are outside
this first adapter. `merge=False` is available in the core for later ablations;
the experiment runner currently uses merging and epsilon 1e-5 for legacy parity.

The adapter uses removable hooks on one model instance's projector. It modifies
features before projection, projects the full grid as the original does, selects
the retained output tokens in spatial order, and shortens the expanded image
placeholders before calling the normal HF generation implementation. It changes
neither installed library files nor the global model class. The scoped hooks are
removed when the context exits, including after errors.

FPS starts from a random token in the original. Experiments use a private Torch
generator seeded with `seed + sample index`, so warmup/order does not change an
image's selection. Source-equivalence unit tests reset RNG for both the original
and extracted functions and check their FPS indices. The first real-image warmup
also compares the modified features and mask against original input-stage code
with those same FPS indices. It compares arithmetic under the new runtime, not
the entire original model under its old dependencies. FP16 merging may vary with
GPU reduction order; the feature comparison reports the observed max difference
and uses absolute/relative tolerance 0.005. Selection masks must match exactly.

Run `research/tests/test_apet_extraction.py` in the Kaggle project environment.
It checks source parity on CPU and each GPU, input immutability, exact identity,
private-RNG repeatability, tiny HF cached generation with shorter sequences, and
hook cleanup. No pretrained checkpoint download is needed for these tests.
