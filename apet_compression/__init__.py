"""Model-independent ApET primitives; adapters live in apet_compression.adapters."""

from .core import CompressionConfig, CompressionResult, compress_tokens, fps

__all__ = ["CompressionConfig", "CompressionResult", "compress_tokens", "fps"]
