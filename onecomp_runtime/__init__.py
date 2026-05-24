"""onecomp-runtime — shared int4 inference runtime for OneCompression checkpoints.

OneCompression *produces* packed-int4 ``safetensors`` checkpoints. This package
is the *consumer* side: the int4 GEMM layers, GPTQ unpack/dequant helpers,
backend selection, and a generic diffusion loader that every per-model runtime
(FLUX.2, LTX-2.3, FireRed/Qwen-Image, Irodori-TTS, ...) builds a thin adapter on.

    from onecomp_runtime.diffusion import load_int4_model
    from onecomp_runtime.layers import FusedInt4Linear, GemLiteInt4Linear

Import name is ``onecomp_runtime``; the distribution is ``onecomp-runtime``.

Copyright 2025-2026 Kizuna Intelligence / Fujitsu Ltd. MIT License.
"""
from __future__ import annotations

from . import backend, device, quant_utils
from .device import (
    accelerator_available,
    ensure_device_available,
    supports_gemlite,
    supports_triton,
    synchronize,
)
from .diffusion import load_int4_model
from .layers import (
    FusedInt4Linear,
    GemLiteInt4Linear,
    PackedEmbedding,
    PackedInt4Conv1d,
    PackedInt4ConvTranspose1d,
    PackedRTNLinear,
    fused_int4_gemm,
    gemlite_available,
    replace_conv_with_packed,
)

__version__ = "0.1.0"

__all__ = [
    "load_int4_model",
    "FusedInt4Linear",
    "fused_int4_gemm",
    "GemLiteInt4Linear",
    "gemlite_available",
    "PackedRTNLinear",
    "PackedEmbedding",
    "PackedInt4Conv1d",
    "PackedInt4ConvTranspose1d",
    "replace_conv_with_packed",
    "accelerator_available",
    "ensure_device_available",
    "supports_gemlite",
    "supports_triton",
    "synchronize",
    "backend",
    "device",
    "quant_utils",
]
