"""Backend selection + per-layer int4 module construction.

These helpers are model-agnostic: they take a ``quant_layers`` entry (the dict
OneCompression writes into ``quant_layers_json``) plus the layer's loaded
tensors, and build the right int4 ``nn.Module`` for the chosen backend.

Backends:
    - ``gemlite`` — GemLite Triton int4 GEMM (fp16 I/O); fastest at large M.
    - ``fused``   — bundled fused Triton dequant+GEMM kernel.
    - ``eager``   — dequantize once to a plain ``nn.Linear`` (fallback for
      groupsize != 32, actorder, or odd shapes).

Copyright 2025-2026 Kizuna Intelligence / Fujitsu Ltd. MIT License.
"""
from __future__ import annotations

from typing import Any

import torch

from .layers.fused_int4_linear import FusedInt4Linear
from .layers.gemlite_int4_linear import GemLiteInt4Linear, gemlite_available
from .quant_utils import dequant_gptq_to_fp

_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    try:
        return _DTYPES[str(dtype).lower()]
    except KeyError:
        raise ValueError(f"unsupported dtype: {dtype!r}")


def can_use_fused(entry: dict[str, Any]) -> bool:
    """True if a layer meets the fused/gemlite kernel constraints."""
    return (
        not bool(entry.get("actorder", False))
        and int(entry["wbits"]) == 4
        and int(entry["groupsize"]) == 32
        and int(entry["in_features"]) % 32 == 0
        and int(entry["out_features"]) % 8 == 0
    )


def resolve_backend(backend: str | None, use_fused: bool = True) -> str:
    """Pick the int4 GEMM backend.

    ``backend`` takes precedence; ``use_fused=False`` (legacy) forces eager.
    ``"auto"`` prefers GemLite when importable (within ~15% of bf16 at large-M
    shapes), then the fused Triton kernel, else eager.
    """
    if not use_fused:
        return "eager"
    b = (backend or "auto").lower()
    if b == "auto":
        return "gemlite" if gemlite_available() else "fused"
    if b == "gemlite" and not gemlite_available():
        raise RuntimeError(
            "backend='gemlite' requested but the 'gemlite' package is not "
            "importable; pip install gemlite or use backend='fused'"
        )
    if b not in ("gemlite", "fused", "eager"):
        raise ValueError(f"unknown backend: {backend!r}")
    return b


def build_gemlite(entry: dict, st: dict, device: torch.device) -> GemLiteInt4Linear:
    return GemLiteInt4Linear(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=st["qweight"],
        scales=st["scales"],
        qzeros=st["qzeros"],
        bias=(st["bias"] if "bias" in st else None),
        groupsize=int(entry["groupsize"]),
        v1=entry.get("checkpoint_format", "gptq") != "gptq_v2",
        device=device,
    )


def build_fused(entry: dict, st: dict, device: torch.device) -> FusedInt4Linear:
    return FusedInt4Linear(
        in_features=int(entry["in_features"]),
        out_features=int(entry["out_features"]),
        qweight=st["qweight"].to(device),
        scales=st["scales"].to(device=device, dtype=torch.float16),
        qzeros=st["qzeros"].to(device),
        bias=(st["bias"].to(device) if "bias" in st else None),
        groupsize=32,
    )


def build_eager(entry: dict, st: dict, dtype: torch.dtype,
                device: torch.device) -> torch.nn.Linear:
    in_f, out_f = int(entry["in_features"]), int(entry["out_features"])
    g_idx = st.get("g_idx")
    if g_idx is None:
        gs = int(entry["groupsize"])
        g_idx = (torch.arange(in_f) // gs).to(torch.long)
    weight = dequant_gptq_to_fp(
        qweight=st["qweight"], scales=st["scales"], qzeros=st["qzeros"],
        g_idx=g_idx, in_features=in_f, out_features=out_f, dtype=dtype,
        v1=entry.get("checkpoint_format", "gptq") != "gptq_v2",
    ).to(device)
    has_bias = "bias" in st
    lin = torch.nn.Linear(in_f, out_f, bias=has_bias, device=device, dtype=dtype)
    with torch.no_grad():
        lin.weight.copy_(weight)
        if has_bias:
            lin.bias.copy_(st["bias"].to(dtype=dtype, device=device))
    return lin


def build_quant_layer(entry: dict, st: dict, backend: str, dtype: torch.dtype,
                      device: torch.device):
    """Dispatch a single quant-layer build, returning ``(module, kind)``.

    ``kind`` is one of ``"gemlite"``, ``"fused"``, ``"eager"`` — useful for
    load-summary counters.
    """
    if backend == "gemlite" and can_use_fused(entry):
        return build_gemlite(entry, st, device), "gemlite"
    if backend in ("gemlite", "fused") and can_use_fused(entry):
        return build_fused(entry, st, device), "fused"
    return build_eager(entry, st, dtype, device), "eager"
