"""Generic int4 loader for OneCompression GPTQ-packed diffusion checkpoints.

A checkpoint produced by an OneCompression ``DiffusionTransformerAdapter``
(``save_quantized_model``) is a single ``safetensors`` file carrying two JSON
metadata blobs:

    - ``config_json``       — the model config to rebuild the bare module
    - ``quant_layers_json`` — per-layer manifest: ``{name, wbits, groupsize,
      actorder, in_features, out_features}`` for every quantized ``nn.Linear``

:func:`load_int4_model` reads those, rebuilds the model on the meta device via a
caller-supplied ``build_meta_model`` callback, swaps each quantized layer for a
fused / gemlite / eager int4 module, and materialises the remaining tensors on
the target device. The only per-model logic is the ``build_meta_model`` (how to
construct the bare module from the config dict) and an optional ``post_load``
hook for buffer fixups (e.g. recomputing RoPE tables left on meta).

Each runtime repo collapses to a thin adapter::

    from onecomp_runtime.diffusion import load_int4_model
    from diffusers import Flux2Transformer2DModel

    def load_int4_transformer(path, **kw):
        return load_int4_model(
            path,
            lambda cfg: Flux2Transformer2DModel.from_config(cfg),
            label="flux2-klein-lite",
            **kw,
        )

Copyright 2025-2026 Kizuna Intelligence / Fujitsu Ltd. MIT License.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

import torch
from safetensors import safe_open

from .backend import build_quant_layer, resolve_backend, resolve_dtype
from .device import ensure_device_available, supports_triton, synchronize
from .layers.fused_int4_linear import FusedInt4Linear
from .layers.gemlite_int4_linear import GemLiteInt4Linear


def load_int4_model(
    checkpoint_path: str,
    build_meta_model: Callable[[dict], torch.nn.Module],
    *,
    device: str = "cuda:0",
    dtype: str | torch.dtype = "bfloat16",
    backend: str | None = "auto",
    use_fused: bool = True,
    post_load: Callable[[torch.nn.Module], None] | None = None,
    warmup: bool = True,
    warmup_m_values: tuple[int, ...] = (64, 128, 256, 1024, 4096),
    label: str = "onecomp_runtime",
) -> torch.nn.Module:
    """Rebuild a packed-int4 diffusion transformer for fast inference.

    Args:
        checkpoint_path: path to the packed ``model.safetensors``.
        build_meta_model: ``(config_dict) -> nn.Module``. Called inside a
            ``torch.device("meta")`` context; should construct the bare model
            from the config (e.g. ``SomeTransformer.from_config(cfg)``).
        device: target device, e.g. ``"cuda:0"`` or ``"xpu:0"``.
        dtype: compute dtype for non-quantized tensors (bf16 recommended;
            fp16 only on the gemlite path, and never for Qwen-Image/LTX which
            overflow to NaN).
        backend: ``"auto"`` (CUDA: gemlite if importable, else fused; XPU:
            fused), ``"gemlite"``, ``"fused"``, or ``"eager"``.
        use_fused: legacy switch; ``False`` forces the eager-dequant fallback.
        post_load: optional ``(model) -> None`` hook run after weights are
            loaded — for per-model buffer fixups (rope tables, etc.).
        warmup: trigger Triton JIT for the M-buckets inference will hit.
        warmup_m_values: token counts to pre-compile (batch * seq).
        label: prefix used in the load-summary print line.

    Returns:
        an ``eval``-mode ``nn.Module`` on ``device``.
    """
    resolved = resolve_dtype(dtype)
    dev = ensure_device_available(device)
    backend = resolve_backend(backend, use_fused, dev)
    path = Path(checkpoint_path)

    with safe_open(str(path), framework="pt", device="cpu") as f:
        md = f.metadata() or {}
        cfg_json = md.get("config_json")
        quant_layers_json = md.get("quant_layers_json")
        if cfg_json is None or quant_layers_json is None:
            raise ValueError(
                f"{path} is missing 'config_json' / 'quant_layers_json' "
                "metadata; not a packed OneCompression diffusion checkpoint"
            )
        cfg = {k: v for k, v in json.loads(cfg_json).items() if not k.startswith("_")}
        quant_layers = json.loads(quant_layers_json)
        ckpt_fmt = md.get("checkpoint_format", "gptq")
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    # Build the bare model on meta, then materialise non-quant tensors directly
    # on the target device — never allocate the full bf16 weight set in VRAM.
    with torch.device("meta"):
        model = build_meta_model(cfg)

    modules = dict(model.named_modules())
    counts = {"gemlite": 0, "fused": 0, "eager": 0}
    quant_keys: set[str] = set()
    quant_names: list[str] = []
    for entry in quant_layers:
        entry["checkpoint_format"] = ckpt_fmt
        name = entry["name"]
        quant_names.append(name)
        parent_name, _, child = name.rpartition(".")
        parent = modules.get(parent_name) if parent_name else model
        if parent is None:
            raise KeyError(f"quant layer parent not found: {parent_name!r}")
        st: dict[str, Any] = {}
        for s in ("qweight", "scales", "qzeros", "g_idx", "bias"):
            k = f"{name}.{s}"
            if k in tensors:
                st[s] = tensors[k]
                quant_keys.add(k)
        quant_keys.add(f"{name}.weight")
        layer, kind = build_quant_layer(entry, st, backend, resolved, dev)
        counts[kind] += 1
        setattr(parent, child, layer)

    # Materialise the remaining (non-quant) tensors on-device.
    non_quant = {
        k: v.to(device=dev, dtype=resolved if v.dtype.is_floating_point else v.dtype)
        for k, v in tensors.items()
        if k not in quant_keys
    }
    missing, unexpected = model.load_state_dict(non_quant, strict=False, assign=True)
    # Swapped quant modules own their buffers; none come from the checkpoint, so
    # drop any missing key that lives under a quantized layer's subtree.
    quant_prefixes = tuple(f"{n}." for n in quant_names)
    real_missing = [
        m for m in missing
        if m not in quant_keys and not m.startswith(quant_prefixes)
    ]
    if real_missing:
        raise RuntimeError(f"missing keys: {real_missing[:8]} ...")
    if unexpected:
        raise RuntimeError(f"unexpected keys: {unexpected[:8]} ...")

    if post_load is not None:
        post_load(model)

    leftover = [n for n, p in model.named_parameters() if p.is_meta]
    if leftover:
        raise RuntimeError(f"parameter left on meta device: {leftover[:8]} ...")

    model.eval()
    print(f"[{label}] loaded {len(quant_layers)} int4 layers "
          f"(gemlite={counts['gemlite']}, fused={counts['fused']}, "
          f"eager={counts['eager']}) on {dev}")

    if warmup and supports_triton(dev):
        t0 = time.perf_counter()
        seen: dict[tuple, Any] = {}
        for m in model.modules():
            if isinstance(m, (FusedInt4Linear, GemLiteInt4Linear)):
                seen.setdefault((m.in_features, m.out_features, m.bias is not None), m)
        if seen:
            for layer in seen.values():
                layer.warmup(m_values=warmup_m_values)
            synchronize(dev)
            print(f"[{label}] warmup {(time.perf_counter()-t0)*1000:.0f} ms "
                  f"({len(seen)} unique signatures)")
    return model
