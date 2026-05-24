# onecomp-runtime

Shared **int4 inference runtime** for [OneCompression](../OneCompression) packed
checkpoints. OneCompression *produces* GPTQ/RTN-packed `safetensors`; this is the
*consumer* side — the int4 GEMM layers, GPTQ unpack/dequant helpers, backend
selection, and a generic diffusion loader that every per-model runtime builds a
thin adapter on top of.

```
pip install onecomp-runtime              # import as onecomp_runtime
pip install onecomp-runtime[gemlite]     # + GemLite int4 kernels
pip install onecomp-runtime[diffusion]   # + diffusers (generic loader builds diffusers classes)
```

XPU is opt-in and needs the PyTorch XPU wheel index:

```bash
pip install onecomp-runtime[xpu] --index-url https://download.pytorch.org/whl/xpu --extra-index-url https://pypi.org/simple
```

## Why

The int4 leaf machinery was copy-pasted across the FLUX.2 / LTX-2.3 / FireRed /
Irodori runtimes — `fused_int4_linear.py` was byte-identical in three of them.
A fix to the kernel (K-padding, warmup buckets, dtype safety) had to be hand-
propagated to every repo. This package is the single source of truth.

## Layout

```
onecomp_runtime/
  layers/
    fused_int4_linear.py   # Triton dequant+GEMM (AutoGPTQ-v1, gs=32)
    gemlite_int4_linear.py # GemLite kernel wrapper (fp16 I/O)
    packed_linear.py       # PackedRTNLinear, PackedEmbedding (RTN uint8-nibble)
    packed_conv.py         # int4 Conv1d / ConvTranspose1d (DAC-VAE codecs)
  quant_utils.py           # GPTQ + RTN unpack/dequant helpers
  backend.py               # resolve_backend / can_use_fused / build_{gemlite,fused,eager}
  diffusion.py             # load_int4_model(build_meta_model, ...) — generic GPTQ loader
```

## Usage — a per-model runtime adapter

```python
from onecomp_runtime.diffusion import load_int4_model
from diffusers import Flux2Transformer2DModel

def load_int4_transformer(path, **kw):
    return load_int4_model(
        path,
        lambda cfg: Flux2Transformer2DModel.from_config(cfg),
        label="flux2-klein-lite",
        **kw,
    )
```

The only per-model code is `build_meta_model` (construct the bare module from the
checkpoint's `config_json`) and an optional `post_load(model)` hook for buffer
fixups (e.g. FireRed/Qwen-Image RoPE tables that meta-init leaves uninitialised).

## Backends

| backend | kernel | I/O dtype | when |
|---|---|---|---|
| `gemlite` | GemLite Triton int4 | fp16 only | CUDA large M (FLUX), if installed |
| `fused` | bundled Triton dequant+GEMM | fp16/bf16/fp32 | CUDA/XPU default; bf16-safe |
| `eager` | dequant once to `nn.Linear` | any | groupsize≠32, actorder, odd shapes |

`backend="auto"` → CUDA uses GemLite if importable, else fused; XPU uses fused;
CPU and non-Triton devices fall back to eager. **bf16 is the safe default** for
Qwen-Image / LTX (fp16 overflows to NaN); fp16 is only required on the GemLite
path.

## XPU / Triton-XPU

XPU support uses the same `FusedInt4Linear` Triton source as CUDA and dispatches
it on tensors placed on `xpu`. The runtime now avoids CUDA-only assumptions in
backend selection and warmup:

```python
model = load_int4_model(
    "model.safetensors",
    build_meta_model,
    device="xpu:0",
    dtype="bfloat16",
    backend="auto",   # resolves to fused on XPU
)
```

GemLite is intentionally CUDA-only here; requesting `backend="gemlite"` on XPU
raises a clear error. The bundled fused kernel has separate CUDA/XPU launch
heuristics, and XPU warmup autotunes each requested `M` bucket from a small set
of tile candidates before caching the fastest config.

## Checkpoint contract

A single `safetensors` with metadata keys `config_json`, `quant_layers_json`
(per-layer manifest: `name`, `wbits`, `groupsize`, `actorder`, `in_features`,
`out_features`), and `checkpoint_format` (`gptq` v1 / `gptq_v2`). The RTN tier
(`packed_linear`, `packed_conv`) consumes the encoder/embedding/conv extras that
Irodori-style checkpoints add — those runtimes drive the layers directly rather
than through `load_int4_model`.
