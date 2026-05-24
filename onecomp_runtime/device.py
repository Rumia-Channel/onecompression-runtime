"""Device helpers shared by CUDA and XPU runtimes."""
from __future__ import annotations

import torch


TRITON_DEVICE_TYPES = {"cuda", "xpu"}


def normalize_device(device: str | torch.device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def accelerator_available(device: str | torch.device) -> bool:
    dev = normalize_device(device)
    if dev.type == "cuda":
        return torch.cuda.is_available()
    if dev.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        return bool(xpu is not None and xpu.is_available())
    if dev.type == "cpu":
        return True
    return False


def ensure_device_available(device: str | torch.device) -> torch.device:
    dev = normalize_device(device)
    if not accelerator_available(dev):
        raise RuntimeError(
            f"device {dev!s} is not available in this PyTorch build/runtime"
        )
    return dev


def supports_triton(device: str | torch.device) -> bool:
    return normalize_device(device).type in TRITON_DEVICE_TYPES


def supports_gemlite(device: str | torch.device) -> bool:
    # GemLite's public wrapper is CUDA-oriented; XPU uses the bundled Triton path.
    return normalize_device(device).type == "cuda"


def synchronize(device: str | torch.device) -> None:
    dev = normalize_device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is not None:
            xpu.synchronize(dev)
