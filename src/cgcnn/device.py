import os

import torch


def get_env_device(env_var: str = "CGCNN_DEVICE") -> str | None:
    value = os.environ.get(env_var)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _module_is_available(module_name: str) -> bool | None:
    module = getattr(torch, module_name, None)
    if module is None:
        return None
    is_available = getattr(module, "is_available", None)
    if callable(is_available):
        return bool(is_available())
    return None


def _device_type_is_available(device_type: str) -> bool | None:
    if device_type == "cpu":
        return True
    if device_type == "cuda":
        return bool(torch.cuda.is_available())
    if device_type == "mps":
        return bool(
            hasattr(torch, "backends")
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
    availability = _module_is_available(device_type)
    if availability is not None:
        return availability
    return None


def _default_accelerator_device() -> torch.device | None:
    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None:
        return None
    is_available = getattr(accelerator, "is_available", None)
    current_accelerator = getattr(accelerator, "current_accelerator", None)
    if not callable(is_available) or not callable(current_accelerator):
        return None
    if not is_available():
        return None
    return torch.device(current_accelerator())


def resolve_device(
    device: str | torch.device | None = None,
    cuda: bool | None = None,
) -> torch.device:
    if device is not None:
        resolved = torch.device(device)
        availability = _device_type_is_available(resolved.type)
        if availability is False:
            raise RuntimeError(
                f"Requested device '{resolved}', but backend '{resolved.type}' is not available."
            )
        return resolved

    if cuda is True:
        if not torch.cuda.is_available():
            raise RuntimeError("cuda=True was requested, but CUDA is not available.")
        return torch.device("cuda")
    if cuda is False:
        return torch.device("cpu")

    accelerator_device = _default_accelerator_device()
    if accelerator_device is not None:
        return accelerator_device

    for device_type in ("cuda", "xpu", "mps", "hpu", "mtia"):
        if _device_type_is_available(device_type):
            return torch.device(device_type)
    return torch.device("cpu")


def use_pinned_memory(device: torch.device) -> bool:
    return device.type in {"cuda", "xpu"}
