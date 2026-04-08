from typing import Literal

import numpy as np
from numpy.typing import ArrayLike

TransformName = Literal["raw", "sqrt", "cbrt", "log", "log10"]
POSITIVE_ONLY_TRANSFORMS: frozenset[TransformName] = frozenset({"log", "log10"})
NONNEGATIVE_TRANSFORMS: frozenset[TransformName] = frozenset({"sqrt"})
SUPPORTED_TRANSFORMS: frozenset[TransformName] = frozenset(
    {"raw", "sqrt", "cbrt", "log", "log10"}
)

__all__ = [
    "NONNEGATIVE_TRANSFORMS",
    "POSITIVE_ONLY_TRANSFORMS",
    "SUPPORTED_TRANSFORMS",
    "TransformName",
    "apply_transform",
    "invert_transform",
    "validate_transform_input",
]


def _as_float_array(values: float | ArrayLike) -> np.ndarray:
    return np.asarray(values, dtype=float)


def _is_scalar_input(values: float | ArrayLike) -> bool:
    return np.isscalar(values) or (isinstance(values, np.ndarray) and values.ndim == 0)


def _return_like_input(values: float | ArrayLike, output: np.ndarray) -> float | np.ndarray:
    if _is_scalar_input(values):
        return float(output.item())
    return output


def validate_transform_input(
    values: float | ArrayLike,
    transform: TransformName,
) -> np.ndarray:
    if transform not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"Unsupported transform {transform!r}. Expected one of {sorted(SUPPORTED_TRANSFORMS)}."
        )

    array = _as_float_array(values)
    if transform in POSITIVE_ONLY_TRANSFORMS and np.any(array <= 0.0):
        raise ValueError(f"Transform {transform!r} requires strictly positive values.")
    if transform in NONNEGATIVE_TRANSFORMS and np.any(array < 0.0):
        raise ValueError(f"Transform {transform!r} requires non-negative values.")
    return array


def apply_transform(values: float | ArrayLike, transform: TransformName) -> float | np.ndarray:
    array = validate_transform_input(values, transform)
    if transform == "raw":
        transformed = array.copy()
    elif transform == "sqrt":
        transformed = np.sqrt(array)
    elif transform == "cbrt":
        transformed = np.cbrt(array)
    elif transform == "log":
        transformed = np.log(array)
    else:
        transformed = np.log10(array)
    return _return_like_input(values, transformed)


def invert_transform(values: float | ArrayLike, transform: TransformName) -> float | np.ndarray:
    if transform not in SUPPORTED_TRANSFORMS:
        raise ValueError(
            f"Unsupported transform {transform!r}. Expected one of {sorted(SUPPORTED_TRANSFORMS)}."
        )

    array = _as_float_array(values)
    if transform == "raw":
        restored = array.copy()
    elif transform == "sqrt":
        restored = np.square(array)
    elif transform == "cbrt":
        restored = np.power(array, 3)
    elif transform == "log":
        restored = np.exp(array)
    else:
        restored = np.power(10.0, array)
    return _return_like_input(values, restored)
