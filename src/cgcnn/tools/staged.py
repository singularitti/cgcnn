from dataclasses import dataclass
from math import inf

import numpy as np

from .thresholds import ThresholdClassifier
from .transforms import TransformName, apply_transform, invert_transform

__all__ = [
    "RegressorStage",
    "ThresholdRegressionPlan",
    "build_regressor_stages",
]


@dataclass(frozen=True)
class RegressorStage:
    """A single regressor bucket spanning a numeric interval."""

    name: str
    transform: TransformName = "raw"
    lower_bound: float | None = None
    upper_bound: float | None = None
    include_lower: bool = False
    include_upper: bool = True

    def contains(self, value: float) -> bool:
        if self.lower_bound is not None:
            if self.include_lower:
                if value < self.lower_bound:
                    return False
            elif value <= self.lower_bound:
                return False
        if self.upper_bound is not None:
            if self.include_upper:
                if value > self.upper_bound:
                    return False
            elif value >= self.upper_bound:
                return False
        return True

    def transform_value(self, value: float) -> float:
        return float(apply_transform(value, self.transform))

    def invert_value(self, value: float) -> float:
        return float(invert_transform(value, self.transform))

    def transform_values(self, values: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        return np.asarray(apply_transform(values, self.transform), dtype=float)

    def invert_values(self, values: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        return np.asarray(invert_transform(values, self.transform), dtype=float)


@dataclass(frozen=True)
class ThresholdRegressionPlan:
    """Plan a threshold classifier followed by one or more regressor stages."""

    classifier: ThresholdClassifier
    regressors: tuple[RegressorStage, ...]

    def __post_init__(self) -> None:
        if not self.regressors:
            raise ValueError("At least one regressor stage is required.")
        stage_names = [stage.name for stage in self.regressors]
        if len(stage_names) != len(set(stage_names)):
            raise ValueError("Regressor stage names must be unique.")

    def select_regressor(self, value: float) -> str | None:
        classifier_label = self.classifier.classify_value(value)
        if classifier_label != self.classifier.positive_label:
            return None
        for stage in self.regressors:
            if stage.contains(value):
                return stage.name
        raise ValueError(f"Value {value:.16g} does not fit any regressor stage.")

    def assign_regressors(
        self,
        values: list[float] | tuple[float, ...] | np.ndarray,
    ) -> list[str | None]:
        return [self.select_regressor(float(value)) for value in np.asarray(values, dtype=float)]

    def split_indices(
        self,
        values: list[float] | tuple[float, ...] | np.ndarray,
    ) -> dict[str, list[int]]:
        assignments = self.assign_regressors(values)
        splits: dict[str, list[int]] = {stage.name: [] for stage in self.regressors}
        for index, stage_name in enumerate(assignments):
            if stage_name is None:
                continue
            splits[stage_name].append(index)
        return splits

    def split_items(
        self,
        items: list[object] | tuple[object, ...],
        values: list[float] | tuple[float, ...] | np.ndarray,
    ) -> dict[str, list[object]]:
        if len(items) != len(values):
            raise ValueError("items and values must have the same length.")
        split_indices = self.split_indices(values)
        return {
            stage.name: [items[index] for index in split_indices[stage.name]]
            for stage in self.regressors
        }

    def transform_targets(
        self,
        values: list[float] | tuple[float, ...] | np.ndarray,
    ) -> dict[str, np.ndarray]:
        array = np.asarray(values, dtype=float)
        split_indices = self.split_indices(array)
        return {
            stage.name: stage.transform_values(array[split_indices[stage.name]])
            for stage in self.regressors
        }

    def inverse_transform_predictions(
        self,
        predictions_by_stage: dict[str, list[float] | tuple[float, ...] | np.ndarray],
    ) -> dict[str, np.ndarray]:
        restored: dict[str, np.ndarray] = {}
        for stage in self.regressors:
            if stage.name not in predictions_by_stage:
                continue
            restored[stage.name] = stage.invert_values(predictions_by_stage[stage.name])
        return restored


def build_regressor_stages(
    thresholds: list[float] | tuple[float, ...],
    *,
    names: list[str] | tuple[str, ...] | None = None,
    transforms: TransformName | list[TransformName] | tuple[TransformName, ...] = "raw",
    lower_bound: float | None = None,
) -> tuple[RegressorStage, ...]:
    sorted_thresholds = sorted(float(threshold) for threshold in thresholds)
    if len(sorted_thresholds) != len(set(sorted_thresholds)):
        raise ValueError("Regressor thresholds must be unique.")

    interval_count = len(sorted_thresholds) + 1
    if names is None:
        names = [f"regressor_{index + 1}" for index in range(interval_count)]
    if len(names) != interval_count:
        raise ValueError(
            f"Expected {interval_count} stage names, received {len(names)}."
        )

    transform_names: list[TransformName]
    if isinstance(transforms, str):
        transform_names = [transforms for _ in range(interval_count)]
    else:
        transform_names = list(transforms)
    if len(transform_names) != interval_count:
        raise ValueError(
            f"Expected {interval_count} transforms, received {len(transform_names)}."
        )

    boundaries = [lower_bound, *sorted_thresholds, inf]
    stages: list[RegressorStage] = []
    for index, name in enumerate(names):
        stage_lower = boundaries[index]
        stage_upper = boundaries[index + 1]
        stages.append(
            RegressorStage(
                name=name,
                transform=transform_names[index],
                lower_bound=stage_lower,
                upper_bound=None if stage_upper == inf else stage_upper,
                include_lower=index == 0 and lower_bound is not None,
                include_upper=True,
            )
        )
    return tuple(stages)
