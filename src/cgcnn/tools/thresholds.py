from dataclasses import dataclass

import numpy as np

__all__ = [
    "ThresholdClassifier",
    "generate_threshold_labels",
    "split_indices_by_threshold",
    "split_items_by_threshold",
]


@dataclass(frozen=True)
class ThresholdClassifier:
    """Binary threshold classifier for scalar targets.

    Values strictly greater than ``threshold`` are assigned ``positive_label`` by
    default, matching the threshold-splitting logic used in the experiment
    scripts.
    """

    threshold: float
    positive_label: int = 1
    negative_label: int = 0
    strict_greater: bool = True

    def classify_value(self, value: float) -> int:
        if self.strict_greater:
            return self.positive_label if value > self.threshold else self.negative_label
        return self.positive_label if value >= self.threshold else self.negative_label

    def classify_values(self, values: list[float] | tuple[float, ...] | np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if self.strict_greater:
            mask = array > self.threshold
        else:
            mask = array >= self.threshold
        labels = np.full(array.shape, self.negative_label, dtype=int)
        labels[mask] = self.positive_label
        return labels

    def split_indices(self, values: list[float] | tuple[float, ...] | np.ndarray) -> tuple[list[int], list[int]]:
        negative_indices: list[int] = []
        positive_indices: list[int] = []
        for index, label in enumerate(self.classify_values(values).tolist()):
            if label == self.positive_label:
                positive_indices.append(index)
            else:
                negative_indices.append(index)
        return negative_indices, positive_indices

    def split_items(
        self,
        items: list[object] | tuple[object, ...],
        values: list[float] | tuple[float, ...] | np.ndarray,
    ) -> tuple[list[object], list[object]]:
        negative_indices, positive_indices = self.split_indices(values)
        return (
            [items[index] for index in negative_indices],
            [items[index] for index in positive_indices],
        )


def generate_threshold_labels(
    values: list[float] | tuple[float, ...] | np.ndarray,
    threshold: float,
    *,
    positive_label: int = 1,
    negative_label: int = 0,
    strict_greater: bool = True,
) -> np.ndarray:
    classifier = ThresholdClassifier(
        threshold=threshold,
        positive_label=positive_label,
        negative_label=negative_label,
        strict_greater=strict_greater,
    )
    return classifier.classify_values(values)


def split_indices_by_threshold(
    values: list[float] | tuple[float, ...] | np.ndarray,
    threshold: float,
    *,
    strict_greater: bool = True,
) -> tuple[list[int], list[int]]:
    classifier = ThresholdClassifier(threshold=threshold, strict_greater=strict_greater)
    return classifier.split_indices(values)


def split_items_by_threshold(
    items: list[object] | tuple[object, ...],
    values: list[float] | tuple[float, ...] | np.ndarray,
    threshold: float,
    *,
    strict_greater: bool = True,
) -> tuple[list[object], list[object]]:
    if len(items) != len(values):
        raise ValueError("items and values must have the same length.")
    classifier = ThresholdClassifier(threshold=threshold, strict_greater=strict_greater)
    return classifier.split_items(items, values)
