from typing import Any, Callable, Iterable, Literal, Sequence
from typing_extensions import Self

from abc import abstractmethod

import tensorflow as tf
from tensorflow import _KerasSerializable  # pyright: ignore[reportPrivateUsage]
from tensorflow import DTypeLike, Operation, Tensor, TensorCompatible

_OutputT = Tensor | dict[str, Tensor]

class Metric(tf.keras.layers.Layer[tf.Tensor, tf.Tensor]):
    def merge_state(self, metrics: Iterable[Self]) -> list[Operation]: ...
    def reset_state(self) -> None: ...
    @abstractmethod
    def update_state(
        self, y_true: TensorCompatible, y_pred: TensorCompatible, sample_weight: TensorCompatible | None = None
    ) -> Operation | None: ...
    @abstractmethod
    def result(self) -> _OutputT: ...
    @classmethod
    def from_config(cls: type[Self], config: dict[str, Any]) -> Self: ...
    def get_config(self) -> dict[str, Any]: ...

class AUC(Metric):
    _from_logits: bool
    _num_labels: int
    num_labels: int | None
    def __init__(
        self,
        num_thresholds: int = 200,
        curve: Literal["ROC", "PR"] = "ROC",
        summation_method: Literal["interpolation", "minoring", "majoring"] = "interpolation",
        name: str | None = None,
        dtype: DTypeLike | None = None,
        thresholds: Sequence[float] | None = None,
        multi_label: bool = False,
        num_labels: int | None = None,
        label_weights: TensorCompatible | None = None,
        from_logits: bool = False,
    ): ...
    def update_state(
        self, y_true: TensorCompatible, y_pred: TensorCompatible, sample_weight: TensorCompatible | None = None
    ) -> Operation: ...
    def result(self) -> tf.Tensor: ...

class Precision(Metric):
    def __init__(
        self,
        thresholds: float | Sequence[float] | None = None,
        top_k: int | None = None,
        class_id: int | None = None,
        name: str | None = None,
        dtype: DTypeLike | None = None,
    ): ...
    def update_state(
        self, y_true: TensorCompatible, y_pred: TensorCompatible, sample_weight: TensorCompatible | None = None
    ) -> Operation: ...
    def result(self) -> tf.Tensor: ...

class Recall(Metric):
    def __init__(
        self,
        thresholds: float | Sequence[float] | None = None,
        top_k: int | None = None,
        class_id: int | None = None,
        name: str | None = None,
        dtype: DTypeLike | None = None,
    ): ...
    def update_state(
        self, y_true: TensorCompatible, y_pred: TensorCompatible, sample_weight: TensorCompatible | None = None
    ) -> Operation: ...
    def result(self) -> tf.Tensor: ...

class BinaryAccuracy(MeanMetricWrapper):
    def __init__(self, name: str | None = None, dtype: DTypeLike | None = None, threshold: float = 0.5): ...

class Accuracy(MeanMetricWrapper):
    def __init__(self, name: str | None = None, dtype: DTypeLike | None = None): ...

class CategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, name: str | None = None, dtype: DTypeLike | None = None): ...

class TopKCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, k: int = 5, name: str | None = None, dtype: DTypeLike | None = None): ...

class SparseTopKCategoricalAccuracy(MeanMetricWrapper):
    def __init__(self, k: int = 5, name: str | None = None, dtype: DTypeLike | None = None): ...

class MeanMetricWrapper(Metric):
    def __init__(
        self, fn: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], name: str | None = None, threshold: float = 0.5
    ): ...
    def update_state(
        self, y_true: TensorCompatible, y_pred: TensorCompatible, sample_weight: TensorCompatible | None = None
    ) -> Operation: ...
    def result(self) -> tf.Tensor: ...

def serialize(metric: _KerasSerializable) -> dict[str, Any]: ...
