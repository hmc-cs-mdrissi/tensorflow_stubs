from abc import ABC, abstractmethod
from typing import Any, Literal

import tensorflow as tf
from tensorflow import Tensor, _KerasSerializable  # pyright: ignore[reportPrivateUsage]
from tensorflow._aliases import TensorCompatible
from typing_extensions import Self, TypeGuard

class Loss(ABC):
    reduction: _ReductionValues
    name: str | None
    def __init__(
        self, reduction: _ReductionValues = "auto", name: str | None = None
    ): ...
    @abstractmethod
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor: ...
    @classmethod
    def from_config(cls: type[Self], config: dict[str, Any]) -> Self: ...
    def get_config(self) -> dict[str, Any]: ...
    def __call__(
        self,
        y_true: TensorCompatible,
        y_pred: TensorCompatible,
        sample_weight: TensorCompatible | None = None,
    ) -> Tensor: ...

class BinaryCrossentropy(Loss):
    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        axis: int = -1,
        reduction: _ReductionValues = Reduction.AUTO,
        name: str | None = "binary_crossentropy",
    ): ...
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor: ...

class CategoricalCrossentropy(Loss):
    def __init__(
        self,
        from_logits: bool = False,
        label_smoothing: float = 0.0,
        axis: int = -1,
        reduction: _ReductionValues = Reduction.AUTO,
        name: str | None = "categorical_crossentropy",
    ): ...
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor: ...

class MeanSquaredError(Loss):
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor: ...

class MeanSquaredLogarithmicError(Loss):
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor: ...

class SparseCategoricalCrossentropy(Loss):
    def __init__(
        self,
        name: str = "sparse_categorical_crossentropy",
        dtype: str | tf.DType | None = None,
        from_logits: bool = False,
        ignore_class: int | None = None,
        axis: int = -1,
    ) -> None: ...
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor: ...

class Huber(Loss):
    def __init__(
        self,
        delta: float = 1.0,
        reduction: _ReductionValues = Reduction.AUTO,
        name: str | None = "huber_loss",
    ): ...
    def call(self, y_true: Tensor, y_pred: Tensor) -> Tensor: ...

class Reduction:
    AUTO = "auto"
    NONE = "none"
    SUM = "sum"
    SUM_OVER_BATCH_SIZE = "sum_over_batch_size"
    @classmethod
    def all(cls) -> tuple[_ReductionValues, ...]: ...
    @classmethod
    def validate(cls, key: object) -> TypeGuard[_ReductionValues]: ...

_ReductionValues = Literal["auto", "none", "sum", "sum_over_batch_size"]

def binary_crossentropy(
    y_true: TensorCompatible,
    y_pred: TensorCompatible,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    axis: int = -1,
) -> Tensor: ...
def binary_focal_crossentropy(
    y_true: TensorCompatible,
    y_pred: TensorCompatible,
    apply_class_balancing: bool = False,
    alpha: float = 0.25,
    gamma: float = 2.0,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    axis: int = -1,
) -> Tensor: ...
def serialize(metric: _KerasSerializable) -> dict[str, Any]: ...
def categorical_crossentropy(
    y_true: TensorCompatible,
    y_pred: TensorCompatible,
    from_logits: bool = False,
    label_smoothing: float = 0.0,
    axis: int = -1,
) -> Tensor: ...
