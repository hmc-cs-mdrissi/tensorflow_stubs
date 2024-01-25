from typing import Callable, Literal

import tensorflow as tf
from tensorflow._aliases import ShapeLike
from typing_extensions import TypeAlias

ClipValueType: TypeAlias = tuple[float, float] | float

class _Optimizer: ...
class Adagrad(_Optimizer): ...
class Adam(_Optimizer): ...

class SGD(_Optimizer):
    def __init__(
        self,
        learning_rate: float | Callable[[], float] = 0.01,
        use_gradient_accumulation: bool = True,
        clip_weight_min: float | None = None,
        clip_weight_max: float | None = None,
        weight_decay_factor: float | None = None,
        multiply_weight_decay_factor_by_learning_rate: bool | None = None,
        clipvalue: ClipValueType | None = None,
        low_dimensional_packing_status: bool = False,
    ) -> None: ...

class FTRL(_Optimizer): ...

class TableConfig:
    vocabulary_size: int
    dim: int
    initializer: Callable[[ShapeLike], tf.Tensor]
    optimizer: _Optimizer | None
    combiner: Literal["mean", "sum", "sqrtn"]
    name: str | None

    def __init__(
        self,
        vocabulary_size: int,
        dim: int,
        initializer: Callable[[ShapeLike], tf.Tensor] | None = None,
        optimizer: _Optimizer | None = None,
        combiner: Literal["mean", "sum", "sqrtn"] = "mean",
        name: str | None = None,
    ): ...

class FeatureConfig:
    table: tf.tpu.experimental.embedding.TableConfig
    max_sequence_length: int
    validate_weights_and_indices: bool
    output_shape: tf.TensorShape
    name: str

    def __init__(
        self,
        table: tf.tpu.experimental.embedding.TableConfig,
        max_sequence_length: int = 0,
        validate_weights_and_indices: bool = True,
        output_shape: list[int] | tf.TensorShape | None = None,
        name: str | None = None,
    ): ...
