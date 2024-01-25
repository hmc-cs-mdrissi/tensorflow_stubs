from abc import ABC
from collections.abc import Mapping
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Sequence,
    TypedDict,
    TypeVar,
    final,
    overload,
)

import tensorflow as tf
from tensorflow import DTypeLike, Tensor, Variable
from tensorflow._aliases import (
    ContainerInputSpec,
    TensorCompatible,
    TensorLike,
    float_array,
)
from tensorflow.distribute import Strategy
from tensorflow.keras.activations import ActivationT
from tensorflow.keras.constraints import ConstraintT
from tensorflow.keras.initializers import InitializerT
from tensorflow.keras.regularizers import RegularizerT
from tensorflow.python.feature_column.feature_column_v2 import (
    DenseColumn,
    SequenceDenseColumn,
)
from typing_extensions import Self, Unpack

_InputT = TypeVar("_InputT", contravariant=True)
_OutputT = TypeVar("_OutputT", covariant=True)

class InputSpec:
    dtype: str | None
    shape: tf.TensorShape | None
    ndim: int | None
    max_ndim: int | None
    min_ndim: int | None
    axes: dict[int, int | None] | None
    def __init__(
        self,
        dtype: tf.DTypeLike | None = None,
        shape: Iterable[int | None] | None = None,
        ndim: int | None = None,
        max_ndim: int | None = None,
        min_ndim: int | None = None,
        axes: dict[int, int | None] | None = None,
        allow_last_axis_squeeze: bool = False,
    ): ...

class _LayerKwargs(TypedDict, total=False):
    trainable: bool
    name: str | None
    dtype: DTypeLike | None
    dynamic: bool

class Layer(Generic[_InputT, _OutputT], tf.Module, ABC):
    input_spec: ContainerInputSpec
    trainable: bool
    def __init__(self, **kwargs: Unpack[_LayerKwargs]) -> None: ...
    @final
    def __call__(self, inputs: _InputT, *, training: bool = False) -> _OutputT: ...
    @overload
    def build(self: Layer[tf.Tensor, object], input_shape: tf.TensorShape) -> None: ...
    @overload
    def build(self, input_shape: Any) -> None: ...
    # Real type here in _InputShapeT and _OutputShapeT. If Higher order kinds
    # existed we could derive these from the input and output types. Without them
    # we would need to make this class have more generic arguments. Overloads at least
    # handle one common case.
    @overload
    def compute_output_shape(self, input_shape: Any) -> Any: ...
    @overload
    def compute_output_shape(
        self: Layer[tf.Tensor, tf.Tensor], input_shape: tf.TensorShape
    ) -> tf.TensorShape: ...
    def add_weight(
        self,
        name: str | None = None,
        shape: Iterable[int | None] | None = None,
        dtype: tf.DTypeLike | None = None,
        initializer: InitializerT | None = None,
        regularizer: tf.keras.regularizers.Regularizer
        | Callable[[tf.Tensor], tf.Tensor]
        | None = None,
        constraint: tf.keras.constraints.Constraint
        | Callable[[tf.Tensor], tf.Tensor]
        | None = None,
        trainable: bool | None = None,
    ) -> tf.Variable: ...
    def add_loss(
        self, losses: tf.Tensor | Sequence[tf.Tensor] | Callable[[], tf.Tensor]
    ) -> None: ...
    def call(
        self,
        inputs: _InputT,
        /,
    ) -> _OutputT: ...
    def count_params(self) -> int: ...
    @property
    def trainable_variables(self) -> list[Variable]: ...
    @property
    def non_trainable_variables(self) -> list[Variable]: ...
    @property
    def trainable_weights(self) -> list[Variable]: ...
    @property
    def non_trainable_weights(self) -> list[Variable]: ...
    @property
    def losses(self) -> list[Tensor]: ...
    built: bool
    def get_weights(self) -> list[float_array]: ...
    def set_weights(self, weights: Sequence[float_array]) -> None: ...
    def get_config(self) -> dict[str, Any]: ...
    @classmethod
    def from_config(cls: type[Self], config: dict[str, Any]) -> Self: ...
    @property
    def distribute_strategy(self) -> Strategy: ...
    @property
    def dtype(self) -> tf.DType: ...
    @property
    def variable_dtype(self) -> tf.DType: ...
    @property
    def compute_dtype(self) -> tf.DType: ...

class Dense(Layer[tf.Tensor, tf.Tensor]):
    input_spec: InputSpec
    def __init__(
        self,
        units: int,
        activation: ActivationT = None,
        use_bias: bool = True,
        kernel_initializer: InitializerT = "glorot_uniform",
        bias_initializer: InitializerT = "zeros",
        kernel_regularizer: RegularizerT = None,
        bias_regularizer: RegularizerT = None,
        activity_regularizer: RegularizerT = None,
        kernel_constraint: ConstraintT = None,
        bias_constraint: ConstraintT = None,
        **kwargs: Unpack[_LayerKwargs],
    ) -> None: ...
    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape: ...
    def call(self, inputs: tf.Tensor) -> tf.Tensor: ...

class BatchNormalization(Layer[tf.Tensor, tf.Tensor]):
    def __init__(
        self,
        axis: int | list[int] | tuple[int, ...] = -1,
        momentum: float = 0.99,
        epsilon: float = 0.001,
        center: bool = True,
        scale: bool = True,
        beta_initializer: InitializerT = "zeros",
        gamma_initializer: InitializerT = "ones",
        moving_mean_initializer: InitializerT = "zeros",
        moving_variance_initializer: InitializerT = "ones",
        beta_regularizer: RegularizerT = None,
        gamma_regularizer: RegularizerT = None,
        beta_constraint: ConstraintT = None,
        gamma_constraint: ConstraintT = None,
        renorm: bool = False,
        **kwargs: Unpack[_LayerKwargs],
    ): ...
    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape: ...
    def call(self, inputs: tf.Tensor) -> tf.Tensor: ...

class ReLU(Layer[tf.Tensor, tf.Tensor]):
    def __init__(
        self,
        max_value: float | None = None,
        negative_slope: float | None = 0.0,
        threshold: float | None = 0.0,
        **kwargs: Unpack[_LayerKwargs],
    ) -> None: ...
    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape: ...
    def call(self, inputs: tf.Tensor) -> tf.Tensor: ...

class Dropout(Layer[tf.Tensor, tf.Tensor]):
    def __init__(
        self,
        rate: float,
        noise_shape: TensorCompatible | Sequence[int | None] | None = None,
        seed: int | None = None,
        **kwargs: Unpack[_LayerKwargs],
    ) -> None: ...
    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape: ...
    def call(self, inputs: tf.Tensor) -> tf.Tensor: ...

class Embedding(Layer[TensorLike, tf.Tensor]):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        embeddings_initializer: InitializerT = "uniform",
        embeddings_regularizer: RegularizerT = None,
        embeddings_constraint: ConstraintT = None,
        mask_zero: bool = False,
        input_length: int | None = None,
        **kwargs: Unpack[_LayerKwargs],
    ): ...
    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape: ...
    def call(self, inputs: TensorLike) -> tf.Tensor: ...

class LayerNormalization(Layer[tf.Tensor, tf.Tensor]):
    def __init__(
        self,
        axis: int = -1,
        epsilon: float = 0.001,
        center: bool = True,
        scale: bool = True,
        beta_initializer: InitializerT = "zeros",
        gamma_initializer: InitializerT = "ones",
        beta_regularizer: RegularizerT = None,
        gamma_regularizer: RegularizerT = None,
        beta_constraint: ConstraintT = None,
        gamma_constraint: ConstraintT = None,
        **kwargs: Unpack[_LayerKwargs],
    ) -> None: ...

class _IndexLookup(Layer[TensorLike, TensorLike]):
    @overload
    def __call__(self, inputs: tf.Tensor) -> tf.Tensor: ...
    @overload
    def __call__(self, inputs: tf.SparseTensor) -> tf.SparseTensor: ...
    @overload
    def __call__(self, inputs: tf.RaggedTensor) -> tf.RaggedTensor: ...
    def vocabulary_size(self) -> int: ...

class StringLookup(_IndexLookup):
    def __init__(
        self,
        max_tokens: int | None = None,
        num_oov_indices: int = 1,
        mask_token: str | None = None,
        oov_token: str = "[UNK]",
        vocabulary: str | None | TensorCompatible = None,
        idf_weights: TensorCompatible | None = None,
        encoding: str = "utf-8",
        invert: bool = False,
        output_mode: Literal["int", "count", "multi_hot", "one_hot", "tf_idf"] = "int",
        sparse: bool = False,
        pad_to_max_tokens: bool = False,
    ) -> None: ...

class IntegerLookup(_IndexLookup):
    def __init__(
        self,
        max_tokens: int | None = None,
        num_oov_indices: int = 1,
        mask_token: int | None = None,
        oov_token: int = -1,
        vocabulary: str | None | TensorCompatible = None,
        vocabulary_dtype: Literal["int64", "int32"] = "int64",
        idf_weights: TensorCompatible | None = None,
        invert: bool = False,
        output_mode: Literal["int", "count", "multi_hot", "one_hot", "tf_idf"] = "int",
        sparse: bool = False,
        pad_to_max_tokens: bool = False,
    ) -> None: ...

class DenseFeatures(Layer[Mapping[str, TensorLike], tf.Tensor]):
    def __init__(
        self,
        feature_columns: Sequence[DenseColumn | SequenceDenseColumn],
        **kwargs: Unpack[_LayerKwargs],
    ) -> None: ...

class MultiHeadAttention(Layer[Any, tf.Tensor]):
    def __init__(
        self,
        num_heads: int,
        key_dim: int | None,
        value_dim: int | None = None,
        dropout: float = 0.0,
        use_bias: bool = True,
        output_shape: tuple[int, ...] | None = None,
        attention_axes: tuple[int, ...] | None = None,
        kernel_initialize: InitializerT = "glorot_uniform",
        bias_initializer: InitializerT = "zeros",
        kernel_regularizer: RegularizerT = None,
        bias_regularizer: RegularizerT = None,
        activity_regularizer: RegularizerT = None,
        kernel_constraint: ConstraintT = None,
        bias_constraint: ConstraintT = None,
        **kwargs: Unpack[_LayerKwargs],
    ) -> None: ...
    @overload
    def __call__(
        self,
        query: tf.Tensor,
        value: tf.Tensor,
        key: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        return_attention_scores: Literal[False] = False,
        training: bool = False,
        use_causal_mask: bool = False,
    ) -> tf.Tensor: ...
    @overload
    def __call__(
        self,
        query: tf.Tensor,
        value: tf.Tensor,
        key: tf.Tensor | None = None,
        attention_mask: tf.Tensor | None = None,
        return_attention_scores: Literal[True] = True,
        training: bool = False,
        use_causal_mask: bool = False,
    ) -> tuple[tf.Tensor, tf.Tensor]: ...

class GaussianDropout(Layer[tf.Tensor, tf.Tensor]):
    def __init__(
        self, rate: float, seed: int | None = None, **kwargs: Unpack[_LayerKwargs]
    ) -> None: ...

def __getattr__(name: str) -> Any: ...
