from os import PathLike
from typing import (
    Any,
    Callable,
    Container,
    Generic,
    Iterable,
    Iterator,
    Literal,
    Mapping,
    Sequence,
    TypeVar,
    overload,
)

import tensorflow as tf
from tensorflow._aliases import (
    ContainerGeneric,
    DTypeLike,
    ShapeLike,
    TensorCompatible,
    any_array,
    float_array,
)
from tensorflow.data import Dataset
from tensorflow.keras import activations as activations
from tensorflow.keras import applications as applications
from tensorflow.keras import backend as backend
from tensorflow.keras import callbacks as callbacks
from tensorflow.keras import constraints as constraints
from tensorflow.keras import estimator as estimator
from tensorflow.keras import initializers as initializers
from tensorflow.keras import layers as layers
from tensorflow.keras import losses as losses
from tensorflow.keras import metrics as metrics
from tensorflow.keras import mixed_precision as mixed_precision
from tensorflow.keras import models as models
from tensorflow.keras import optimizers as optimizers
from tensorflow.keras import regularizers as regularizers
from tensorflow.keras import utils as utils
from tensorflow.keras.callbacks import Callback, CallbackList, History
from tensorflow.keras.layers import InputSpec, Layer, _InputT  # type: ignore
from tensorflow.keras.utils import Sequence as KerasSequence
from tensorflow.keras.utils.experimental import DatasetCreator
from tensorflow.saved_model import SaveOptions
from tensorflow.train import _CheckpointLoadStatus  # type: ignore
from tensorflow.types.experimental import ConcreteFunction, GenericFunction

_Loss = (
    str
    | tf.keras.losses.Loss
    | Callable[[TensorCompatible, TensorCompatible], tf.Tensor]
)
_Metric = (
    str
    | tf.keras.metrics.Metric
    | Callable[[TensorCompatible, TensorCompatible], tf.Tensor]
    | None
)

# Models are invariant unlike layers due to fit/evaluate.
_OutputT = TypeVar("_OutputT")
_Verbosity = Literal["auto", 0, 1, 2]
_ValidationDataT = (
    ContainerGeneric[any_array]
    | tuple[_InputT, _OutputT]
    | tuple[_InputT, _OutputT, tf.Tensor]
    | Dataset[tuple[_InputT, _OutputT]]
    | Dataset[tuple[_InputT, _OutputT, tf.Tensor]]
    | Sequence[tuple[_InputT, _OutputT]]
    | Sequence[tuple[_InputT, _OutputT, tf.Tensor]]
    | DatasetCreator[
        tuple[_InputT, _OutputT] | tuple[_InputT, _OutputT, tf.Tensor | _OutputT]
    ]
    | None
)

class _LossesContainer(Generic[_OutputT]):
    def __call__(
        self,
        y_true: _OutputT,
        y_pred: _OutputT,
        sample_weight: tf.Tensor | _OutputT | None = None,
        regularization_losses: Iterable[tf.Tensor] | None = None,
    ): ...

_BothOptimizer = tf.optimizers.Optimizer | tf.optimizers.experimental.Optimizer

class Model(Layer[_InputT, _OutputT]):
    _train_counter: tf.Variable
    _test_counter: tf.Variable
    optimizer: _BothOptimizer | None
    loss: tf.keras.losses.Loss | dict[str, tf.keras.losses.Loss]
    stop_training: bool

    # Ideally loss/metrics/output would share
    # the same structure but higher kinded types
    # are not supported.
    def compile(
        self,
        optimizer: str | _BothOptimizer = "rmsprop",
        loss: ContainerGeneric[_Loss] | None = None,
        metrics: ContainerGeneric[_Metric] | None = None,
        loss_weights: ContainerGeneric[float] | None = None,
        weighted_metrics: ContainerGeneric[_Metric] | None = None,
        run_eagerly: bool = False,
        steps_per_execution: int = 1,
        jit_compile: bool = False,
    ) -> None: ...
    def compute_loss(
        self,
        x: _InputT | None = None,
        y: _OutputT | None = None,
        y_pred: _OutputT | None = None,
        sample_weight: tf.Tensor | _OutputT | None = None,
    ) -> tf.Tensor: ...
    def compute_metrics(
        self,
        x: _InputT,
        y: _OutputT,
        y_pred: _OutputT,
        sample_weight: tf.Tensor | _OutputT | None,
    ) -> dict[str, tf.Tensor]: ...
    # The overloads mainly capture relationship between first argument
    # and the rest. There are other relationships between the arguments
    # that could be captured with more overloads. Fully specifying all
    # relationships is likely 10-20 overloads.
    @overload
    def fit(
        self,
        x: ContainerGeneric[any_array],
        y: ContainerGeneric[any_array],
        *,
        batch_size: int | None = None,
        epochs: int = 1,
        verbose: _Verbosity = "auto",
        callbacks: CallbackList | Sequence[Callback] | None = None,
        validation_split: float = 0.0,
        validation_data: _ValidationDataT[_InputT, _OutputT] = None,
        shuffle: bool = True,
        class_weight: Mapping[int, float] | None = None,
        sample_weight: float_array | None = None,
        initial_epoch: int = 0,
        steps_per_epoch: int | None = None,
        validation_steps: int | None = None,
        validation_batch_size: int | None = None,
        validation_freq: int | Container[int] = 1,
    ) -> History: ...
    @overload
    def fit(
        self,
        x: _InputT,
        y: _OutputT,
        *,
        batch_size: int | None = None,
        epochs: int = 1,
        verbose: _Verbosity = "auto",
        callbacks: CallbackList | Sequence[Callback] | None = None,
        validation_split: float = 0.0,
        validation_data: _ValidationDataT[_InputT, _OutputT] = None,
        shuffle: bool = True,
        class_weight: Mapping[int, float] | None = None,
        sample_weight: float_array | None = None,
        initial_epoch: int = 0,
        steps_per_epoch: int | None = None,
        validation_steps: int | None = None,
        validation_batch_size: int | None = None,
        validation_freq: int | Container[int] = 1,
    ) -> History: ...
    @overload
    def fit(
        self,
        # Third generic argument is optional sample weights.
        # It can either be one tensor shared for all outputs
        # or a container of tensors matching the output structure.
        x: Dataset[
            tuple[_InputT, _OutputT] | tuple[_InputT, _OutputT, tf.Tensor | _OutputT]
        ]
        | Iterator[
            tuple[_InputT, _OutputT] | tuple[_InputT, _OutputT, tf.Tensor | _OutputT]
        ]
        | KerasSequence[
            tuple[_InputT, _OutputT] | tuple[_InputT, _OutputT, tf.Tensor | _OutputT]
        ]
        | DatasetCreator[
            tuple[_InputT, _OutputT] | tuple[_InputT, _OutputT, tf.Tensor | _OutputT]
        ],
        *,
        epochs: int = 1,
        verbose: _Verbosity = "auto",
        callbacks: CallbackList | Sequence[Callback] | None = None,
        validation_data: _ValidationDataT[_InputT, _OutputT] = None,
        class_weight: Mapping[int, float] | None = None,
        initial_epoch: int = 0,
        steps_per_epoch: int | None = None,
        validation_steps: int | None = None,
        validation_batch_size: int | None = None,
        validation_freq: int | Container[int] = 1,
        validation_split: float = 0.0,
        max_queue_size: int = 10,
        workers: int = 1,
        use_multiprocessing: bool = False,
    ) -> History: ...
    def get_layer(
        self, name: str | None = None, index: int | None = None
    ) -> Layer[Any, Any]: ...
    @overload
    def evaluate(
        self,
        x: ContainerGeneric[any_array],
        y: ContainerGeneric[any_array],
        *,
        batch_size: int | None = None,
        verbose: _Verbosity = "auto",
        callbacks: CallbackList | Sequence[Callback] | None = None,
        sample_weight: float_array | None = None,
        steps: int | None = None,
        return_dict: bool = False,
    ) -> History: ...
    @overload
    def evaluate(
        self,
        x: Dataset[
            tuple[_InputT, _OutputT] | tuple[_InputT, _OutputT, tf.Tensor | _OutputT]
        ]
        | Iterator[
            tuple[_InputT, _OutputT] | tuple[_InputT, _OutputT, tf.Tensor | _OutputT]
        ]
        | KerasSequence[
            tuple[_InputT, _OutputT] | tuple[_InputT, _OutputT, tf.Tensor | _OutputT]
        ]
        | DatasetCreator[
            tuple[_InputT, _OutputT] | tuple[_InputT, _OutputT, tf.Tensor | _OutputT]
        ],
        *,
        batch_size: int | None = None,
        verbose: _Verbosity = "auto",
        callbacks: CallbackList | Sequence[Callback] | None = None,
        sample_weight: float_array | None = None,
        steps: int | None = None,
        return_dict: bool = False,
    ) -> History: ...

    # Fix type Any later
    def predict(
        self,
        x: Any,
        batch_size: int | None = None,
        verbose: str = "auto",
        steps: int | None = None,
        callbacks: CallbackList | Sequence[Callback] | None = None,
        max_queue_size: int = 10,
        workers: int = 1,
        use_multiprocessing: bool = False,
    ) -> Any: ...
    def call(self, inputs: _InputT) -> _OutputT: ...
    def compute_output_shape(self, input_shape: Any) -> Any: ...
    def save(
        self,
        filepath: str | PathLike[str],
        overwrite: bool = True,
        include_optimizer: bool = True,
        save_format: Literal["h5", "tf", None] = None,
        signatures: ConcreteFunction[..., object]
        | GenericFunction[..., object]
        | Mapping[str, ConcreteFunction[..., object] | GenericFunction[..., object]]
        | None = None,
        options: SaveOptions | None = None,
        save_traces: bool = True,
    ) -> None: ...
    def save_weights(
        self,
        filepath: str,
        overwrite: bool = True,
        save_format: Literal["tf", "h5", None] = None,
        options: tf.train.CheckpointOptions | None = None,
    ): ...
    def load_weights(
        self,
        filepath: str,
        by_name: bool = False,
        skip_mismatch: bool = False,
        options: tf.train.CheckpointOptions | None = None,
    ) -> _CheckpointLoadStatus: ...
    def compiled_loss(
        self,
        y: _OutputT | None,
        y_pred: _OutputT | None,
        sample_weight: tf.Tensor | _OutputT | None,
        regularization_losses: Sequence[tf.Tensor],
    ) -> tf.Tensor: ...
    def reset_metrics(self) -> None: ...
    def summary(
        self,
        line_length: str | None = None,
        positions: list[float] | None = None,
        print_fn: Callable[[str], object] | None = None,
        expand_nested: bool = False,
        show_trainable: bool = False,
    ) -> None: ...
    @property
    def layers(self) -> list[Layer[Any, Any]]: ...

@overload
def Input(
    *,
    name: str | None = None,
    type_spec: tf.TypeSpec | None = None,
) -> tf.Tensor: ...
@overload
def Input(
    shape: ShapeLike | None = None,
    batch_size: int | None = None,
    name: str | None = None,
    dtype: DTypeLike | None = None,
    sparse: bool | None = None,
    tensor: tf.Tensor | None = None,
    ragged: bool | None = None,
) -> tf.Tensor: ...

_T = TypeVar("_T", bound=Layer[tf.Tensor, tf.Tensor])

class Sequential(Model[tf.Tensor, tf.Tensor], Generic[_T]):
    input_spec: InputSpec
    layers: list[_T]
    def __init__(
        self, layers: Sequence[_T] | None = None, name: str | None = None
    ) -> None: ...
    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape: ...
    def call(self, inputs: tf.Tensor, training: bool | None = None) -> tf.Tensor: ...
    def add(self, layer: _T) -> None: ...
