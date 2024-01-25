from abc import ABC, abstractmethod
from collections.abc import Container
from os import PathLike
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    NamedTuple,
    Protocol,
    Sequence,
    TypeVar,
)

import tensorflow as tf
from tensorflow._aliases import (
    ContainerGeneric,
    TensorCompatible,
    TensorLike,
    any_array,
)
from tensorflow.compat.v1 import ConfigProto, RunMetadata, RunOptions, Session
from tensorflow.estimator import export as export
from tensorflow.estimator.export import ExportOutput
from tensorflow.keras.initializers import Initializer
from tensorflow.keras.losses import _ReductionValues  # type: ignore
from typing_extensions import Self, TypeAlias

_FeaturesT = TypeVar(
    "_FeaturesT", bound=TensorLike | Mapping[str, TensorLike], contravariant=True
)
_LabelsT = TypeVar(
    "_LabelsT", bound=tf.Tensor | Mapping[str, tf.Tensor], contravariant=True
)

# The duplication is an artifact of there is no nice way to indicate
# an argument may optionally be present. There's 8 possible cases
# and we currently have 3 (full version + 2 used in codebase).
# More cases can be added as needed.
class ModelFn1(Generic[_FeaturesT, _LabelsT], Protocol):
    def __call__(
        self,
        features: _FeaturesT,
        labels: _LabelsT | None,
        *,
        mode: ModeKeysT,
        config: RunConfig,
        params: dict[str, object],
    ) -> EstimatorSpec: ...

class ModelFn2(Generic[_FeaturesT, _LabelsT], Protocol):
    def __call__(
        self,
        features: _FeaturesT,
        labels: _LabelsT | None,
        *,
        mode: ModeKeysT,
        params: dict[str, object],
    ) -> EstimatorSpec: ...

class ModelFn3(Generic[_FeaturesT, _LabelsT], Protocol):
    def __call__(
        self,
        features: _FeaturesT,
        labels: _LabelsT | None,
        *,
        mode: ModeKeysT,
    ) -> EstimatorSpec: ...

class ModelFn4(Generic[_FeaturesT, _LabelsT], Protocol):
    def __call__(
        self,
        features: _FeaturesT,
        labels: _LabelsT | None,
    ) -> EstimatorSpec: ...

ModelFn = (
    ModelFn1[_FeaturesT, _LabelsT]
    | ModelFn2[_FeaturesT, _LabelsT]
    | ModelFn3[_FeaturesT, _LabelsT]
    | ModelFn4[_FeaturesT, _LabelsT]
)

class CheckpointSaverListener: ...
class SessionRunHook: ...

class Estimator(Generic[_FeaturesT, _LabelsT]):
    def __init__(
        self,
        model_fn: ModelFn[_FeaturesT, _LabelsT],
        model_dir: PathLike[str] | str | None = None,
        config: RunConfig | None = None,
        params: Mapping[str, object] | None = None,
        warm_start_from: str | WarmStartSettings | None = None,
    ) -> None: ...
    def _add_meta_graph_for_mode(
        self,
        builder: tf.compat.v1.saved_model.Builder,
        input_receiver_fn_map: dict[
            ModeKeysT, Callable[[], tf.estimator.export.ServingInputReceiver]
        ],
        checkpoint_path: str,
        save_variables: bool = True,
        mode: ModeKeysT = ModeKeys.PREDICT,
        export_tags: Iterable[str] | None = None,
        check_variables: bool = True,
        strip_default_attrs: bool = True,
    ): ...
    def train(
        self,
        input_fn: Callable[[], tf.data.Dataset[tuple[_FeaturesT, _LabelsT]]],
        hooks: Sequence[SessionRunHook] | None = None,
        steps: int | None = None,
        max_steps: int | None = None,
        saving_listeners: Sequence[CheckpointSaverListener] | None = None,
    ) -> Self: ...
    def __getattr__(self, name: str) -> Any: ...

class _DeviceFn(Protocol):
    def __call__(self, op: tf.Operation) -> str: ...

class RunConfig:
    def __init__(
        self,
        model_dir: str | None = None,
        tf_random_seed: int | None = None,
        save_summary_steps: int = 100,
        save_checkpoints_steps: int | None = ...,
        save_checkpoints_secs: int | None = ...,
        session_config: ConfigProto | None = None,
        keep_checkpoint_max: int = 5,
        keep_checkpoint_every_n_hours: int = 10000,
        log_step_count_steps: int = 100,
        train_distribute: tf.distribute.Strategy | None = None,
        device_fn: _DeviceFn | None = None,
        protocol: Literal["grpc", "grpc+verbs", None] = None,
        eval_distribute: tf.distribute.Strategy | None = None,
        # The documentation mentions tf.contrib.distribute.DistributeConfig,
        # but tf.contrib was entirely removed years ago.
        experimental_distribute: None = None,
        experimental_max_worker_delay_secs: int | None = None,
        session_creation_timeout_secs: int = 7200,
        checkpoint_save_graph_def: bool = True,
    ) -> None: ...
    def replace(self, **kwargs: object) -> RunConfig: ...
    @staticmethod
    def _replace(
        config: RunConfig,
        allowed_properties_list: Container[str] | None = None,
        **kwargs: object,
    ) -> RunConfig: ...
    @property
    def session_config(self) -> ConfigProto | None: ...
    @property
    def model_dir(self) -> str: ...
    @property
    def task_id(self) -> int: ...
    @property
    def task_type(self) -> str: ...
    @property
    def num_worker_replicas(self) -> int: ...
    @property
    def num_ps_replicas(self) -> int: ...
    @property
    def is_chief(self) -> bool: ...

class VocabInfo(NamedTuple):
    new_vocab: str
    new_vocab_size: int
    num_oov_buckets: int
    old_vocab: str
    old_vocab_size: int = -1
    backup_initializer: Initializer | str | None = None
    axis: Literal[0, 1] = 0

class WarmStartSettings(NamedTuple):
    ckpt_to_initialize_from: str
    vars_to_warm_start: str | Iterable[str] | Iterable[tf.Variable] | None = ".*"
    var_name_to_vocab_info: Mapping[str, VocabInfo] | None = None
    var_name_to_prev_var_name: Mapping[str, str] | None = None

class EstimatorSpec:
    def __init__(
        self,
        mode: ModeKeysT,
        predictions: tf.Tensor | Mapping[str, tf.Tensor] | None = None,
        loss: tf.Tensor | None = None,
        train_op: tf.Operation | None = None,
        eval_metric_ops: dict[str, Any] | None = None,
        export_outputs: Mapping[str, ExportOutput] | None = None,
        training_chief_hooks: Iterable[SessionRunHook] | None = None,
        training_hooks: Iterable[SessionRunHook] | None = None,
        evaluation_hooks: Iterable[SessionRunHook] | None = None,
        prediction_hooks: Iterable[SessionRunHook] | None = None,
    ) -> None: ...
    def __getattr__(self, name: str) -> Any: ...

class SessionRunContext:
    def __init__(self, original_args: SessionRunArgs, session: Session) -> None: ...
    @property
    def original_args(self) -> SessionRunArgs: ...
    @property
    def session(self) -> Session: ...
    @property
    def stop_requested(self) -> bool: ...
    def request_stop(self) -> None: ...

class SessionRunValues(NamedTuple):
    results: ContainerGeneric[any_array]
    options: RunOptions
    run_metadata: RunMetadata

FetchT = tf.Tensor | tf.Operation | tf.SparseTensor | str

class SessionRunArgs:
    def __init__(
        self,
        fetches: ContainerGeneric[FetchT],
        feed_dict: Mapping[tf.Tensor, TensorCompatible] | None = None,
    ) -> None: ...
    @property
    def fetches(self) -> ContainerGeneric[FetchT]: ...
    @property
    def feed_dict(self) -> Mapping[tf.Tensor, TensorCompatible]: ...

class ModeKeys:
    TRAIN = "train"
    EVAL = "eval"
    PREDICT = "infer"

ModeKeysT = Literal["train", "eval", "infer"]

class Head(ABC):
    _weight_column: str | None
    def create_estimator_spec(
        self,
        features: Mapping[str, tf.Tensor | tf.SparseTensor],
        mode: ModeKeys,
        logits: tf.Tensor,
        labels: tf.Tensor | Mapping[str, tf.Tensor] | None = None,
        optimizer: tf.keras.optimizers.Optimizer | None = None,
        trainable_variables: Sequence[tf.Variable] | None = None,
        train_op_fn: Callable[[tf.Tensor], tf.Operation] | None = None,
        update_ops: Sequence[tf.Operation] | None = None,
        regularization_losses: Sequence[tf.Tensor] | None = None,
    ) -> EstimatorSpec: ...
    @property
    def loss_reduction(self) -> _ReductionValues: ...
    @property
    def logits_dimension(self) -> int: ...
    @property
    def name(self) -> str: ...
    @abstractmethod
    def metrics(
        self, regularization_losses: Sequence[tf.Tensor] | None = None
    ) -> dict[str, tf.keras.metrics.Metric]: ...
    def _summary_key(self, key: str) -> str: ...
    @abstractmethod
    def update_metrics(
        self,
        eval_metrics: Mapping[str, tf.keras.metrics.Metric],
        features: Mapping[str, TensorLike],
        logits: tf.Tensor,
        labels: tf.Tensor,
        regularization_losses: list[tf.Tensor] | None = None,
    ) -> dict[str, tf.keras.metrics.Metric]: ...
    @abstractmethod
    def predictions(
        self, logits: tf.Tensor, keys: list[str] | None = None
    ) -> dict[str, tf.Tensor]: ...

_LossFn: TypeAlias = (
    Callable[[tf.Tensor, tf.Tensor], tf.Tensor]
    | Callable[
        [
            tf.Tensor,
            tf.Tensor,
            Mapping[str, tf.Tensor | tf.SparseTensor],
            _ReductionValues,
        ],
        tf.Tensor,
    ]
)

class BinaryClassHead(Head):
    _auc_key: str
    _thresholds: tuple[float, ...]
    _label_vocabulary: Sequence[str] | None
    _loss_fn: _LossFn | None
    def __init__(
        self,
        weight_column: str | None = None,
        thresholds: Iterable[float] | None = None,
        label_vocabulary: Sequence[str] | None = None,
        loss_reduction: _ReductionValues = tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
        loss_fn: _LossFn | None = None,
        name: str | None = None,
    ) -> None: ...
    def metrics(
        self, regularization_losses: Sequence[tf.Tensor] | None = None
    ) -> dict[str, tf.keras.metrics.Metric]: ...
    def update_metrics(
        self,
        eval_metrics: Mapping[str, tf.keras.metrics.Metric],
        features: Mapping[str, TensorLike],
        logits: tf.Tensor,
        labels: tf.Tensor,
        regularization_losses: list[tf.Tensor] | None = None,
    ) -> dict[str, tf.keras.metrics.Metric]: ...
    def predictions(
        self, logits: tf.Tensor, keys: list[str] | None = None
    ) -> dict[str, tf.Tensor]: ...

class MultiClassHead(Head):
    _n_classes: int
    _label_vocabulary: Sequence[str] | None
    _loss_fn: _LossFn | None
    def __init__(
        self,
        n_classes: int,
        weight_column: str | None = None,
        label_vocabulary: Sequence[str] | None = None,
        loss_reduction: _ReductionValues = tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
        loss_fn: _LossFn | None = None,
        name: str | None = None,
    ) -> None: ...
    def metrics(
        self, regularization_losses: Sequence[tf.Tensor] | None = None
    ) -> dict[str, tf.keras.metrics.Metric]: ...
    def update_metrics(
        self,
        eval_metrics: Mapping[str, tf.keras.metrics.Metric],
        features: Mapping[str, TensorLike],
        logits: tf.Tensor,
        labels: tf.Tensor,
        regularization_losses: list[tf.Tensor] | None = None,
    ) -> dict[str, tf.keras.metrics.Metric]: ...
    def predictions(
        self, logits: tf.Tensor, keys: list[str] | None = None
    ) -> dict[str, tf.Tensor]: ...
    def _processed_labels(self, logits: tf.Tensor, labels: tf.Tensor) -> tf.Tensor: ...
    def _unweighted_loss_and_weights(
        self,
        logits: tf.Tensor,
        label_ids: tf.Tensor,
        features: Mapping[str, TensorLike],
    ) -> tuple[tf.Tensor, tf.Tensor]: ...

class RegressionHead(Head):
    _logits_dimension: int
    _loss_fn: _LossFn | None
    _inverse_link_fn: Callable[[tf.Tensor], tf.Tensor] | None
    def __init__(
        self,
        label_dimension: int = 1,
        weight_column: str | None = None,
        loss_reduction: _ReductionValues = tf.losses.Reduction.SUM_OVER_BATCH_SIZE,
        loss_fn: _LossFn | None = None,
        inverse_link_fn: Callable[[tf.Tensor], tf.Tensor] | None = None,
        name: str | None = None,
    ) -> None: ...
    def metrics(
        self, regularization_losses: Sequence[tf.Tensor] | None = None
    ) -> dict[str, tf.keras.metrics.Metric]: ...
    def update_metrics(
        self,
        eval_metrics: Mapping[str, tf.keras.metrics.Metric],
        features: Mapping[str, TensorLike],
        logits: tf.Tensor,
        labels: tf.Tensor,
        regularization_losses: list[tf.Tensor] | None = None,
    ) -> dict[str, tf.keras.metrics.Metric]: ...
    def predictions(
        self, logits: tf.Tensor, keys: list[str] | None = None
    ) -> dict[str, tf.Tensor]: ...

def __getattr__(name: str) -> Any: ...
