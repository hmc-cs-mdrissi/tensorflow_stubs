from typing import Any, Iterable, Literal, Sequence
from typing_extensions import TypeAlias

from collections.abc import Callable

import tensorflow as tf
from tensorflow import Graph, IndexedSlices, Operation, Tensor, Variable
from tensorflow.compat.v1 import Session
from tensorflow.keras.optimizers import schedules

from _typeshed import Incomplete

_LearningRateT: TypeAlias = float | Tensor | schedules.LearningRateSchedule | Callable[[], float | Tensor]

class BaseSaverBuilder:
    def __getattr__(self, name: str) -> Any: ...

class Saver:
    def __init__(
        self,
        var_list: list[Variable] | set[Variable] | tuple[Variable, ...] | dict[str, Variable] | None = None,
        reshape: bool = False,
        sharded: bool = False,
        max_to_keep: int = 5,
        keep_checkpoint_every_n_hours: float = 10000.0,
        name: str | None = None,
        restore_sequentially: bool = False,
        # The incompletes are protobuf types.
        saver_def: Incomplete = None,
        builder: BaseSaverBuilder | None = None,
        defer_build: bool = False,
        allow_empty: bool = False,
        write_version: Incomplete = ...,
        pad_step_number: bool = False,
        save_relative_paths: bool = False,
        filename: str | None = None,
    ): ...
    def save(
        self,
        sess: Session | None,
        save_path: str,
        global_step: int | None = None,
        latest_filename: str | None = None,
        meta_graph_suffix: str = "meta",
        write_meta_graph: bool = True,
        write_state: bool = True,
        strip_default_attrs: bool = False,
        save_debug_info: bool = False,
    ): ...

GATE_NONE = 0
GATE_OP = 1
GATE_GRAPH = 2

class Optimizer:
    _use_locking: bool
    def __init__(self, use_locking: bool, name: str) -> None: ...
    def minimize(
        self,
        loss: Tensor,
        global_step: Variable | None = None,
        var_list: Sequence[Variable] | None = None,
        gate_gradients: Literal[0, 1, 2] = GATE_OP,
        aggregation_method: Literal[0, 1, 2] | None = None,
        colocate_gradients_with_ops: bool = False,
        name: str | None = None,
        grad_loss: Tensor | None = None,
    ) -> Operation: ...
    def apply_gradients(
        self,
        grads_and_vars: Iterable[tuple[Tensor | IndexedSlices, Variable]],
        global_step: Variable | None = None,
        name: str | None = None,
    ) -> Operation: ...
    def compute_gradients(
        self,
        loss: Tensor,
        var_list: Sequence[Variable] | None = None,
        gate_gradients: Literal[0, 1, 2] = GATE_OP,
        aggregation_method: tf.AggregationMethod | None = None,
        colocate_gradients_with_ops: bool = False,
        grad_loss: Tensor | None = None,
    ) -> list[tuple[Tensor | IndexedSlices, Variable]]: ...
    def get_slot(self, var: Variable, name: str) -> Variable | None: ...
    def get_slot_names(self) -> list[str]: ...
    def get_name(self) -> str: ...

class AdamOptimizer(Optimizer):
    _epsilon: float
    _beta1: float
    _beta2: float

    _lr_t: Tensor
    _epsilon_t: Tensor
    _beta1_t: Tensor
    _beta2_t: Tensor

    def __init__(
        self,
        learning_rate: _LearningRateT = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-08,
        use_locking: bool = False,
        name: str = "Adam",
    ) -> None: ...
    def _get_beta_accumulators(self) -> tuple[Variable, Variable]: ...

class AdagradOptimizer(Optimizer):
    _initial_accumulator_value: float
    def __init__(
        self,
        learning_rate: _LearningRateT,
        initial_accumulator_value: float = 0.1,
        use_locking: bool = False,
        name: str = "Adagrad",
    ): ...

class GradientDescentOptimizer(Optimizer):
    def __init__(
        self,
        learning_rate: _LearningRateT,
        use_locking: bool = False,
        name: str = "GradientDescent",
    ) -> None: ...

def get_global_step(graph: Graph | None = None) -> Variable | None: ...
def __getattr__(name: str) -> Any: ...
