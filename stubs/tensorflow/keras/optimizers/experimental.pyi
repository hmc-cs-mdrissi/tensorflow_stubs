from typing import Any
from typing_extensions import Self

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Sequence

import tensorflow as tf
from tensorflow.keras.optimizers import _GradientsT, _LearningRateT  # pyright: ignore
from tensorflow.python.training.tracking.autotrackable import AutoTrackable

class Optimizer(AutoTrackable, ABC):
    name: str
    _current_learning_rate: tf.Variable
    _learning_rate: _LearningRateT
    _critical_section: tf.CriticalSection

    jit_compile: bool
    _index_dict: dict[str, int]
    _iterations: tf.Variable
    def __init__(
        self,
        name: str | None = None,
        weight_decay: float = 0,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        jit_compile: bool = True,
    ) -> None: ...
    @classmethod
    def from_config(cls: type[Self], config: dict[str, Any], custom_objects: dict[str, type] | None = None) -> Self: ...
    def get_config(self) -> dict[str, Any]: ...
    @abstractmethod
    def update_step(self, gradient: tf.Tensor | tf.IndexedSlices, variable: tf.Variable, /) -> None: ...
    def apply_gradients(
        self,
        grads_and_vars: Iterable[tuple[_GradientsT, tf.Variable]],
        skip_gradients_aggregation: bool = False,
    ) -> tf.Variable: ...
    @property
    def learning_rate(self) -> tf.Variable: ...
    def minimize(
        self,
        loss: tf.Tensor | Callable[[], tf.Tensor],
        var_list: Sequence[tf.Variable] | None = None,
        tape: tf.GradientTape | None = None,
    ) -> tf.Variable: ...
    @property
    def variables(self) -> Sequence[tf.Variable]: ...
    @abstractmethod
    def build(self, var_list: Sequence[tf.Variable]) -> None: ...
    @property
    def iterations(self) -> tf.Variable: ...
    @iterations.setter
    def iterations(self, value: tf.Variable): ...
    def _var_key(self, var: tf.Variable) -> str: ...
    def _build_learning_rate(self, learning_rate: _LearningRateT) -> _LearningRateT: ...
    def _create_iteration_variable(self) -> None: ...
    def _distributed_apply_gradients_fn(
        self,
        distribution: tf.distribute.Strategy,
        grads_and_vars: Iterable[tuple[_GradientsT, tf.Variable]],
        **kwargs: object,
    ) -> tf.Variable: ...

class Adam(Optimizer):
    amsgrad: bool
    beta_1: float
    beta_2: float
    epsilon: float
    _momentums: list[tf.Variable]
    _velocities: list[tf.Variable]
    def __init__(
        self,
        learning_rate: _LearningRateT = 0.001,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        epsilon: float = 1e-07,
        amsgrad: bool = False,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        jit_compile: bool = True,
        name: str = "Adam",
    ): ...
    def build(self, var_list: Sequence[tf.Variable]) -> None: ...
    def update_step(self, gradient: tf.Tensor | tf.IndexedSlices, variable: tf.Variable): ...

class SGD(Optimizer):
    momentum: float
    def __init__(
        self,
        learning_rate: _LearningRateT = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        jit_compile: bool = True,
        name: str = "SGD",
    ) -> None: ...
    def build(self, var_list: Sequence[tf.Variable]) -> None: ...
    def update_step(self, gradient: tf.Tensor | tf.IndexedSlices, variable: tf.Variable) -> None: ...

class Adagrad(Optimizer):
    _accumulators: list[tf.Variable]
    initial_accumulator_value: float
    epsilon: float
    def __init__(
        self,
        learning_rate: _LearningRateT = 0.001,
        initial_accumulator_value: float = 0.1,
        epsilon: float = 1e-07,
        weight_decay: float | None = None,
        clipnorm: float | None = None,
        clipvalue: float | None = None,
        global_clipnorm: float | None = None,
        use_ema: bool = False,
        ema_momentum: float = 0.99,
        ema_overwrite_frequency: int | None = None,
        jit_compile: bool = True,
        name: str = "Adagrad",
    ) -> None: ...
    def build(self, var_list: Sequence[tf.Variable]) -> None: ...
    def update_step(self, gradient: tf.Tensor | tf.IndexedSlices, variable: tf.Variable, /) -> None: ...

class Ftrl(Optimizer):
    def __init__(
        self,
        learning_rate: float = 0.001,
        learning_rate_power: float = -0.5,
        initial_accumulator_value: float = 0.1,
        l1_regularization_strength: float = 0.0,
        l2_regularization_strength: float = 0.0,
        name: str = "Ftrl",
        l2_shrinkage_regularization_strength: float = 0.0,
        beta: float = 0.0,
    ) -> None: ...
    def build(self, var_list: Sequence[tf.Variable]) -> None: ...
    def update_step(self, gradient: tf.Tensor | tf.IndexedSlices, variable: tf.Variable) -> None: ...
