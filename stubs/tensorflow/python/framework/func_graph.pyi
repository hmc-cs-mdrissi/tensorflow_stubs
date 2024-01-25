from typing import Any, Callable

import tensorflow as tf
from tensorflow._aliases import ContainerGeneric, TensorCompatible

class FuncGraph(tf.Graph):
    name: str
    inputs: list[tf.Tensor]
    outputs: list[tf.Tensor]
    control_outputs: list[tf.Operation]
    structured_input_signature: tuple[tuple[Any, ...], dict[str, tf.TensorSpec]]
    seed: int | None

    def capture_call_time_value(
        self,
        closure: Callable[[], TensorCompatible],
        spec: ContainerGeneric[tf.TensorSpec],
        key: object = None,
    ) -> tf.Tensor: ...
