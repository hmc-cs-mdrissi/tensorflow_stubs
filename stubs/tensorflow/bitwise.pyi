from typing import overload

import tensorflow as tf
from tensorflow._aliases import TensorCompatible

@overload
def bitwise_or(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> tf.Tensor: ...
@overload
def bitwise_or(
    x: tf.RaggedTensor, y: tf.RaggedTensor, name: str | None = None
) -> tf.RaggedTensor: ...
@overload
def bitwise_and(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> tf.Tensor: ...
@overload
def bitwise_and(
    x: tf.RaggedTensor, y: tf.RaggedTensor, name: str | None = None
) -> tf.RaggedTensor: ...
