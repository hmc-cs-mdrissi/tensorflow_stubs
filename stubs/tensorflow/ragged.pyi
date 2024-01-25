from typing import Any, Callable, Sequence

import tensorflow as tf
from tensorflow import RaggedTensor, ScalarTensorCompatible, TensorCompatible

def stack(
    values: Sequence[TensorCompatible | RaggedTensor], axis: ScalarTensorCompatible = 0, name: str | None = None
) -> RaggedTensor: ...
def constant(
    pylist: TensorCompatible,
    dtype: tf.DType | None = None,
    ragged_rank: int | None = None,
    inner_shape: tuple[int, ...] | None = None,
    name: str | None = None,
    row_splits_dtype: tf.DType = tf.dtypes.int64,
) -> RaggedTensor: ...
def map_flat_values(
    op: Callable[..., tf.Tensor], *args: tf.RaggedTensor, **kwargs: tf.RaggedTensor
) -> RaggedTensor: ...
def range(
    starts: TensorCompatible,
    limits: TensorCompatible | None = None,
    deltas: TensorCompatible = 1,
    dtype: tf.DType | None = None,
    name: str | None = None,
    row_splits_dtype: tf.DType = tf.dtypes.int64,
) -> RaggedTensor: ...
def __getattr__(name: str) -> Any: ...
