from typing import Iterable, overload

import tensorflow as tf
from tensorflow import (
    IndexedSlices,
    RaggedTensor,
    ShapeLike,
    Tensor,
    TensorCompatible,
    TensorCompatibleT,
)
from tensorflow._aliases import DTypeLike, RaggedTensorLikeT, SparseTensorCompatible
from tensorflow.sparse import SparseTensor

# The documentation for tf.equal is a lie. It claims to support sparse tensors, but crashes on them.
# Whether an operation supports sparse tensors is poorly documented and needs to be verified
# manually. Most operations do not support sparse tensors.
@overload
def abs(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def abs(x: SparseTensor, name: str | None = None) -> SparseTensor: ...
@overload
def abs(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def sin(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def sin(x: RaggedTensorLikeT, name: str | None = None) -> RaggedTensorLikeT: ...
@overload
def cos(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def cos(x: RaggedTensorLikeT, name: str | None = None) -> RaggedTensorLikeT: ...
@overload
def exp(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def exp(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def sinh(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def sinh(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def cosh(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def cosh(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def tanh(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def tanh(x: SparseTensor, name: str | None = None) -> SparseTensor: ...
@overload
def tanh(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
def expm1(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def log(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def log(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def log1p(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def log1p(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def negative(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def negative(x: SparseTensor, name: str | None = None) -> SparseTensor: ...
@overload
def negative(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
def sigmoid(x: TensorCompatible, name: str | None = None) -> Tensor: ...
def add(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def add_n(
    x: Iterable[TensorCompatible | IndexedSlices], name: str | None = None
) -> Tensor: ...
@overload
def add_n(x: Iterable[RaggedTensor], name: str | None = None) -> RaggedTensor: ...
@overload
def subtract(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def subtract(
    x: TensorCompatible | RaggedTensor, y: RaggedTensor, name: str | None = None
) -> RaggedTensor: ...
@overload
def subtract(
    x: TensorCompatible | RaggedTensor,
    y: TensorCompatible | RaggedTensor,
    name: str | None = None,
) -> Tensor | RaggedTensor: ...
@overload
def multiply(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def multiply(
    x: RaggedTensor, y: RaggedTensor, name: str | None = None
) -> RaggedTensor: ...
def multiply_no_nan(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
def divide(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
def divide_no_nan(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def floormod(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def floormod(
    x: RaggedTensor, y: RaggedTensor | TensorCompatible, name: str | None = None
) -> RaggedTensor: ...
@overload
def ceil(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def ceil(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def floor(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def floor(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
def accumulate_n(
    inputs: list[TensorCompatibleT] | tuple[TensorCompatibleT, ...],
    shape: ShapeLike | None = None,
) -> Tensor: ...
@overload
def pow(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def pow(x: RaggedTensor, y: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def reciprocal(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def reciprocal(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def is_nan(x: TensorCompatible | IndexedSlices, name: str | None = None) -> Tensor: ...
@overload
def is_nan(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def is_finite(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def is_finite(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def minimum(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def minimum(
    x: tf.RaggedTensor, y: TensorCompatible | tf.RaggedTensor, name: str | None = None
) -> tf.RaggedTensor: ...
@overload
def minimum(
    x: TensorCompatible | tf.RaggedTensor, y: tf.RaggedTensor, name: str | None = None
) -> tf.RaggedTensor: ...
@overload
def maximum(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def maximum(
    x: tf.RaggedTensor, y: TensorCompatible | tf.RaggedTensor, name: str | None = None
) -> tf.RaggedTensor: ...
@overload
def maximum(
    x: TensorCompatible | tf.RaggedTensor, y: tf.RaggedTensor, name: str | None = None
) -> tf.RaggedTensor: ...
@overload
def logical_not(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def logical_not(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def logical_and(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def logical_and(
    x: RaggedTensor, y: RaggedTensor, name: str | None = None
) -> RaggedTensor: ...
@overload
def logical_or(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def logical_or(
    x: tf.RaggedTensor, y: tf.RaggedTensor, name: str | None = None
) -> tf.RaggedTensor: ...
def logical_xor(
    x: TensorCompatible, y: TensorCompatible, name: str | None = "LogicalXor"
) -> Tensor: ...
@overload
def equal(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def equal(
    x: RaggedTensor, y: RaggedTensor | TensorCompatible, name: str | None = None
) -> RaggedTensor: ...
@overload
def equal(
    x: TensorCompatible | RaggedTensor, y: RaggedTensor, name: str | None = None
) -> RaggedTensor: ...
@overload
def not_equal(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def not_equal(
    x: RaggedTensor, y: RaggedTensor | float, name: str | None = None
) -> RaggedTensor: ...
@overload
def greater(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def greater(
    x: RaggedTensor, y: RaggedTensor | float, name: str | None = None
) -> RaggedTensor: ...
@overload
def greater_equal(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def greater_equal(
    x: RaggedTensor, y: RaggedTensor | float, name: str | None = None
) -> RaggedTensor: ...
@overload
def less(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def less(x: RaggedTensor, y: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def less(
    x: RaggedTensor, y: TensorCompatible, name: str | None = None
) -> RaggedTensor: ...
@overload
def less(
    x: TensorCompatible, y: RaggedTensor, name: str | None = None
) -> RaggedTensor: ...
@overload
def less_equal(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def less_equal(
    x: RaggedTensor, y: RaggedTensor, name: str | None = None
) -> RaggedTensor: ...
def round(x: TensorCompatible, name: str | None = None) -> Tensor: ...
def segment_sum(
    data: TensorCompatible, segment_ids: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def sign(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def sign(x: SparseTensor, name: str | None = None) -> SparseTensor: ...
@overload
def sign(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def sqrt(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def sqrt(x: SparseTensor, name: str | None = None) -> SparseTensor: ...
@overload
def sqrt(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
def rsqrt(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def square(x: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def square(x: SparseTensor, name: str | None = None) -> SparseTensor: ...
@overload
def square(x: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def squared_difference(
    x: TensorCompatible, y: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def squared_difference(
    x: TensorCompatible | RaggedTensor, y: RaggedTensor, name: str | None = None
) -> RaggedTensor: ...
@overload
def squared_difference(
    x: RaggedTensor, y: TensorCompatible | RaggedTensor, name: str | None = None
) -> RaggedTensor: ...
def softplus(features: TensorCompatible, name: str | None = None) -> Tensor: ...

# Depending on the method axis is either a rank 0 tensor or a rank 0/1 tensor.
def reduce_mean(
    input_tensor: TensorCompatible | RaggedTensor,
    axis: TensorCompatible | None = None,
    keepdims: bool = False,
    name: str | None = None,
) -> Tensor: ...
def reduce_sum(
    input_tensor: TensorCompatible | RaggedTensor,
    axis: TensorCompatible | None = None,
    keepdims: bool = False,
    name: str | None = None,
) -> Tensor: ...
def reduce_max(
    input_tensor: TensorCompatible | RaggedTensor,
    axis: TensorCompatible | None = None,
    keepdims: bool = False,
    name: str | None = None,
) -> Tensor: ...
def reduce_min(
    input_tensor: TensorCompatible | RaggedTensor,
    axis: TensorCompatible | None = None,
    keepdims: bool = False,
    name: str | None = None,
) -> Tensor: ...
def reduce_prod(
    input_tensor: TensorCompatible | RaggedTensor,
    axis: TensorCompatible | None = None,
    keepdims: bool = False,
    name: str | None = None,
) -> Tensor: ...
def reduce_std(
    input_tensor: TensorCompatible | RaggedTensor,
    axis: TensorCompatible | None = None,
    keepdims: bool = False,
    name: str | None = None,
) -> Tensor: ...
def argmax(
    input: TensorCompatible,
    axis: TensorCompatible | None = None,
    output_type: DTypeLike = tf.int32,
    name: str | None = None,
) -> Tensor: ...
def argmin(
    input: TensorCompatible,
    axis: TensorCompatible | None = None,
    output_type: DTypeLike = tf.int32,
    name: str | None = None,
) -> Tensor: ...

# Only for bool tensors.
def reduce_any(
    input_tensor: TensorCompatible | RaggedTensor,
    axis: TensorCompatible | None = None,
    keepdims: bool = False,
    name: str | None = None,
) -> Tensor: ...
def reduce_all(
    input_tensor: TensorCompatible | RaggedTensor,
    axis: TensorCompatible | None = None,
    keepdims: bool = False,
    name: str | None = None,
) -> Tensor: ...
def count_nonzero(
    input: SparseTensorCompatible,
    axis: TensorCompatible | None = None,
    keepdims: bool | None = None,
    dtype: DTypeLike = tf.dtypes.int64,
    name: str | None = None,
) -> Tensor: ...
def l2_normalize(
    x: TensorCompatible,
    axis: int | None = None,
    epsilon: float = 1e-12,
    name: str | None = None,
    dim: int | None = None,
) -> Tensor: ...
def unsorted_segment_sum(
    data: TensorCompatible,
    segment_ids: TensorCompatible,
    num_segments: TensorCompatible,
    name: str | None = None,
) -> Tensor: ...
def bincount(
    arr: TensorCompatible,
    weights: TensorCompatible | None = None,
    minlength: int | None = None,
    maxlength: int | None = None,
    dtype: DTypeLike = ...,
    name: str | None = None,
    axis: int | None = None,
    binary_output: bool = False,
) -> Tensor: ...
