from collections.abc import Callable
from typing import Literal, overload

from tensorflow import (
    SparseTensorCompatible,
    Tensor,
    TensorCompatible,
    TensorLike,
    TensorShape,
)
from tensorflow._aliases import ShapeLike
from tensorflow.dtypes import DType

class SparseTensor:
    indices: Tensor
    values: Tensor
    dense_shape: Tensor
    shape: TensorShape
    dtype: DType
    name: str
    def __init__(
        self,
        indices: TensorCompatible,
        values: TensorCompatible,
        dense_shape: TensorCompatible,
    ) -> None: ...
    def get_shape(self) -> TensorShape: ...
    def with_values(self, new_values: Tensor) -> SparseTensor: ...
    # Many arithmetic operations are not directly supported. Some have alternatives like tf.sparse.add instead of +.
    def __div__(self, y: SparseTensorCompatible) -> SparseTensor: ...
    def __truediv__(self, y: SparseTensorCompatible) -> SparseTensor: ...
    def __mul__(self, y: SparseTensorCompatible) -> SparseTensor: ...
    def __rmul__(self, y: SparseTensorCompatible) -> SparseTensor: ...

def concat(
    axis: int,
    sp_inputs: list[SparseTensor],
    expand_nonconcat_dims: bool = False,
    name: str | None = None,
) -> SparseTensor: ...
def to_dense(
    sp_input: SparseTensor,
    default_value: TensorCompatible | None = None,
    validate_indices: bool = True,
    name: str | None = None,
) -> Tensor: ...
def from_dense(tensor: TensorCompatible, name: str | None = None) -> SparseTensor: ...
def slice(
    sp_input: SparseTensor,
    start: list[int] | tuple[int, ...] | Tensor,
    size: TensorCompatible,
    name: str | None = None,
) -> SparseTensor: ...
def fill_empty_rows(
    sp_input: SparseTensor, default_value: TensorCompatible, name: str | None = None
) -> tuple[SparseTensor, Tensor]: ...
def reset_shape(
    sp_input: SparseTensor,
    new_shape: ShapeLike | None = None,
) -> SparseTensor: ...
def to_indicator(
    sp_input: SparseTensor, vocab_size: int, name: str | None = None
) -> Tensor: ...
def map_values(
    op: Callable[..., Tensor],
    *args: TensorLike | TensorCompatible,
    **kwargs: TensorLike | TensorCompatible,
) -> SparseTensor: ...
def expand_dims(
    sp_input: SparseTensor, axis: int | None = None, name: str | None = None
) -> SparseTensor: ...
@overload
def reduce_sum(
    sp_input: SparseTensor,
    axis: int | None = None,
    keepdims: bool | None = None,
    output_is_sparse: Literal[False] = False,
    name: str | None = None,
) -> Tensor: ...
@overload
def reduce_sum(
    sp_input: SparseTensor,
    axis: int | None = None,
    keepdims: bool | None = None,
    *,
    output_is_sparse: Literal[True],
    name: str | None = None,
) -> SparseTensor: ...
def segment_sum(
    data: Tensor,
    indices: Tensor,
    segment_ids: Tensor,
    num_segments: int | None = None,
    name: str | None = None,
) -> Tensor: ...
def segment_mean(
    data: Tensor,
    indices: Tensor,
    segment_ids: Tensor,
    num_segments: int | None = None,
    name: str | None = None,
) -> Tensor: ...
def segment_sqrt_n(
    data: Tensor,
    indices: Tensor,
    segment_ids: Tensor,
    num_segments: int | None = None,
    name: str | None = None,
) -> Tensor: ...
def reshape(
    sp_input: SparseTensor, shape: ShapeLike, name: str | None = None
) -> SparseTensor: ...
def retain(sp_input: SparseTensor, to_retain: Tensor) -> SparseTensor: ...
