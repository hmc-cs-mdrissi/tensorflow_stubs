from builtins import bool as _bool
from typing import Literal, overload

from tensorflow import RaggedTensor, ScalarTensorCompatible, Tensor, TensorCompatible
from tensorflow import norm as norm
from tensorflow._aliases import DTypeLike
from tensorflow.math import l2_normalize as l2_normalize

@overload
def matmul(
    a: TensorCompatible,
    b: TensorCompatible,
    transpose_a: _bool = False,
    transpose_b: _bool = False,
    adjoint_a: _bool = False,
    adjoint_b: _bool = False,
    a_is_sparse: _bool = False,
    b_is_sparse: _bool = False,
    output_type: DTypeLike | None = None,
    name: str | None = None,
) -> Tensor: ...
@overload
def matmul(
    a: RaggedTensor,
    b: RaggedTensor,
    transpose_a: _bool = False,
    transpose_b: _bool = False,
    adjoint_a: _bool = False,
    adjoint_b: _bool = False,
    a_is_sparse: _bool = False,
    b_is_sparse: _bool = False,
    output_type: DTypeLike | None = None,
    name: str | None = None,
) -> RaggedTensor: ...
def set_diag(
    input: TensorCompatible,
    diagonal: TensorCompatible,
    name: str | None = "set_diag",
    k: int = 0,
    align: Literal[
        "RIGHT_LEFT", "RIGHT_RIGHT", "LEFT_LEFT", "LEFT_RIGHT"
    ] = "RIGHT_LEFT",
) -> Tensor: ...
def eye(
    num_rows: ScalarTensorCompatible,
    num_columns: ScalarTensorCompatible | None = None,
    batch_shape: int | tuple[int, ...] | None = None,
    dtype: DTypeLike = ...,
    name: str | None = None,
) -> Tensor: ...
def band_part(
    input: TensorCompatible, num_lower: int, num_upper: int, name: str | None = None
) -> Tensor: ...
