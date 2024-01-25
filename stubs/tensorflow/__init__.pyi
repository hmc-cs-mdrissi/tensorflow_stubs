from builtins import bool as _bool
from builtins import slice as _slice
from contextlib import contextmanager
from enum import Enum
from logging import Logger
from types import TracebackType
from typing import (Any, Callable, ContextManager, Generator, Iterable,
                    Iterator, Mapping, NoReturn, Protocol, Sequence, TextIO,
                    TypedDict, TypeVar, overload)

import numpy as np
import tensorflow as tf
from tensorflow import autograph as autograph
from tensorflow import bitwise as bitwise
from tensorflow import compat as compat
from tensorflow import config as config
from tensorflow import data as data
from tensorflow import debugging as debugging
from tensorflow import distribute as distribute
from tensorflow import dtypes as dtypes
from tensorflow import errors as errors
from tensorflow import estimator as estimator
from tensorflow import experimental as experimental
from tensorflow import feature_column as feature_column
from tensorflow import image as image
from tensorflow import initializers as initializers
from tensorflow import io as io
from tensorflow import keras as keras
from tensorflow import linalg as linalg
from tensorflow import lookup as lookup
from tensorflow import losses as losses
from tensorflow import math as math
from tensorflow import metrics as metrics
from tensorflow import nn as nn
from tensorflow import optimizers as optimizers
from tensorflow import profiler as profiler
from tensorflow import ragged as ragged
from tensorflow import random as random
from tensorflow import raw_ops as raw_ops
from tensorflow import saved_model as saved_model
from tensorflow import sparse as sparse
from tensorflow import strings as strings
from tensorflow import summary as summary
from tensorflow import sysconfig as sysconfig
from tensorflow import test as test
from tensorflow import tpu as tpu
from tensorflow import train as train
from tensorflow import types as types
from tensorflow._aliases import *
from tensorflow.compat.v1 import NodeDef, _FeedDict  # type: ignore
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.dtypes import *
from tensorflow.graph_util import import_graph_def as import_graph_def
from tensorflow.keras.constraints import Constraint
from tensorflow.keras.initializers import Zeros
from tensorflow.linalg import eye as eye
from tensorflow.linalg import matmul as matmul
# Most tf.math functions are exported as tf, but sadly not all are.
from tensorflow.math import abs as abs
from tensorflow.math import add as add
from tensorflow.math import add_n as add_n
from tensorflow.math import argmax as argmax
from tensorflow.math import argmin as argmin
from tensorflow.math import cos as cos
from tensorflow.math import cosh as cosh
from tensorflow.math import divide as divide
from tensorflow.math import equal as equal
from tensorflow.math import exp as exp
from tensorflow.math import greater as greater
from tensorflow.math import greater_equal as greater_equal
from tensorflow.math import less as less
from tensorflow.math import less_equal as less_equal
from tensorflow.math import logical_and as logical_and
from tensorflow.math import logical_not as logical_not
from tensorflow.math import logical_or as logical_or
from tensorflow.math import maximum as maximum
from tensorflow.math import minimum as minimum
from tensorflow.math import multiply as multiply
from tensorflow.math import not_equal as not_equal
from tensorflow.math import pow as pow
from tensorflow.math import reduce_all as reduce_all
from tensorflow.math import reduce_any as reduce_any
from tensorflow.math import reduce_max as reduce_max
from tensorflow.math import reduce_mean as reduce_mean
from tensorflow.math import reduce_min as reduce_min
from tensorflow.math import reduce_prod as reduce_prod
from tensorflow.math import reduce_sum as reduce_sum
from tensorflow.math import round as round
from tensorflow.math import sigmoid as sigmoid
from tensorflow.math import sign as sign
from tensorflow.math import sin as sin
from tensorflow.math import sinh as sinh
from tensorflow.math import sqrt as sqrt
from tensorflow.math import square as square
from tensorflow.math import subtract as subtract
from tensorflow.math import tanh as tanh
from tensorflow.python.distribute.sharded_variable import ShardedVariable
from tensorflow.python.training.tracking.autotrackable import AutoTrackable
from tensorflow.sparse import SparseTensor as SparseTensor
from tensorflow.strings import as_string as as_string
from typing_extensions import (Literal, ParamSpec, Self, TypeAlias, TypeGuard,
                               Unpack)

# These types are written based on usage. If type annotation is inconsistent with runtime feel free to improve it.
# Some of these classes may be missing methods. You can use def __getattr__(self, name: str) -> Any: ...
# when you know it is incomplete with major methods missing. If most common methods are present it is fine to
# leave it out and then fill in more when errors are encountered.

# tf.compat.v1 api will mostly not be type stubbed because goal is to remove that from codebase entirely. If there
# is functionality with no tf2 equivalents we may include them in the stubs. But stuff like tf.Session/placeholder should be
# strongly avoided.

# Tensors ideally should be a generic type, but properly typing data type/shape
# will be a lot of work. Until we have good non-generic tensorflow stubs,
# we will skip making Tensor generic. Also good type hints for shapes will
# run quickly into many places where type system is not strong enough today.
# So shape typing is probably not worth doing anytime soon.
_SliceT: TypeAlias = int | _slice | None | type(Ellipsis) # type: ignore

_R = TypeVar("_R")
_P = ParamSpec("_P")

_ShapeLike = ShapeLike

class _KerasSerializable1(Protocol):
    def get_config(self) -> dict[str, Any]: ...

class _KerasSerializable2(Protocol):
    __name__: str

_KerasSerializable: TypeAlias = _KerasSerializable1 | _KerasSerializable2

__internal__: Any
__version__: str

class Tensor:
    def consumers(self) -> list[Operation]: ...
    @property
    def shape(self) -> TensorShape: ...
    def get_shape(self) -> TensorShape: ...
    def set_shape(self, shape: TensorShape | Sequence[int | None]) -> None: ...
    @property
    def dtype(self) -> DType: ...
    @property
    def graph(self) -> Graph: ...
    @property
    def name(self) -> str: ...
    @property
    def op(self) -> Operation: ...
    def numpy(self) -> np.ndarray[Any, Any]: ...
    def eval(
        self,
        feed_dict: _FeedDict | None = None,
        session: tf.compat.v1.Session | None = None,
    ) -> np.ndarray[Any, Any]: ...
    def __int__(self) -> int: ...
    def __abs__(self) -> Tensor: ...
    def __add__(self, other: TensorCompatible) -> Tensor: ...
    def __radd__(self, other: TensorCompatible) -> Tensor: ...
    def __sub__(self, other: TensorCompatible) -> Tensor: ...
    def __rsub__(self, other: TensorCompatible) -> Tensor: ...
    def __mul__(self, other: TensorCompatible) -> Tensor: ...
    def __rmul__(self, other: TensorCompatible) -> Tensor: ...
    def __mod__(self, other: TensorCompatible) -> Tensor: ...
    def __rmod__(self, other: TensorCompatible) -> Tensor: ...
    def __pow__(self, other: TensorCompatible) -> Tensor: ...
    def __matmul__(self, other: TensorCompatible) -> Tensor: ...
    def __rmatmul__(self, other: TensorCompatible) -> Tensor: ...
    def __floordiv__(self, other: TensorCompatible) -> Tensor: ...
    def __rfloordiv__(self, other: TensorCompatible) -> Tensor: ...
    def __truediv__(self, other: TensorCompatible) -> Tensor: ...
    def __rtruediv__(self, other: TensorCompatible) -> Tensor: ...
    def __neg__(self) -> Tensor: ...
    def __and__(self, other: TensorCompatible) -> Tensor: ...
    def __rand__(self, other: TensorCompatible) -> Tensor: ...
    def __or__(self, other: TensorCompatible) -> Tensor: ...
    def __ror__(self, other: TensorCompatible) -> Tensor: ...
    def __eq__(self, other: TensorCompatible) -> Tensor: ...  # type: ignore
    def __ne__(self, other: TensorCompatible) -> Tensor: ...  # type: ignore
    def __ge__(self, other: TensorCompatible) -> Tensor: ...
    def __gt__(self, other: TensorCompatible) -> Tensor: ...
    def __le__(self, other: TensorCompatible) -> Tensor: ...
    def __lt__(self, other: TensorCompatible) -> Tensor: ...
    def __invert__(self) -> Tensor: ...
    def __bool__(self) -> NoReturn: ...
    def __getitem__(self, slice_spec: _SliceT | tuple[_SliceT, ...]) -> Tensor: ...
    def __len__(self) -> int: ...
    # This only works for rank 0 tensors.
    def __index__(self) -> int: ...
    # Only works in tf1 mode
    def __hash__(self) -> int: ...

# This is mostly a white lie. Variable is not a real subclass, but it behaves very similar to one.
# Most functions/operations on tensors also work on variables. isinstance is main difference.
class Variable(Tensor):
    def __init__(
        self,
        initial_value: TensorCompatible | Callable[[], TensorCompatible],
        trainable: None | _bool = None,
        validate_shape: _bool = True,
        name: str | None = None,
        dtype: DTypeLike | None = None,
        # constraint should be used rarely. It's incompatible with asynchronous training.
        constraint: Callable[[Tensor], Tensor] | None = None,
        synchronization: VariableSynchronization = VariableSynchronization.AUTO,
        aggregation: VariableAggregation = VariableAggregation.NONE,
    ): ...
    def value(self) -> Tensor: ...
    @overload
    def assign(
        self,
        value: TensorCompatible,
        use_locking: _bool = False,
        name: str | None = None,
        read_value: Literal[True] = True,
    ) -> Self: ...
    @overload
    def assign(
        self,
        value: TensorCompatible,
        use_locking: _bool = False,
        name: str | None = None,
        read_value: Literal[False] = False,
    ) -> Operation | None: ...
    @overload
    def assign_add(
        self,
        delta: TensorCompatible,
        use_locking: _bool = False,
        name: str | None = None,
        read_value: Literal[True] = True,
    ) -> Self: ...
    @overload
    def assign_add(
        self,
        delta: TensorCompatible,
        use_locking: _bool = False,
        name: str | None = None,
        read_value: Literal[False] = False,
    ) -> Operation | None: ...
    @overload
    def assign_sub(
        self,
        delta: TensorCompatible,
        use_locking: _bool = False,
        name: str | None = None,
        read_value: Literal[True] = True,
    ) -> Operation | None: ...
    @overload
    def assign_sub(
        self,
        delta: TensorCompatible,
        use_locking: _bool = False,
        name: str | None = None,
        read_value: Literal[False] = False,
    ) -> Operation | None: ...
    def read_value(self) -> Tensor: ...
    def scatter_add(
        self,
        sparse_delta: IndexedSlices,
        use_locking: _bool = False,
        name: str | None = None,
    ) -> Self: ...
    def scatter_sub(
        self,
        sparse_delta: IndexedSlices,
        use_locking: _bool = False,
        name: str | None = None,
    ) -> Self: ...
    def scatter_update(
        self,
        sparse_delta: IndexedSlices,
        use_locking: _bool = False,
        name: str | None = None,
    ) -> Self: ...
    # This actually belongs to BaseResourceVariable, which most variables are. BaseResourceVariable
    # is internal though.
    @property
    def handle(self) -> Tensor: ...

# Most type annotations currently ignore ragged tensors due to rarity.
class RaggedTensor:
    def bounding_shape(
        self,
        axis: TensorCompatible | None = None,
        name: str | None = None,
        out_type: DTypeLike | None = None,
    ) -> Tensor: ...
    @classmethod
    def from_sparse(
        cls,
        st_input: SparseTensor,
        name: str | None = None,
        row_splits_dtype: DTypeLike = tf.int64,
    ) -> RaggedTensor: ...
    @classmethod
    def from_tensor(
        cls,
        tensor: TensorCompatible,
        lengths: Tensor | None = None,
        padding: Tensor | None = None,
        ragged_rank: int = 1,
        name: str | None = None,
        row_splits_dtype: DTypeLike = tf.int64,
    ) -> RaggedTensor: ...
    def to_sparse(self, name: str | None = None) -> SparseTensor: ...
    def to_tensor(
        self,
        default_value: float | str | None = None,
        name: str | None = None,
        shape: ShapeLike | None = None,
    ) -> Tensor: ...
    def to_list(self) -> list[list[Any]]: ...
    @classmethod
    def from_row_splits(
        cls,
        values: tf.Tensor | tf.RaggedTensor,
        row_splits: tf.Tensor,
        name: str | None = None,
        validate: _bool = True,
    ) -> RaggedTensor: ...
    @classmethod
    def from_row_lengths(
        cls,
        values: tf.Tensor | tf.RaggedTensor,
        row_lengths: tf.Tensor,
        name: str | None = None,
        validate: _bool = True,
    ) -> RaggedTensor: ...
    @property
    def ragged_rank(self) -> int: ...
    @property
    def row_splits(self) -> Tensor: ...
    @property
    def values(self) -> Tensor | RaggedTensor: ...
    @property
    def flat_values(self) -> Tensor: ...
    @property
    def shape(self) -> TensorShape: ...
    @property
    def dtype(self) -> DType: ...
    def merge_dims(self, outer_axis: int, inner_axis: int) -> RaggedTensor: ...
    def nrows(
        self, out_type: DType | None = None, name: str | None = None
    ) -> Tensor: ...
    def row_lengths(self, axis: int = 1, name: str | None = None) -> Tensor: ...
    def value_rowids(self, name: str | None = None) -> Tensor: ...
    def get_shape(self) -> TensorShape: ...
    def with_values(self, new_values: Tensor | tf.RaggedTensor) -> tf.RaggedTensor: ...
    def with_flat_values(self, new_values: Tensor) -> tf.RaggedTensor: ...
    def __add__(self, other: RaggedTensor | float) -> RaggedTensor: ...
    def __radd__(self, other: RaggedTensor | float) -> RaggedTensor: ...
    def __sub__(self, other: RaggedTensor | float) -> RaggedTensor: ...
    def __mul__(self, other: RaggedTensor | float) -> RaggedTensor: ...
    def __rmul__(self, other: RaggedTensor | float) -> RaggedTensor: ...
    def __floordiv__(self, other: RaggedTensor | float) -> RaggedTensor: ...
    def __truediv__(self, other: RaggedTensor | float) -> RaggedTensor: ...
    def __mod__(self, other: RaggedTensor | int) -> RaggedTensor: ...
    def __getitem__(self, slice_spec: _SliceT | tuple[_SliceT, ...]) -> Tensor: ...

class VariableSynchronization(Enum):
    AUTO = 0
    NONE = 1
    ON_WRITE = 2
    ON_READ = 3

class VariableAggregation(Enum):
    NONE = 0
    SUM = 1
    MEAN = 2
    ONLY_FIRST_REPLICA = 3

class AggregationMethod:
    ADD_N = 0
    DEFAULT = 0
    EXPERIMENTAL_TREE = 1
    EXPERIMENTAL_ACCUMULATE_N = 2

class TensorShape:
    def __init__(
        self, dims: ShapeLike | tensor_shape_pb2.TensorShapeProto | None
    ) -> None: ...
    @property
    def rank(self) -> int: ...
    @property
    def ndims(self) -> int: ...
    def as_list(self) -> list[int | None]: ...
    def as_proto(self) -> tensor_shape_pb2.TensorShapeProto: ...
    def assert_has_rank(self, rank: int) -> None: ...
    def assert_is_compatible_with(self, other: Iterable[int | None]) -> None: ...
    def concatenate(self, other: TensorShape) -> TensorShape: ...
    def __bool__(self) -> _bool: ...
    @overload
    def __getitem__(self, key: int) -> int | None: ...
    @overload
    def __getitem__(self, key: _slice) -> TensorShape: ...
    def __iter__(self) -> Iterator[int | None]: ...
    def __len__(self) -> int: ...
    def __add__(self, other: Iterable[int | None]) -> TensorShape: ...
    def __radd__(self, other: Iterable[int | None]) -> TensorShape: ...
    def __eq__(self, other: Iterable[int | None]) -> _bool: ...  # type: ignore

class TypeSpec:
    def is_compatible_with(
        self, spec_or_value: TypeSpec | TensorCompatible | SparseTensor | RaggedTensor
    ) -> _bool: ...
    def most_specific_compatible_type(self, other: TypeSpec) -> TypeSpec: ...

class TensorSpec(TypeSpec):
    def __init__(
        self, shape: ShapeLike, dtype: DTypeLike = float32, name: str | None = None
    ) -> None: ...
    @property
    def shape(self) -> TensorShape: ...
    @property
    def dtype(self) -> DType: ...
    @property
    def name(self) -> str | None: ...
    @classmethod
    def from_spec(cls, spec: TypeSpec, name: str | None = None) -> Self: ...
    @classmethod
    def from_tensor(cls, tensor: Tensor, name: str | None = None) -> Self: ...

class SparseTensorSpec(TypeSpec):
    def __init__(
        self, shape: ShapeLike | None = None, dtype: DTypeLike = float32
    ) -> None: ...
    @property
    def shape(self) -> TensorShape: ...
    @property
    def dtype(self) -> DType: ...
    @classmethod
    def from_value(cls, sparse_tensor: SparseTensor) -> Self: ...

class RaggedTensorSpec(TypeSpec):
    def __init__(
        self,
        shape: ShapeLike | None = None,
        dtype: DTypeLike = tf.dtypes.float32,
        ragged_rank: int | None = None,
        row_splits_dtype: DTypeLike = tf.dtypes.int64,
        flat_values_spec: TypeSpec | None = None,
    ): ...
    @property
    def shape(self) -> TensorShape: ...
    @property
    def dtype(self) -> DType: ...
    @property
    def ragged_rank(self) -> int: ...
    @property
    def row_splits_dtype(self) -> DType: ...
    @property
    def flat_values_spec(self) -> TypeSpec: ...
    @classmethod
    def from_value(cls, ragged_tensor: RaggedTensor) -> Self: ...

class IndexedSlices:
    def __init__(
        self, values: Tensor, indices: Tensor, dense_shape: None | Tensor = None
    ): ...
    @property
    def values(self) -> Tensor: ...
    @property
    def indices(self) -> Tensor: ...
    @property
    def dense_shape(self) -> None | Tensor: ...
    @property
    def shape(self) -> TensorShape: ...
    @property
    def dtype(self) -> DType: ...
    def __getattr__(self, name: str) -> Any: ...

class GradientTape:
    def __init__(
        self, persistent: _bool = False, watch_accessed_variables: _bool = True
    ): ...
    def __enter__(self) -> GradientTape: ...
    def __exit__(
        self,
        typ: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...
    def watch(self, tensor: ContainerTensorsLike) -> None: ...
    def watched_variables(self) -> tuple[Variable, ...]: ...
    # Higher kinded types would be nice here and these overloads are a way to simulate some of them.
    @overload
    def gradient(
        self,
        target: ContainerTensors,
        sources: TensorLike,
        output_gradients: list[Tensor] | None = None,
    ) -> GradientsT: ...
    @overload
    def gradient(
        self,
        target: ContainerTensors,
        sources: Sequence[Tensor],
        output_gradients: list[Tensor] | None = None,
    ) -> list[GradientsT]: ...
    @overload
    def gradient(
        self,
        target: ContainerTensors,
        sources: Mapping[str, Tensor],
        output_gradients: list[Tensor] | None = None,
    ) -> dict[str, GradientsT]: ...
    @overload
    def gradient(
        self,
        target: ContainerTensors,
        sources: ContainerTensors,
        output_gradients: list[Tensor] | None = None,
    ) -> ContainerGradients: ...

_ClipT = TypeVar("_ClipT", bound=Tensor | IndexedSlices | None)

def clip_by_global_norm(
    t_list: Sequence[_ClipT],
    clip_norm: ScalarTensorCompatible,
    use_norm: ScalarTensorCompatible | None = None,
    name: str | None = None,
) -> tuple[list[_ClipT], Tensor]: ...
def dynamic_partition(
    data: TensorCompatible,
    partitions: TensorCompatible,
    num_partitions: int,
    name: str | None = None,
) -> list[Tensor]: ...
def executing_eagerly() -> _bool: ...
def fingerprint(
    data: TensorCompatible,
    method: Literal["farmhash64"] = "farmhash64",
    name: str | None = None,
) -> Tensor: ...
@overload
def identity(input: TensorCompatible, name: str | None = None) -> Tensor: ...
@overload
def identity(input: SparseTensor, name: str | None = None) -> SparseTensor: ...
@overload
def identity(input: RaggedTensor, name: str | None = None) -> RaggedTensor: ...
@overload
def identity(
    input: SparseTensor | TensorCompatible, name: str | None = None
) -> SparseTensor | Tensor: ...
def clip_by_value(
    x: RaggedTensorLikeT,
    clip_value_min: float | Tensor,
    clip_value_max: float | Tensor,
    name: str | None = None,
) -> RaggedTensorLikeT: ...
@overload
def reverse(
    tensor: TensorCompatible, axis: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def reverse(
    tensor: RaggedTensor, axis: TensorCompatible, name: str | None = None
) -> RaggedTensor: ...
def searchsorted(
    sorted_sequence: TensorCompatible,
    values: Tensor,
    side: Literal["left", "right"] = "left",
    name: str | None = None,
) -> Tensor: ...
def sort(
    values: TensorCompatible,
    axis: int = -1,
    direction: Literal["ASCENDING", "DESCENDING"] = "ASCENDING",
    name: str | None = None,
) -> Tensor: ...
def constant(
    value: TensorCompatible,
    dtype: DTypeLike | None = None,
    shape: ShapeLike | None = None,
    name: str | None = None,
) -> Tensor: ...
def convert_to_tensor(
    value: TensorCompatible | IndexedSlices,
    dtype: DTypeLike | None = None,
    dtype_hint: DTypeLike | None = None,
    name: str | None = None,
) -> Tensor: ...
def zeros(
    shape: ShapeLike, dtype: DTypeLike = dtypes.float32, name: str | None = None
) -> Tensor: ...
def ones(
    shape: ShapeLike, dtype: DTypeLike = dtypes.float32, name: str | None = None
) -> Tensor: ...
@overload
def zeros_like(
    input: TensorCompatible | IndexedSlices,
    dtype: DTypeLike | None = None,
    name: str | None = None,
) -> Tensor: ...
@overload
def zeros_like(
    input: RaggedTensor, dtype: DTypeLike | None = None, name: str | None = None
) -> RaggedTensor: ...
@overload
def ones_like(
    input: TensorCompatible, dtype: DTypeLike | None = None, name: str | None = None
) -> Tensor: ...
@overload
def ones_like(
    input: RaggedTensor, dtype: DTypeLike | None = None, name: str | None = None
) -> RaggedTensor: ...
def range(
    start: TensorCompatible,
    limit: TensorCompatible | None = None,
    delta: TensorCompatible = 1,
    dtype: DTypeLike | None = None,
    name: str | None = None,
) -> Tensor: ...
@overload
def concat(
    values: Sequence[TensorCompatible] | TensorCompatible,
    axis: int,
    name: str | None = None,
) -> Tensor: ...
@overload
def concat(
    values: Sequence[RaggedTensor | TensorCompatible] | RaggedTensor,
    axis: int,
    name: str | None = None,
) -> RaggedTensor: ...
def reshape(
    tensor: TensorCompatible, shape: ShapeLike | Tensor, name: str | None = None
) -> Tensor: ...
@overload
def cast(x: TensorCompatible, dtype: DTypeLike, name: str | None = None) -> Tensor: ...
@overload
def cast(
    x: SparseTensor, dtype: DTypeLike, name: str | None = None
) -> SparseTensor: ...
@overload
def cast(
    x: RaggedTensor, dtype: DTypeLike, name: str | None = None
) -> RaggedTensor: ...
@overload
def where(
    condition: TensorCompatible | RaggedTensor,
    x: None = None,
    y: None = None,
    name: str | None = None,
) -> Tensor: ...
@overload
def where(
    condition: TensorCompatible,
    x: TensorCompatible | IndexedSlices,
    y: TensorCompatible | IndexedSlices,
    name: str | None = None,
) -> Tensor: ...
@overload
def where(
    condition: RaggedTensor,
    x: RaggedTensor | float,
    y: RaggedTensor,
    name: str | None = None,
) -> RaggedTensor: ...
def meshgrid(
    *tensors: TensorCompatible,
    indexing: Literal["xy", "ij"] = "xy",
    name: str | None = None,
) -> tuple[Tensor, ...]: ...
def shape(
    input: SparseTensorCompatible, out_type: DTypeLike = int32, name: str | None = None
) -> Tensor: ...
def broadcast_to(
    tensor: TensorCompatible,
    shape: TensorCompatible | TensorShape,
    name: str | None = None,
) -> Tensor: ...
def split(
    value: TensorCompatible,
    num_or_size_splits: TensorCompatible,
    axis: int = 0,
    num: int | None = None,
    name: str | None = None,
) -> list[Tensor]: ...
def stack(
    values: Sequence[TensorCompatible], axis: int = 0, name: str | None = "stack"
) -> Tensor: ...
@overload
def tile(
    input: TensorCompatible, multiples: TensorCompatible, name: str | None = None
) -> Tensor: ...
@overload
def tile(
    input: RaggedTensor, multiples: TensorCompatible, name: str | None = None
) -> RaggedTensor: ...
def fill(
    dims: TensorCompatible, value: str | float | Tensor, name: str | None = None
) -> Tensor: ...
@overload
def gather(
    params: TensorCompatible,
    indices: TensorCompatible,
    axis: ScalarTensorCompatible | None = None,
    batch_dims: int = 0,
    name: str | None = None,
) -> Tensor: ...
@overload
def gather(
    params: RaggedTensor,
    indices: TensorCompatible,
    axis: ScalarTensorCompatible | None = None,
    batch_dims: int = 0,
    name: str | None = None,
) -> RaggedTensor: ...
@overload
def gather_nd(
    params: TensorCompatible,
    indices: TensorCompatible,
    batch_dims: int = 0,
    name: str | None = None,
) -> Tensor: ...
@overload
def gather_nd(
    params: RaggedTensor,
    indices: TensorCompatible | RaggedTensor,
    batch_dims: int = 0,
    name: str | None = None,
) -> RaggedTensor: ...
@overload
def expand_dims(
    input: TensorCompatible, axis: int, name: str | None = None
) -> Tensor: ...
@overload
def expand_dims(
    input: RaggedTensor, axis: int, name: str | None = None
) -> RaggedTensor: ...
@overload
def squeeze(
    input: TensorCompatible,
    axis: int | tuple[int, ...] | list[int] | None = None,
    name: str | None = None,
) -> Tensor: ...
@overload
def squeeze(
    input: RaggedTensor,
    axis: int | tuple[int, ...] | list[int],
    name: str | None = None,
) -> RaggedTensor: ...
def zeros_initializer() -> Zeros: ...
def pad(
    tensor: TensorCompatible,
    paddings: TensorCompatible,
    mode: Literal[
        "constant", "CONSTANT", "reflect", "REFLECT", "symmetric", "SYMMETRIC"
    ] = "CONSTANT",
    constant_values: float | str = 0,
    name: str | None = None,
) -> Tensor: ...
def transpose(
    a: TensorCompatible,
    perm: TensorCompatible | None = None,
    conjugate: _bool = False,
    name: str | None = None,
) -> Tensor: ...
def tensordot(
    a: TensorCompatible,
    b: TensorCompatible,
    axes: TensorCompatible,
    name: str | None = None,
) -> Tensor: ...
def bitcast(
    input: TensorCompatible, dtype: DTypeLike, name: str | None = None
) -> Tensor: ...
def repeat(
    input: TensorCompatible,
    repeats: TensorCompatible,
    axis: int | None = None,
    name: str | None = None,
) -> Tensor: ...
def broadcast_static_shape(
    shape_x: TensorShape, shape_y: TensorShape
) -> TensorShape: ...
@overload
def one_hot(
    indices: TensorCompatible,
    depth: ScalarTensorCompatible,
    on_value: float = 1,
    off_value: float = 0,
    axis: int = -1,
    dtype: DTypeLike | None = None,
    name: str | None = None,
) -> Tensor: ...
@overload
def one_hot(
    indices: RaggedTensor,
    depth: ScalarTensorCompatible,
    on_value: float = 1,
    off_value: float = 0,
    axis: int = -1,
    dtype: DTypeLike | None = None,
    name: str | None = None,
) -> RaggedTensor: ...
def group(
    *args: ContainerGeneric[Tensor | Operation | None], name: str | None = None
) -> Operation: ...

class _LoggerFn(Protocol):
    def __call__(self, msg: object, *args: object, **kwargs: object) -> object: ...

def print(
    *args: object,
    summarize: int = 3,
    name: str | None = None,
    sep: str = " ",
    end: str = "\n",
    output_stream: str | TextIO | _LoggerFn = ...,
) -> Operation: ...
def unstack(
    value: TensorCompatible,
    num: int | None = None,
    axis: int = 0,
    name: str = "unstack",
) -> list[tf.Tensor]: ...
def unique(
    x: TensorCompatible, out_idx: DType = ..., name: str | None = None
) -> tuple[tf.Tensor, tf.Tensor]: ...
def unique_with_counts(
    x: TensorCompatible, out_idx: DType = ..., name: str | None = None
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]: ...

class name_scope:
    def __init__(self, name: str): ...
    def __enter__(self) -> str: ...
    def __exit__(
        self,
        typ: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

@contextmanager
def init_scope() -> Iterator[None]: ...

class Graph:
    def add_to_collection(self, name: str, value: object): ...
    def add_to_collections(self, names: Iterable[str] | str, value: object): ...
    def get_collection(self, name: str, scope: str | None = None) -> list[object]: ...
    @contextmanager
    def as_default(self) -> Iterator[Self]: ...
    def finalize(self) -> None: ...
    def get_tensor_by_name(self, name: str) -> Tensor: ...
    def get_operation_by_name(self, name: str) -> Operation: ...
    def get_operations(self) -> list[Operation]: ...
    def as_graph_def(
        self, from_version: int | None = None, add_shapes: _bool = False
    ) -> compat.v1.GraphDef: ...
    def get_name_scope(self) -> str: ...

class Operation:
    @property
    def inputs(self) -> list[Tensor]: ...
    @property
    def input_types(self) -> list[DType]: ...
    @property
    def outputs(self) -> list[Tensor]: ...
    @property
    def output_types(self) -> list[DType]: ...
    @property
    def device(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def node_def(self) -> NodeDef: ...
    @property
    def type(self) -> str: ...
    def get_attr(self, name: str) -> Any: ...
    def __getitem__(
        self, slice_spec: int | _slice | tuple[int | _slice, ...]
    ) -> Operation: ...
    def _set_attr(self, attr_name: str, attr_value: tf.compat.v1.AttrValue) -> None: ...
    def _set_shape_list_attr(
        self, attr_name: str, shapes: list[TensorShape]
    ) -> None: ...
    def mark_used(self) -> None: ...

class Module(AutoTrackable):
    def __init__(self, name: str | None) -> None: ...
    @property
    def name(self) -> str: ...
    @property
    def name_scope(self) -> tf.name_scope: ...
    @property
    def variables(self) -> Sequence[Variable]: ...
    @property
    def trainable_variables(self) -> Sequence[Variable]: ...
    @property
    def non_trainable_variables(self) -> Sequence[Variable]: ...
    @property
    def submodules(self) -> Sequence[Module]: ...

_AccumT = TypeVar("_AccumT", bound=Tensor | Sequence[Tensor])
_ElemT = TypeVar("_ElemT", bound=Tensor | Sequence[Tensor])

def foldl(
    fn: Callable[[_AccumT, _ElemT], _AccumT],
    elems: _ElemT,
    initializer: _AccumT | None = None,
    parallel_iterations: int = 10,
    back_prop: _bool = True,
    swap_memory: _bool = False,
    name: str | None = None,
) -> _AccumT: ...
@overload
def function(
    func: None = None,
    input_signature: ContainerGeneric[TypeSpec] | None = None,
    autograph: _bool = True,
    jit_compile: _bool | None = None,
    reduce_retracing: _bool = False,
    experimental_implements: str | None = None,
    experimental_autograph_options: tuple[tf.autograph.experimental.Feature, ...]
    | None = None,
    experimental_follow_type_hints: _bool | None = None,
) -> Callable[[Callable[_P, _R]], tf.types.experimental.GenericFunction[_P, _R]]: ...
@overload
def function(
    func: Callable[_P, _R],
    input_signature: ContainerGeneric[TypeSpec] | None = None,
    autograph: _bool = True,
    jit_compile: _bool | None = None,
    reduce_retracing: _bool = False,
    experimental_implements: str | None = None,
    experimental_autograph_options: tuple[tf.autograph.experimental.Feature, ...]
    | None = None,
    experimental_follow_type_hints: _bool | None = None,
) -> tf.types.experimental.GenericFunction[_P, _R]: ...
def gradients(
    ys: Tensor | Sequence[Tensor],
    xs: Tensor | Sequence[Tensor],
    grad_ys: Tensor | Sequence[Tensor] | None = None,
    name: str = "gradients",
    gate_gradients: _bool = False,
    aggregation_method: Literal[0, 1, 2] | None = None,
    stop_gradients: Tensor | Sequence[Tensor] | None = None,
    unconnected_gradients: UnconnectedGradients
    | Literal["none", "zero"] = tf.UnconnectedGradients.NONE,
) -> list[Tensor | IndexedSlices]: ...
def py_function(
    func: Callable[..., object],
    inp: Sequence[TensorLike] | TensorLike,
    Tout: TypeSpec | Sequence[tf.DType] | tf.DType,
    name: str | None = None,
) -> Any: ...
def get_logger() -> Logger: ...
def type_spec_from_value(
    value: TensorCompatible | SparseTensor | RaggedTensor,
) -> TypeSpec: ...
def control_dependencies(
    control_inputs: Iterable[Operation | TensorLike] | None,
) -> ContextManager[None]: ...
def is_tensor(x: object) -> TypeGuard[Tensor]: ...
def timestamp(name: str | None = None) -> Tensor: ...
def linspace(
    start: TensorCompatible,
    stop: TensorCompatible,
    num: int,
    name: str | None = None,
    axis: int = 0,
) -> Tensor: ...
def make_tensor_proto(
    values: TensorCompatible,
    dtype: DTypeLike | None = None,
    shape: TensorShape | None = None,
    verify_shape: _bool = False,
    allow_broadcast: _bool = False,
) -> tf.compat.v1.TensorProto: ...
def make_ndarray(
    tensor: tf.compat.v1.TensorProto,
) -> any_array: ...

_MapInputT = TypeVar("_MapInputT", bound=TensorLike | Sequence[TensorLike])
_MapOutputT = TypeVar("_MapOutputT", bound=ContainerGeneric[TensorLike])

def map_fn(
    fn: Callable[[_MapInputT], _MapOutputT],
    elems: _MapInputT,
    dtype: DTypeLike | None = None,
    parallel_iterations: int | None = None,
    back_prop: _bool = True,
    swap_memory: _bool = False,
    infer_shape: _bool = True,
    name: str | None = None,
    fn_output_signature: ContainerGeneric[TypeSpec | DType] | None = None,
) -> _MapOutputT: ...
def stop_gradient(input: TensorCompatible, name: str | None = None) -> Tensor: ...
def cond(
    pred: TensorCompatible,
    true_fn: Callable[[], Tensor | None],
    false_fn: Callable[[], Tensor | None],
    name: str | None = None,
) -> Tensor: ...
@contextmanager
def device(device_name: str) -> Iterator[None]: ...
def ensure_shape(
    tensor: TensorCompatible,
    shape: TensorShape | Sequence[int | None],
    name: str | None = None,
) -> Tensor: ...
def norm(
    tensor: tf.Tensor,
    ord: Literal["euclidean", "fro", 1, 2] | float = "euclidean",
    axis: int | None = None,
    keepdims: _bool | None = None,
    name: str | None = None,
) -> Tensor: ...
def size(
    input: TensorCompatible | SparseTensor | RaggedTensor,
    out_type: DTypeLike = ...,
    name: str | None = None,
) -> Tensor: ...
def sequence_mask(
    lengths: Tensor,
    maxlen: int | None = None,
    dtype: DTypeLike = ...,
    name: str | None = None,
) -> Tensor: ...
def tensor_scatter_nd_update(
    tensor: TensorCompatible,
    indices: TensorCompatible,
    updates: TensorCompatible,
    name: str | None = None,
) -> Tensor: ...
def reverse_sequence(
    input: TensorCompatible,
    seq_lengths: TensorCompatible,
    seq_axis: int | None = None,
    batch_axis: int | None = None,
    name: str | None = None,
) -> Tensor: ...
def while_loop(
    cond: Callable[[Tensor, Tensor], Tensor],
    body: Callable[[Tensor, Tensor], tuple[Tensor, Tensor]],
    loop_vars: ContainerTensors,
    shape_invariants: ContainerTensorShape | None = None,
    parallel_iterations: int = 10,
    back_prop: _bool = True,
    swap_memory: _bool = False,
    maximum_iterations: int | None = None,
    name: str | None = None,
) -> list[Tensor]: ...

class UnconnectedGradients(Enum):
    NONE = "none"
    ZERO = "zero"

class _VariableCreatorKwargs(TypedDict, total=False):
    initial_value: TensorCompatible
    trainable: _bool
    validate_shape: _bool
    caching_device: str
    name: str
    shape: tf.TensorShape
    constraint: Constraint | str | dict[str, Any] | None
    synchronization: tf.VariableSynchronization
    aggregation: tf.VariableAggregation
    distribute_strategy: tf.distribute.Strategy

_VariableLike = tf.Variable | ShardedVariable

class _VariableCreator(Protocol):
    def __call__(
        self,
        next_creator: _VariableCreator | None = None,
        **kwargs: Unpack[_VariableCreatorKwargs],
    ) -> _VariableLike: ...

@contextmanager
def variable_creator_scope(
    variable_creator: _VariableCreator,
) -> Generator[None, None, None]: ...

class CriticalSection:
    def execute(
        self,
        fn: Callable[[], _R],
        exclusive_resource_access: _bool = True,
        name: str | None = None,
    ) -> _R: ...

def slice(
    input_: Tensor,
    begin: list[int] | tuple[int, ...] | Tensor,
    size: TensorCompatible,
    name: str | None = None,
) -> Tensor: ...
