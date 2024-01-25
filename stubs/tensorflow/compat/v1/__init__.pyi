from typing import Any, Mapping, MutableSequence, NamedTuple, Sequence, overload
from typing_extensions import Self

from collections.abc import Callable, Iterable
from types import TracebackType

from google.protobuf.internal.containers import MessageMap, RepeatedCompositeFieldContainer
from google.protobuf.message import Message

import numpy as np

import tensorflow as tf
from tensorflow import DTypeLike, ShapeLike, TensorCompatible
from tensorflow.compat.v1 import data as data
from tensorflow.compat.v1 import graph_util as graph_util
from tensorflow.compat.v1 import ragged as ragged
from tensorflow.compat.v1 import saved_model as saved_model
from tensorflow.compat.v1 import train as train
from tensorflow.core.framework import types_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf.config_pb2 import ConfigProto as ConfigProto
from tensorflow.keras.constraints import ConstraintT
from tensorflow.keras.initializers import InitializerT
from tensorflow.keras.regularizers import RegularizerT

# Would be better to use mypy-protobuf to make this.
class GraphDef(Message):
    node: RepeatedCompositeFieldContainer[NodeDef]
    versions: Message
    library: FunctionDefLibrary

class FunctionDefLibrary(Message):
    function: RepeatedCompositeFieldContainer[FunctionDef]

class FunctionDef(Message):
    signature: OpDef
    node_def: RepeatedCompositeFieldContainer[NodeDef]

class OpDef(Message):
    class ArgDef(Message): ...
    name: str
    input_arg: RepeatedCompositeFieldContainer[OpDef.ArgDef]

class NodeDef(Message):
    name: str
    device: str
    op: str
    input: MutableSequence[str]
    attr: MessageMap[str, AttrValue]

class AttrValue(Message):
    class ListValue(Message):
        s: MutableSequence[bytes]
        i: MutableSequence[int]
        f: MutableSequence[float]
        b: MutableSequence[bool]
        type: MutableSequence[types_pb2.DataType]
        shape: MutableSequence[TensorShapeProto]
        tensor: MutableSequence[TensorProto]
        func: MutableSequence[NameAttrList]
    s: bytes
    i: int
    f: float
    b: bool
    type: types_pb2.DataType
    shape: TensorShapeProto
    tensor: TensorProto
    list: ListValue
    func: NameAttrList

class TensorShapeProto(Message):
    class Dim(Message):
        size: int
        name: str
    dim: RepeatedCompositeFieldContainer[TensorShapeProto.Dim]

class TensorProto(Message):
    float_val: MutableSequence[float]
    double_val: MutableSequence[float]
    int_val: MutableSequence[int]
    string_val: MutableSequence[bytes]
    tensor_shape: TensorShapeProto
    dtype: int
    tensor_content: bytes

class NameAttrList(Message):
    name: str
    attr: MessageMap[str, AttrValue]

class RunOptions(Message):
    FULL_TRACE = 3

class RunMetadata(Message): ...

_GraphElement = tf.Tensor | tf.SparseTensor | tf.RaggedTensor | tf.Operation | str
_FeedElement = TensorCompatible
# This is a simplification. Key being invariant in a Mapping makes the real type difficult to write. This
# is enough to cover vast majority of use cases.
_FeedDict = Mapping[str, _FeedElement] | Mapping[tf.Tensor, _FeedElement] | Mapping[tf.SparseTensor, _FeedElement]

class Session:
    graph: tf.Graph
    graph_def: GraphDef
    def __init__(
        self,
        *,
        graph: tf.Graph | None = None,
        config: config_pb2.ConfigProto | None = None,
    ) -> None: ...
    @overload
    def run(
        self,
        fetches: _GraphElement,
        feed_dict: _FeedDict | None = None,
        options: RunOptions | None = None,
        run_metadata: RunMetadata | None = None,
    ) -> np.ndarray[Any, Any]: ...
    @overload
    def run(
        self,
        fetches: Sequence[_GraphElement],
        feed_dict: _FeedDict | None = None,
        options: RunOptions | None = None,
        run_metadata: RunMetadata | None = None,
    ) -> list[np.ndarray[Any, Any]]: ...
    @overload
    def run(
        self,
        fetches: Mapping[str, _GraphElement],
        feed_dict: _FeedDict | None = None,
        options: RunOptions | None = None,
        run_metadata: RunMetadata | None = None,
    ) -> dict[str, np.ndarray[Any, Any]]: ...
    def __enter__(self) -> Self: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None: ...

def disable_eager_execution() -> None: ...
def disable_v2_behavior() -> None: ...
def enable_resource_variables() -> None: ...
def global_variables(scope: str | None = None) -> list[tf.Variable]: ...
def global_variables_initializer() -> tf.Operation: ...
def variables_initializer(var_list: Iterable[tf.Variable], name: str = "init") -> tf.Operation: ...
def tables_initializer() -> tf.Operation: ...
def trainable_variables(scope: str | None = None) -> list[tf.Variable]: ...
def get_default_graph() -> tf.Graph: ...
def reset_default_graph() -> None: ...

class SparseTensorValue(NamedTuple):
    indices: np.ndarray[Any, Any]
    values: np.ndarray[Any, Any]
    dense_shape: np.ndarray[Any, Any]

def placeholder(dtype: DTypeLike, shape: ShapeLike | None = None, name: str | None = None) -> tf.Tensor: ...
def sparse_placeholder(
    dtype: DTypeLike, shape: ShapeLike | None = None, name: str | None = None
) -> tf.SparseTensor: ...

class variable_scope:
    def __init__(
        self,
        name_or_scope: str | None,
        default_name: str | None = None,
        values: Sequence[tf.Tensor] | None = None,
        initializer: InitializerT = None,
        regularizer: RegularizerT = None,
        caching_device: str | None = None,
        partitioner: Callable[[tf.TensorShape, tf.DType], Sequence[int]] | None = None,
        custom_getter: Callable[..., tf.Tensor] | None = None,
        reuse: bool | None = None,
        dtype: tf.DType | None = None,
        use_resource: bool | None = None,
        constraint: ConstraintT = None,
        auxiliary_name_scope: bool = True,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None
    ) -> None: ...

def get_variable(
    name: str,
    shape: ShapeLike | None = None,
    dtype: tf.DType | None = None,
    initializer: TensorCompatible | Callable[[], TensorCompatible] | tf.keras.initializers.Initializer | None = None,
    regularizer: RegularizerT = None,
    trainable: bool | None = None,
    collections: Sequence[str] | None = None,
    caching_device: str | None = None,
    partitioner: Callable[[tf.TensorShape, tf.DType], Sequence[int]] | None = None,
    validate_shape: bool = True,
    use_resource: bool | None = None,
    custom_getter: Callable[..., tf.Tensor] | None = None,
    constraint: ConstraintT = None,
    synchronization: tf.VariableSynchronization = tf.VariableSynchronization.AUTO,
    aggregation: tf.VariableAggregation = tf.VariableAggregation.NONE,
) -> tf.Variable: ...
def min_max_variable_partitioner(
    max_partitions: int = 1, axis: int = 0, min_slice_size: int = (256 << 10), bytes_per_string_element: int = 16
) -> Callable[[tf.TensorShape, tf.DType], list[int]]: ...
def __getattr__(name: str) -> Any: ...
