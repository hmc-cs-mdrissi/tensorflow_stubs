from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Callable, Generic
from typing import Iterator as _Iterator
from typing import Literal, TypeVar, overload

import numpy as np
import tensorflow as tf
from tensorflow import Tensor, TypeSpec
from tensorflow._aliases import (
    ContainerGeneric,
    ScalarTensorCompatible,
    TensorCompatible,
    TensorLike,
)
from tensorflow.data import experimental as experimental
from tensorflow.data.experimental import AUTOTUNE as AUTOTUNE
from tensorflow.dtypes import DType
from typing_extensions import Self, TypeVarTuple, Unpack

_T1 = TypeVar("_T1", covariant=True)
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")

_Ts = TypeVarTuple("_Ts")

class Iterator(_Iterator[_T1], ABC):
    @abstractmethod
    def get_next(self) -> _T1: ...
    @abstractmethod
    def get_next_as_optional(self) -> tf.experimental.Optional[_T1]: ...

class Dataset(Generic[_T1]):
    element_spec: ContainerGeneric[TypeSpec]
    def apply(
        self: Dataset[_T1], transformation_func: Callable[[Dataset[_T1]], Dataset[_T2]]
    ) -> Dataset[_T2]: ...
    @overload
    def as_numpy_iterator(
        self: Dataset[tf.Tensor],
    ) -> Iterator[np.ndarray[Any, Any]]: ...
    @overload
    def as_numpy_iterator(
        self: Dataset[Mapping[str, tf.Tensor]]
    ) -> Iterator[Mapping[str, np.ndarray[Any, Any]]]: ...
    @overload
    def as_numpy_iterator(self: Dataset[Any]) -> Iterator[Any]: ...
    def batch(
        self: Self,
        batch_size: ScalarTensorCompatible,
        drop_remainder: bool = False,
        num_parallel_calls: int | None = None,
        deterministic: bool | None = None,
        name: str | None = None,
    ) -> Self: ...
    def cache(
        self: Self, filename: TensorCompatible = "", name: str | None = None
    ) -> Self: ...
    def concatenate(self, dataset: Dataset[_T2]) -> Dataset[_T1 | _T2]: ...
    def filter(
        self: Self,
        predicate: Callable[[_T1], bool | tf.Tensor],
        name: str | None = None,
    ) -> Self: ...
    @staticmethod
    def from_tensors(tensors: Any, name: str | None = None) -> Dataset[Any]: ...
    @staticmethod
    def from_tensor_slices(
        tensors: ContainerGeneric[TensorCompatible], name: str | None = None
    ) -> Dataset[Any]: ...
    @staticmethod
    def from_generator(
        generator: Callable[[Unpack[_Ts]], _T2],
        output_types: ContainerGeneric[DType] | None = None,
        output_shapes: ContainerGeneric[tf.TensorShape | Sequence[int | None]]
        | None = None,
        args: tuple[Unpack[_Ts]] | None = None,
        output_signature: ContainerGeneric[TypeSpec] | None = None,
        name: str | None = None,
    ) -> Dataset[_T2]: ...
    def __iter__(self) -> Iterator[_T1]: ...
    @staticmethod
    def list_files(
        file_pattern: str | Sequence[str] | TensorCompatible,
        shuffle: bool | None = None,
        seed: int | None = None,
        name: str | None = None,
    ) -> Dataset[str]: ...
    @overload
    def map(
        self: Dataset[tuple[Unpack[_Ts]]],
        map_func: Callable[[Unpack[_Ts]], _T2],
        num_parallel_calls: int | None = None,
        deterministic: None | bool = None,
        name: str | None = None,
    ) -> Dataset[_T2]: ...
    @overload
    def map(
        self: Dataset[_T1],
        map_func: Callable[[_T1], _T2],
        num_parallel_calls: int | None = None,
        deterministic: None | bool = None,
        name: str | None = None,
    ) -> Dataset[_T2]: ...
    def prefetch(self: Self, buffer_size: int, name: str | None = None) -> Self: ...
    @staticmethod
    @overload
    def range(
        stop: int, /, output_type: DType = tf.int64, name: str | None = None
    ) -> Dataset[tf.Tensor]: ...
    @staticmethod
    @overload
    def range(
        start: int,
        stop: int,
        step: int = 1,
        /,
        output_type: DType = tf.int64,
        name: str | None = None,
    ) -> Dataset[tf.Tensor]: ...
    def rebatch(
        self, batch_size: int, drop_remainder: bool = False, name: str | None = None
    ) -> Dataset[_T1]: ...
    def reduce(
        self,
        initial_state: _T2,
        reduce_func: Callable[[_T2, _T1], _T2],
        name: str | None = None,
    ) -> _T2: ...
    def repeat(
        self: Self, count: int | None = None, name: str | None = None
    ) -> Self: ...
    def shard(
        self: Self, num_shards: int, index: int | tf.Tensor, name: str | None = None
    ) -> Self: ...
    def shuffle(
        self: Self,
        buffer_size: int,
        seed: int | None = None,
        reshuffle_each_iteration: bool = True,
        name: str | None = None,
    ) -> Self: ...
    def take(self: Self, count: int, name: str | None = None) -> Self: ...
    def unbatch(self) -> Dataset[_T1]: ...
    @overload
    @staticmethod
    def zip(
        datasets: tuple[Dataset[_T2], Dataset[_T3]], name: str | None = None
    ) -> Dataset[tuple[_T2, _T3]]: ...
    @overload
    @staticmethod
    def zip(
        datasets: tuple[Dataset[_T2], ...], name: str | None = None
    ) -> Dataset[tuple[_T2, ...]]: ...
    @overload
    def interleave(
        self: Dataset[tuple[Unpack[_Ts]]],
        map_func: Callable[[Unpack[_Ts]], Dataset[_T2]],
        cycle_length: int | None = None,
        block_length: int | None = None,
        num_parallel_calls: int | None = None,
        deterministic: bool | None = None,
        name: str | None = None,
    ) -> Dataset[_T2]: ...
    @overload
    def interleave(
        self,
        map_func: Callable[[_T1], Dataset[_T2]],
        cycle_length: int | None = None,
        block_length: int | None = None,
        num_parallel_calls: int | None = None,
        deterministic: bool | None = None,
        name: str | None = None,
    ) -> Dataset[_T2]: ...
    def save(
        self,
        path: str,
        compression: str | None = None,
        shard_func: Callable[[TensorLike], int] | None = None,
    ): ...

class TFRecordDataset(Dataset[Tensor]):
    def __init__(
        self,
        filenames: TensorCompatible | Dataset[str],
        compression_type: Literal["", "ZLIB", "GZIP"] | None = None,
        buffer_size: int | None = None,
        num_parallel_reads: int | None = None,
        name: str | None = None,
    ) -> None: ...
