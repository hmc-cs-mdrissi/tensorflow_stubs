from typing import Any, Callable, TypeVar
from typing_extensions import Self

from google.protobuf.message import Message

import numpy as np

import tensorflow as tf
from tensorflow.python.training.tracking.base import Trackable

class Example(Message):
    features: Features

class Features(Message):
    feature: dict[str, Feature]

class Feature(Message):
    float_list: FloatList
    int64_list: Int64List
    bytes_list: BytesList

class FloatList(Message):
    value: list[float]

class Int64List(Message):
    value: list[int]

class BytesList(Message):
    value: list[bytes]

class ServerDef(Message): ...
class ClusterDef(Message): ...

_T = TypeVar("_T", bound=list[str] | tuple[str] | dict[int, str])

class ClusterSpec:
    def __init__(self, cluster: dict[str, _T] | ClusterDef | ClusterSpec) -> None: ...
    def as_dict(self) -> dict[str, list[str] | tuple[str] | dict[int, str]]: ...
    def num_tasks(self, job_name: str) -> int: ...

class CheckpointOptions:
    def __init__(
        self, experimental_io_device: str | None = None, experimental_enable_async_checkpoint: bool = False
    ): ...

class CheckpointReader:
    def get_variable_to_shape_map(self) -> dict[str, list[int]]: ...
    def get_variable_to_dtype_map(self) -> dict[str, tf.DType]: ...
    def get_tensor(self, name: str) -> np.ndarray[Any, Any] | Any: ...

class _CheckpointLoadStatus:
    def assert_consumed(self) -> Self: ...
    def assert_existing_objects_matched(self) -> Self: ...
    def assert_nontrivial_match(self) -> Self: ...
    def expect_partial(self) -> Self: ...

class Checkpoint:
    def __init__(self, root: Trackable | None = None, **kwargs: Trackable): ...
    def write(self, file_prefix: str, options: CheckpointOptions | None = None) -> str: ...
    def read(self, file_prefix: str, options: CheckpointOptions | None = None) -> _CheckpointLoadStatus: ...
    def restore(self, file_prefix: str, options: CheckpointOptions | None = None) -> _CheckpointLoadStatus: ...

class CheckpointManager:
    def __init__(
        self,
        checkpoint: Checkpoint,
        directory: str,
        max_to_keep: int,
        keep_checkpoint_every_n_hours: int | None = None,
        checkpoint_name: str = "ckpt",
        step_counter: tf.Variable | None = None,
        checkpoint_interval: int | None = None,
        init_fn: Callable[[], object] | None = None,
    ): ...
    def _sweep(self) -> None: ...

def latest_checkpoint(checkpoint_dir: str, latest_filename: str | None = None) -> str: ...
def load_checkpoint(ckpt_dir_or_file: str) -> CheckpointReader: ...
def load_variable(ckpt_dir_or_file: str, name: str) -> np.ndarray[Any, Any]: ...
def list_variables(ckpt_dir_or_file: str) -> list[tuple[str, list[int]]]: ...
