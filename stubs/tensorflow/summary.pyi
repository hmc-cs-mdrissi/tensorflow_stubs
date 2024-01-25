from typing import Callable, ContextManager, Iterator
from typing_extensions import Self

from contextlib import contextmanager

import tensorflow as tf

class SummaryWriter:
    def as_default(self, step: int | None = None) -> ContextManager[Self]: ...

def scalar(
    name: str, data: float | tf.Tensor, step: int | tf.Tensor | None = None, description: str | None = None
) -> bool: ...
def histogram(
    name: str, data: tf.Tensor, step: int | None = None, buckets: int | None = None, description: str | None = None
) -> bool: ...
def graph(graph_data: tf.Graph | tf.compat.v1.GraphDef) -> bool: ...
@contextmanager
def record_if(condition: bool | tf.Tensor | Callable[[], bool]) -> Iterator[None]: ...
def create_file_writer(
    logdir: str,
    max_queue: int | None = None,
    flush_millis: int | None = None,
    filename_suffix: str | None = None,
    name: str | None = None,
    experimental_trackable: bool = False,
) -> SummaryWriter: ...
