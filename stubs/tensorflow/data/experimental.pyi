from typing import Callable, TypeVar

from collections.abc import Sequence

from tensorflow import Tensor, TensorCompatible
from tensorflow.data import Dataset

AUTOTUNE: int
INFINITE_CARDINALITY: int
SHARD_HINT: int
UNKNOWN_CARDINALITY: int

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")

def parallel_interleave(
    map_func: Callable[[_T1], Dataset[_T2]],
    cycle_length: int,
    block_length: int = 1,
    sloppy: bool | None = False,
    buffer_output_elements: int | None = None,
    prefetch_input_elements: int | None = None,
) -> Callable[[Dataset[_T1]], Dataset[_T2]]: ...
def enable_debug_mode() -> None: ...
def cardinality(dataset: Dataset[object]) -> Tensor: ...
def sample_from_datasets(
    datasets: Sequence[Dataset[_T1]],
    weights: TensorCompatible | None = None,
    seed: int | None = None,
    stop_on_empty_dataset: bool = False,
) -> Dataset[_T1]: ...
def ignore_errors(log_warning: bool = False) -> Callable[[Dataset[_T1]], Dataset[_T1]]: ...
