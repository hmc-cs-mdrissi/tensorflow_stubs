from typing import Callable, Generic, Iterator, NoReturn
from typing import Sequence as SequenceT
from typing import TypeVar

from abc import ABC, abstractmethod

import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.callbacks import Callback, CallbackList
from tensorflow.keras.utils import experimental as experimental

_T1 = TypeVar("_T1", covariant=True)
_InputT = TypeVar("_InputT")
_OutputT = TypeVar("_OutputT")

class Sequence(Generic[_T1], ABC):
    def on_epoch_end(self) -> None: ...
    @abstractmethod
    def __getitem__(self, index: int) -> _T1: ...
    @abstractmethod
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[_T1]: ...

def set_random_seed(seed: int) -> None: ...
def register_keras_serializable(
    package: str = "Custom", name: str | None = None
) -> Callable[[type[_T1]], type[_T1]]: ...

class SidecarEvaluator:
    def __init__(
        self,
        model: tf.keras.Model[_InputT, _OutputT],
        data: Dataset[tuple[_InputT, _OutputT] | tuple[_InputT, _OutputT, tf.Tensor | _OutputT]]
        | Sequence[tuple[_InputT, _OutputT] | tuple[_InputT, _OutputT, tf.Tensor | _OutputT]],
        checkpoint_dir: str,
        steps: int | None = None,
        max_evaluations: int | None = None,
        callbacks: CallbackList | SequenceT[Callback] | None = None,
    ) -> None: ...
    def start(self) -> NoReturn: ...
