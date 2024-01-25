from typing import Any, Callable, Generic, Mapping, NamedTuple, TypeVar

from tensorflow import RaggedTensor, SparseTensor, Tensor
from tensorflow.io import _FeatureSpecs  # type: ignore

class ServingInputReceiver(NamedTuple):
    features: Tensor | SparseTensor | Mapping[str, Tensor | SparseTensor | RaggedTensor]
    receiver_tensors: Tensor | SparseTensor | Mapping[str, Tensor | SparseTensor | RaggedTensor]
    receiver_tensors_alternatives: Any = None

_T = TypeVar("_T", bound=Tensor | dict[str, Tensor])

class ExportOutput: ...

class PredictOutput(Generic[_T], ExportOutput):
    outputs: _T
    def __init__(self, outputs: _T) -> None: ...

def build_parsing_serving_input_receiver_fn(
    feature_spec: _FeatureSpecs, default_batch_size: int | None = None
) -> Callable[[], ServingInputReceiver]: ...
