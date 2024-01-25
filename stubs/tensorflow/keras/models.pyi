from typing import Any, Mapping, TypeVar
from typing_extensions import ParamSpec

from tensorflow.keras import Model as Model
from tensorflow.keras import Sequential as Sequential
from tensorflow.saved_model import LoadOptions, _LoadedAttributes  # type: ignore

_P = ParamSpec("_P")
_R = TypeVar("_R", covariant=True)

class _LoadedKerasModel(Model[Any, Any], _LoadedAttributes[_P, _R]): ...

def load_model(
    filepath: str,
    custom_objects: Mapping[str, object] | None = None,
    compile: bool = True,
    options: LoadOptions | None = None,
) -> _LoadedKerasModel[..., Any]: ...
