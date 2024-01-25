from typing import Any, Callable, Dict, TypedDict

from tensorflow import Tensor

from _typeshed import Incomplete

ActivationT = str | None | Callable[[Tensor], Tensor] | Dict[str, Any] | TypedDict

def get(identifier: ActivationT) -> Callable[[Tensor], Tensor]: ...
def __getattr__(name: str) -> Incomplete: ...
