from typing import Any, Callable, overload
from typing_extensions import Self

from tensorflow import Tensor

class Regularizer:
    def get_config(self) -> dict[str, Any]: ...
    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Self: ...
    def __call__(self, x: Tensor, /) -> Tensor: ...

RegularizerT = str | dict[str, Any] | Regularizer | None

@overload
def get(identifer: None) -> None: ...
@overload
def get(identifer: Callable[[Tensor], Tensor]) -> Callable[[Tensor], Tensor]: ...
@overload
def get(identifer: str | dict[str, Any]) -> Regularizer: ...
def __getattr__(name: str) -> Any: ...
