from typing import Generic, TypeVar

from collections.abc import Callable, Mapping

_T1 = TypeVar("_T1", covariant=True)

class Optional(Generic[_T1]): ...

def dispatch_for_api(api: Callable[..., object], *signatures: Mapping[str, type]) -> Callable[[_T1], _T1]: ...
