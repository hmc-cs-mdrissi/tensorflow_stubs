from typing import Any, Generic, TypeVar

_Value = TypeVar("_Value", covariant=True)

class RemoteValue(Generic[_Value]):
    # With HKT you may be able to better describe fetch.
    def fetch(self) -> Any: ...
    def get(self) -> _Value: ...
