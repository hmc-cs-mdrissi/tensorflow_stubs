from typing import Literal, NamedTuple

from types import TracebackType

class ProfilerOptions(NamedTuple):
    host_tracer_level: Literal[1, 2, 3] = 2
    python_tracer_level: Literal[0, 1] = 0
    device_tracer_level: Literal[0, 1] = 1
    delay_ms: int | None = None

def start(logdir: str, options: ProfilerOptions | None = None) -> None: ...
def stop(save: bool = True) -> None: ...

class Trace:
    def __init__(self, name: str, **kwargs: object) -> None: ...
    def set_metadata(self, **kwargs: object) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self, exc_type: type[BaseException] | None, exc_value: BaseException | None, tb: TracebackType | None
    ) -> bool: ...
