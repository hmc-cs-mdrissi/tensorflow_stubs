from typing import Any, Callable

def register_checkpoint_saver(
    package: str = "Custom",
    name: str | None = None,
    predicate: Callable[[Any], bool] | None = None,
    save_fn: Callable[[Any, str], Any] | None = None,
    restore_fn: Callable[[Any, str], None] | None = None,
): ...
def get_save_function(register_name: str) -> str: ...
