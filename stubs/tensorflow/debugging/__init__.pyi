from tensorflow import IndexedSlices, Operation, Tensor, TensorCompatible

def assert_all_finite(x: TensorCompatible | IndexedSlices, message: str, name: str | None = None) -> Tensor: ...
def assert_equal(
    x: TensorCompatible,
    y: TensorCompatible,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> Operation | None: ...
def disable_traceback_filtering() -> None: ...
def enable_traceback_filtering() -> None: ...
def enable_check_numerics(stack_height_limit: int = 30, path_length_limit: int = 50) -> None: ...
def disable_check_numerics() -> None: ...
def assert_greater(
    x: TensorCompatible,
    y: TensorCompatible,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> Operation | None: ...
def assert_greater_equal(
    x: TensorCompatible,
    y: TensorCompatible,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> Operation | None: ...
def assert_less(
    x: TensorCompatible,
    y: TensorCompatible,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> Operation | None: ...
def assert_less_equal(
    x: TensorCompatible,
    y: TensorCompatible,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> Operation | None: ...
def assert_positive(
    x: TensorCompatible,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> Operation | None: ...
def assert_negative(
    x: TensorCompatible,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> Operation | None: ...
def assert_non_positive(
    x: TensorCompatible,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> Operation | None: ...
def assert_non_negative(
    x: TensorCompatible,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> Operation | None: ...
