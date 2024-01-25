# TODO: Fill in remaining initializers
from typing import Any, Callable, Literal, overload

from tensorflow import DTypeLike, ShapeLike, Tensor, TensorCompatible
from typing_extensions import Self

class Initializer:
    def __call__(self, shape: ShapeLike, dtype: DTypeLike | None = None) -> Tensor: ...
    def get_config(self) -> dict[str, Any]: ...
    @classmethod
    def from_config(cls: type[Self], config: dict[str, Any]) -> Self: ...

class Constant(Initializer):
    def __init__(self, value: TensorCompatible) -> None: ...

class GlorotNormal(Initializer):
    def __init__(self, seed: int | None = None) -> None: ...

class GlorotUniform(Initializer):
    def __init__(self, seed: int | None = None) -> None: ...

class TruncatedNormal(Initializer):
    def __init__(
        self,
        mean: TensorCompatible = 0.0,
        stddev: TensorCompatible = 0.05,
        seed: int | None = None,
    ): ...

class RandomNormal(Initializer):
    def __init__(
        self,
        mean: TensorCompatible = 0.0,
        stddev: TensorCompatible = 0.05,
        seed: int | None = None,
    ): ...

class RandomUniform(Initializer):
    def __init__(
        self,
        minval: TensorCompatible = 0.0,
        maxval: TensorCompatible = 0.05,
        seed: int | None = None,
    ): ...

class VarianceScaling(Initializer):
    def __init__(
        self,
        scale: TensorCompatible = 1.0,
        mode: Literal["fan_in", "fan_out", "fan_avg"] = "fan_in",
        distribution: str = "truncated_normal",
        seed: int | None = None,
    ) -> None: ...

class Zeros(Initializer):
    def __init__(self) -> None: ...

class HeNormal(Initializer):
    def __init__(self, seed: int | None = None) -> None: ...

constant = Constant
glorot_normal = GlorotNormal
glorot_uniform = GlorotUniform
truncated_normal = TruncatedNormal
zeros = Zeros
he_normal = HeNormal

InitializerT = str | Initializer | Callable[[ShapeLike], Tensor] | dict[str, Any] | None

@overload
def get(identifier: None) -> None: ...
@overload
def get(identifier: str | Initializer | dict[str, Any]) -> Initializer: ...
def __getattr__(name: str) -> Any: ...
