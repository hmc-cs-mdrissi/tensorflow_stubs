from tensorflow import Tensor, TensorCompatible

from _typeshed import Incomplete

def decode_jpeg(
    contents: TensorCompatible,
    channels: int = 0,
    ratio: int = 1,
    fancy_upscaling: bool = True,
    try_recover_truncated: bool = False,
    acceptable_fraction: float = 1,
    dct_method: str = "",
    name: str | None = None,
) -> Tensor: ...
def __getattr__(name: str) -> Incomplete: ...
