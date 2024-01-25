from tensorflow import DTypeLike, RaggedTensor, ShapeLike

def placeholder(
    dtype: DTypeLike, ragged_rank: int, value_shape: ShapeLike | None = None, name: str | None = None
) -> RaggedTensor: ...
