from collections.abc import Sequence

import tensorflow as tf

class ShardedVariable:
    def __init__(self, variables: Sequence[tf.Variable], name: str | None = "ShardedVariable") -> None: ...
    @property
    def variables(self) -> list[tf.Variable]: ...
    @property
    def name(self) -> str: ...
