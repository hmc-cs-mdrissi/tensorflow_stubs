from typing_extensions import Self

from collections.abc import Callable

import tensorflow as tf
from tensorflow import _ShapeLike  # pyright: ignore
from tensorflow.python.feature_column.feature_column_v2 import _ExampleSpec  # pyright: ignore
from tensorflow.python.feature_column.feature_column_v2 import FeatureColumn, SequenceDenseColumn

# Strangely at runtime most of Sequence feature columns are defined in feature_column_v2 except
# for this one.
class SequenceNumericColumn(SequenceDenseColumn):
    key: str
    shape: _ShapeLike
    default_value: float
    dtype: tf.DType
    normalizer_fn: Callable[[tf.Tensor], tf.Tensor] | None

    def __new__(
        _cls,
        key: str,
        shape: _ShapeLike,
        default_value: float,
        dtype: tf.DType,
        normalizer_fn: Callable[[tf.Tensor], tf.Tensor] | None,
    ) -> Self: ...
    @property
    def name(self) -> str: ...
    @property
    def parse_example_spec(self) -> _ExampleSpec: ...
    @property
    def parents(self) -> list[FeatureColumn | str]: ...
