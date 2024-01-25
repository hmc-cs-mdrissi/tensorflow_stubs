from typing import TypeVar

import tensorflow as tf

_T = TypeVar("_T")

def make_one_shot_iterator(dataset: tf.data.Dataset[_T]) -> tf.data.Iterator[_T]: ...
