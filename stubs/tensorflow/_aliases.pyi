from typing import Any, Iterable, Mapping, Sequence, TypeVar

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import InputSpec

_T = TypeVar("_T")

TensorLike = tf.Tensor | tf.SparseTensor | tf.RaggedTensor
SparseTensorLike = tf.Tensor | tf.SparseTensor
RaggedTensorLike = tf.Tensor | tf.RaggedTensor
RaggedTensorLikeT = TypeVar("RaggedTensorLikeT", tf.Tensor, tf.RaggedTensor)

FloatDataSequence = Sequence[float] | Sequence[FloatDataSequence]
StrDataSequence = Sequence[str] | Sequence[StrDataSequence]
ScalarTensorCompatible = tf.Tensor | str | float | np.ndarray[Any, Any] | np.number[Any]

TensorCompatible = ScalarTensorCompatible | Sequence[TensorCompatible]
TensorCompatibleT = TypeVar("TensorCompatibleT", bound=TensorCompatible)
# Sparse tensors are very annoying. Some operations work on them, but many do not. You
# will need to manually verify if an operation supports them. SparseTensorCompatible is intended to be a
# broader type than TensorCompatible and not all operations will support broader version. If unsure,
# use TensorCompatible instead.
SparseTensorCompatible = TensorCompatible | tf.SparseTensor

ShapeLike = tf.TensorShape | Iterable[ScalarTensorCompatible | None] | int | tf.Tensor
DTypeLike = tf.DType | str | np.dtype[Any] | int
GradientsT = tf.Tensor | tf.IndexedSlices

ContainerGeneric = (
    Mapping[str, ContainerGeneric[_T]] | Sequence[ContainerGeneric[_T]] | _T
)

ContainerTensors = ContainerGeneric[tf.Tensor]
ContainerTensorsLike = ContainerGeneric[TensorLike]
ContainerTensorCompatible = ContainerGeneric[TensorCompatible]
ContainerGradients = ContainerGeneric[GradientsT]
ContainerTensorShape = ContainerGeneric[tf.TensorShape]
ContainerInputSpec = ContainerGeneric[InputSpec]

any_array = np.ndarray[Any, Any]
float_array = np.ndarray[Any, np.dtype[np.float32 | np.float64]]
int_array = np.ndarray[Any, np.dtype[np.int32 | np.int64]]
