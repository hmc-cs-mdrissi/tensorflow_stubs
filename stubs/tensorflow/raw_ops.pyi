from typing import Literal

import tensorflow as tf
from tensorflow import Tensor
from tensorflow._aliases import TensorCompatible

def Fingerprint(
    data: TensorCompatible, method: Literal["farmhash64"], name: str | None = None
) -> Tensor: ...
def Snapshot(*, input: TensorCompatible, name: str | None = None) -> Tensor: ...
def ResourceApplyAdagradV2(
    var: tf.Tensor,
    accum: tf.Tensor,
    lr: TensorCompatible,
    epsilon: TensorCompatible,
    grad: TensorCompatible,
    use_locking: bool = False,
    update_slots: bool = True,
) -> None: ...
def ResourceSparseApplyAdagradV2(
    var: tf.Tensor,
    accum: tf.Tensor,
    lr: TensorCompatible,
    epsilon: TensorCompatible,
    grad: TensorCompatible,
    indices: TensorCompatible,
    use_locking: bool = False,
    update_slots: bool = True,
) -> None: ...
def ResourceApplyAdam(
    var: tf.Tensor,
    m: tf.Tensor,
    v: tf.Tensor,
    beta1_power: TensorCompatible,
    beta2_power: TensorCompatible,
    lr: TensorCompatible,
    beta1: TensorCompatible,
    beta2: TensorCompatible,
    epsilon: TensorCompatible,
    grad: TensorCompatible,
    use_locking: bool = False,
    use_nesterov: bool = False,
) -> None: ...
