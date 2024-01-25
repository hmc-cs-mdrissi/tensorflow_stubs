from typing import NamedTuple

from tensorflow.tpu import experimental as experimental

class XLAOptions(NamedTuple):
    use_spmd_for_xla_partitioning: bool = True
    enable_xla_dynamic_padder: bool = True
