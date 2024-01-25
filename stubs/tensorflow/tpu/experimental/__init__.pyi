from enum import Enum
from typing import Optional

from tensorflow._aliases import int_array
from tensorflow.distribute.cluster_resolver import TPUClusterResolver
from tensorflow.tpu.experimental import embedding as embedding

class Topology: ...

class DeviceAssignment:
    @staticmethod
    def build(
        topology: Topology,
        computation_shape: Optional[int_array] = None,
        computation_stride: Optional[int_array] = None,
        num_replicas: int = 1,
    ) -> DeviceAssignment: ...

def initialize_tpu_system(
    cluster_resolver: TPUClusterResolver | None = None,
) -> Topology: ...

class HardwareFeature:
    class EmbeddingFeature(Enum):
        UNSUPPORTED = "UNSUPPORTED"
        V1 = "V1"
        V2 = "V2"
