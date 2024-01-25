from enum import Enum

from tensorflow.distribute import MultiWorkerMirroredStrategy as _MultiWorkerMirroredStrategy
from tensorflow.distribute import Strategy, StrategyExtended
from tensorflow.distribute.cluster_resolver import ClusterResolver
from tensorflow.distribute.experimental import coordinator as coordinator
from tensorflow.distribute.experimental import partitioners as partitioners
from tensorflow.distribute.experimental.partitioners import Partitioner

class ParameterServerStrategyExtended(StrategyExtended):
    _variable_partitioner: Partitioner | None

class ParameterServerStrategy(Strategy):
    _num_workers: int
    _cluster_resolver: ClusterResolver
    def __init__(self, cluster_resolver: ClusterResolver, variable_partitioner: Partitioner | None = None) -> None: ...
    @property
    def extended(self) -> ParameterServerStrategyExtended: ...

class MultiWorkerMirroredStrategy(_MultiWorkerMirroredStrategy):
    def __init__(
        self,
        communication: CommunicationImplementation = CommunicationImplementation.AUTO,
        cluster_resolver: ClusterResolver | None = None,
    ) -> None: ...

class CommunicationOptions:
    def __init__(
        self,
        bytes_per_pack: int = 0,
        timeout_seconds: float | None = None,
        implementation: CommunicationImplementation = CommunicationImplementation.AUTO,
    ) -> None: ...

class CentralStorageStrategy(Strategy):
    def __init__(self, compute_devices: list[str] | None = None, parameter_device: str | None = None): ...

class CommunicationImplementation(Enum):
    AUTO = "AUTO"
    RING = "RING"
    NCCL = "NCCL"
