from collections.abc import Sequence
from contextlib import AbstractContextManager
from enum import Enum
from typing import (
    Any,
    Callable,
    Generic,
    Iterator,
    Literal,
    Mapping,
    NamedTuple,
    NoReturn,
    TypeVar,
)

import tensorflow as tf
from tensorflow import TypeSpec
from tensorflow._aliases import ContainerGeneric
from tensorflow.distribute import cluster_resolver as cluster_resolver
from tensorflow.distribute import experimental as experimental
from tensorflow.distribute.cluster_resolver import ClusterResolver, TPUClusterResolver
from tensorflow.distribute.experimental import CommunicationOptions
from tensorflow.tpu import XLAOptions
from tensorflow.tpu.experimental import DeviceAssignment
from tensorflow.train import ClusterDef, ClusterSpec, ServerDef
from typing_extensions import Self

class InputContext:
    def __init__(
        self,
        num_input_pipelines: int = 1,
        input_pipeline_id: int = 0,
        num_replicas_in_sync: int = 1,
    ) -> None: ...
    @property
    def num_input_pipelines(self) -> int: ...
    @property
    def input_pipeline_id(self) -> int: ...
    @property
    def num_replicas_in_sync(self) -> int: ...
    def get_per_replica_batch_size(self, global_batch_size: int) -> int: ...

class ReplicaContext:
    num_replicas_in_sync: int
    replica_id_in_sync_group: int
    strategy: Strategy

def get_replica_context() -> ReplicaContext | None: ...

class InputReplicationMode(Enum):
    PER_WORKER = "PER_WORKER"
    PER_REPLICA = "PER_REPLICA"

class InputOptions(NamedTuple):
    experimental_fetch_to_device: bool | None = None
    experimental_replication_mode: InputReplicationMode = (
        InputReplicationMode.PER_WORKER
    )
    experimental_place_dataset_on_device: bool = False
    experimental_per_replica_buffer_size: int = 1

_T1 = TypeVar("_T1", covariant=True)

class DistributedIterator(Generic[_T1], Iterator[_T1]):
    @property
    def element_spec(self) -> ContainerGeneric[TypeSpec]: ...

class DistributedDataset(Generic[_T1]):
    @property
    def element_spec(self) -> ContainerGeneric[TypeSpec]: ...
    def __iter__(self) -> DistributedIterator[_T1]: ...

class RunOptions(NamedTuple):
    experimental_enable_dynamic_batch_size: bool = True
    experimental_bucketizing_dynamic_shape: bool = False
    experimental_xla_options: XLAOptions | None = None

class DistributedValues(Generic[_T1]): ...

class Strategy:
    def scope(self) -> AbstractContextManager[Self]: ...
    def distribute_datasets_from_function(
        self,
        dataset_fn: Callable[[InputContext], tf.data.Dataset[_T1]],
        options: InputOptions | None = None,
    ) -> DistributedDataset[_T1]: ...
    def experimental_distribute_dataset(
        self, dataset: tf.data.Dataset[_T1], options: InputOptions | None = None
    ) -> DistributedDataset[_T1]: ...
    def run(
        self,
        fn: Callable[..., _T1],
        args: tuple[object, ...] = (),
        kwargs: Mapping[str, object] | None = None,
        options: RunOptions | None = None,
    ) -> DistributedValues[_T1]: ...
    @property
    def cluster_resolver(self) -> ClusterResolver: ...
    @property
    def num_replicas_in_sync(self) -> int: ...
    @property
    def extended(self) -> StrategyExtended: ...

class StrategyExtended:
    def update(
        self,
        var: tf.Variable,
        fn: Callable[..., tf.Tensor],
        args: Sequence[object] = (),
        kwargs: Mapping[str, object] | None = None,
        group: bool = True,
    ) -> Any: ...

class CrossDeviceOps(object): ...

class AllReduceCrossDeviceOps(CrossDeviceOps):
    def __init__(
        self, all_reduce_alg: str | None = "nccl", num_packs: int | None = 1
    ): ...

class HierarchicalCopyAllReduce(AllReduceCrossDeviceOps):
    def __init__(self, num_packs: int | None = 1): ...

class MirroredStrategy(Strategy):
    def __init__(
        self,
        devices: list[str] | None = None,
        cross_device_ops: CrossDeviceOps | None = None,
    ): ...

class MultiWorkerMirroredStrategy(Strategy):
    def __init__(
        self,
        communication_options: CommunicationOptions | None = None,
        cluster_resolver: ClusterResolver | None = None,
    ) -> None: ...

class TPUStrategy(Strategy):
    def __init__(
        self,
        tpu_cluster_resolver: TPUClusterResolver | None = None,
        experimental_device_assignment: DeviceAssignment | None = None,
        experimental_spmd_xla_partitioning: bool = False,
    ) -> None: ...

def get_strategy() -> Strategy: ...
def has_strategy() -> bool: ...

class Server:
    def __init__(
        self,
        server_or_cluster_def: ServerDef | ClusterDef | ClusterSpec,
        job_name: str | None = None,
        task_index: int | None = None,
        protocol: Literal["grpc+verbs", "grpc", None] = None,
        config: tf.compat.v1.ConfigProto | None = None,
        start: bool = True,
    ) -> None: ...
    def join(self) -> NoReturn: ...
