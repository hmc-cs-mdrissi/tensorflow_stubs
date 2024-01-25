from typing import Literal

from abc import ABC, abstractmethod

from tensorflow.train import ClusterSpec

from _typeshed import Incomplete

class ClusterResolver(ABC):
    @abstractmethod
    def cluster_spec(self) -> ClusterSpec: ...
    @property
    def task_type(self) -> str | None: ...
    @property
    def task_id(self) -> int: ...
    @property
    def rpc_layer(self) -> Literal["grpc", "grpc+verbs"] | None: ...

class GCEClusterResolver:
    def __init__(
        self,
        project: str,
        zone: str,
        instance_group: str,
        port: int,
        task_type: str = "worker",
        task_id: int = 0,
        rpc_layer: str = "grpc",
        credentials: str = "default",
        service: Incomplete | None = None,
    ) -> None: ...
    def cluster_spec(self) -> ClusterSpec: ...

class KubernetesClusterResolver(ClusterResolver):
    def cluster_spec(self) -> ClusterSpec: ...

class TPUClusterResolver(ClusterResolver):
    def __init__(
        self,
        # Only length 1 lists are supported.
        tpu: str | list[str] | None = None,
        zone: str | None = None,
        project: str | None = None,
        job_name: str = "worker",
        coordinator_name: str | None = None,
        coordinator_address: str | None = None,
        credentials: Incomplete = "default",
        service: Incomplete | None = None,
        discovery_url: str | None = None,
    ) -> None: ...
    def cluster_spec(self) -> ClusterSpec: ...

class TFConfigClusterResolver(ClusterResolver):
    def cluster_spec(self) -> ClusterSpec: ...

class SlurmClusterResolver(ClusterResolver):
    def cluster_spec(self) -> ClusterSpec: ...

class SimpleClusterResolver(ClusterResolver):
    def __init__(
        self,
        cluster_spec: ClusterSpec,
        master: str = "",
        task_type: str | None = None,
        task_id: int | None = None,
        environment: str | None = "",
        num_accelerators: int | None = None,
        rpc_layer: str | None = None,
    ): ...
    def cluster_spec(self) -> ClusterSpec: ...
