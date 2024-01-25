import tensorflow as tf
from tensorflow.config import experimental as experimental
from tensorflow.config import optimizer as optimizer
from tensorflow.config import threading as threading
from tensorflow.config.experimental import ClusterDeviceFilters
from tensorflow.distribute.cluster_resolver import ClusterResolver

class PhysicalDevice: ...

def experimental_connect_to_cluster(
    cluster_spec_or_resolver: tf.train.ClusterSpec | ClusterResolver,
    job_name: str = "localhost",
    task_index: int = 0,
    protocol: str | None = None,
    make_master_device_default: bool = True,
    cluster_device_filters: ClusterDeviceFilters | None = None,
): ...
def list_physical_devices(device_type: str | None = None) -> list[PhysicalDevice]: ...
def run_functions_eagerly(run_eagerly: bool) -> bool: ...
def set_soft_device_placement(enabled: bool) -> bool: ...
def set_visible_devices(devices: list[PhysicalDevice], device_type: str | None = None) -> None: ...
