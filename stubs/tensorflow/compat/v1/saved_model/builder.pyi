from typing import Iterable, Mapping

from tensorflow import Operation, Tensor
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1.train import Saver
from tensorflow.core.protobuf.meta_graph_pb2 import SignatureDef

class SavedModelBuilder:
    def __init__(self, export_dir: str) -> None: ...
    def save(self, as_text: bool = False) -> str: ...
    def add_meta_graph_and_variables(
        self,
        sess: Session,
        tags: Iterable[str],
        signature_def_map: Mapping[str, SignatureDef] | None = None,
        assets_list: Iterable[Tensor] | None = None,
        clear_devices: bool = False,
        init_op: Operation | None = None,
        train_op: Operation | None = None,
        strip_default_attrs: bool = False,
        saver: Saver | None = None,
        main_op: Operation | None = None,
        legacy_init_op: Operation | None = None,
    ): ...
