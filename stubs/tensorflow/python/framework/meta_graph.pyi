from collections.abc import Iterable

import tensorflow as tf
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef, SaverDef

def create_meta_graph_def(
    meta_info_def: MetaGraphDef.MetaInfoDef | None = None,
    graph_def: tf.compat.v1.GraphDef | None = None,
    saver_def: SaverDef | None = None,
    collection_list: Iterable[str] | None = None,
    graph: tf.Graph | None = None,
    export_scope: str | None = None,
    exclude_nodes: Iterable[str] | None = None,
    clear_extraneous_savers: bool = False,
    strip_default_attrs: bool = False,
) -> MetaGraphDef: ...
