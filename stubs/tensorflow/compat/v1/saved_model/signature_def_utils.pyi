from typing import Mapping

from tensorflow import Tensor
from tensorflow.core.protobuf.meta_graph_pb2 import SignatureDef

def predict_signature_def(inputs: Mapping[str, Tensor], outputs: Mapping[str, Tensor]) -> SignatureDef: ...
