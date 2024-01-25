import tensorflow
from tensorflow.core.protobuf import meta_graph_pb2

def build_tensor_info(tensor: tensorflow.Tensor) -> meta_graph_pb2.TensorInfo: ...
