from builtins import bool as _bool
from typing import Any

from tensorflow._aliases import DTypeLike
from tensorflow.core.framework import types_pb2

# If we want to handle tensors as generic on dtypes we likely need to make
# this class an Enum. That's a minor lie type wise, but Literals only work
# with basic types + enums.
class DType:
    @property
    def name(self) -> str: ...
    @property
    def as_datatype_enum(self) -> types_pb2.DataType: ...
    @property
    def as_numpy_dtype(self) -> type[Any]: ...
    @property
    def is_numpy_compatible(self) -> _bool: ...
    @property
    def is_bool(self) -> _bool: ...
    @property
    def is_floating(self) -> _bool: ...
    @property
    def is_integer(self) -> _bool: ...
    @property
    def is_quantized(self) -> _bool: ...
    @property
    def is_unsigned(self) -> _bool: ...
    @property
    def base_dtype(self) -> DType: ...
    @property
    def size(self) -> int: ...

bool: DType = ...
complex128: DType = ...
complex64: DType = ...
bfloat16: DType = ...
float16: DType = ...
half: DType = ...
float32: DType = ...
float64: DType = ...
double: DType = ...
int8: DType = ...
int16: DType = ...
int32: DType = ...
int64: DType = ...
uint8: DType = ...
uint16: DType = ...
uint32: DType = ...
uint64: DType = ...
qint8: DType = ...
qint16: DType = ...
qint32: DType = ...
quint8: DType = ...
quint16: DType = ...
resource: DType = ...
string: DType = ...

def as_dtype(dtype: DTypeLike) -> DType: ...
