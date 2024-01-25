from typing import Any, ContextManager, overload

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager

import numpy as np

from tensorflow import ContainerArrays, ContainerTensorsLike, Graph, Operation, Tensor
from tensorflow.compat.v1 import Session
from tensorflow.core.protobuf.config_pb2 import ConfigProto

class TestCase:
    def setUp(self) -> None: ...
    def tearDown(self) -> None: ...
    @classmethod
    def setUpClass(cls) -> None: ...
    @classmethod
    def tearDownClass(cls) -> None: ...
    @overload
    def evaluate(self, tensors: Tensor) -> np.ndarray[Any, Any]: ...
    @overload
    def evaluate(self, tensors: Operation) -> None: ...
    @overload
    def evaluate(self, tensors: Mapping[str, Tensor]) -> Mapping[str, np.ndarray[Any, Any]]: ...
    @overload
    def evaluate(self, tensors: Sequence[Tensor]) -> Sequence[np.ndarray[Any, Any]]: ...
    @overload
    def evaluate(self, tensors: ContainerTensorsLike) -> ContainerArrays: ...
    def assertEqual(self, first: object, second: object, msg: str | None = None) -> None: ...
    def assertRaises(self, exception: type[Exception]) -> ContextManager[Exception]: ...
    def get_temp_dir(self) -> str: ...
    @contextmanager
    def session(
        self,
        graph: Graph | None = None,
        config: ConfigProto | None = None,
        use_gpu: bool = True,
        force_gpu: bool = False,
    ) -> Iterator[Session]: ...