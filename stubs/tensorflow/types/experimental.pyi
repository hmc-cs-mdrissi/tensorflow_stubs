from typing import Any, Generic, TypeVar, overload

import tensorflow as tf
from tensorflow._aliases import ContainerGeneric
from tensorflow.python.framework.func_graph import FuncGraph
from typing_extensions import ParamSpec

P = ParamSpec("P")
R = TypeVar("R", covariant=True)

class Callable(Generic[P, R]):
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R: ...

class GenericFunction(Callable[P, R]):
    @overload
    def get_concrete_function(
        self, *args: P.args, **kwargs: P.kwargs
    ) -> ConcreteFunction[P, R]: ...
    @overload
    def get_concrete_function(  # type: ignore
        self,
        *args: ContainerGeneric[tf.TypeSpec],
        **kwargs: ContainerGeneric[tf.TypeSpec]
    ) -> ConcreteFunction[P, R]: ...

class ConcreteFunction(Callable[P, R]):
    @property
    def structured_input_signature(
        self,
    ) -> tuple[tuple[Any, ...], dict[str, tf.TypeSpec]]: ...
    @property
    def structured_outputs(
        self,
    ) -> dict[str, tf.Tensor | tf.SparseTensor | tf.RaggedTensor | tf.TypeSpec]: ...
    @property
    def graph(self) -> FuncGraph: ...
    @property
    def inputs(self) -> list[tf.Tensor]: ...
    @property
    def outputs(self) -> list[tf.Tensor]: ...
