from typing import Any

from tensorflow.keras.optimizers import *
from tensorflow.optimizers import experimental as experimental
from tensorflow.optimizers import legacy as legacy

def __getattr__(name: str) -> Any: ...
