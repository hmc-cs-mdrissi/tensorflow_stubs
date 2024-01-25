from typing import Mapping

import tensorflow as tf
from tensorflow.compat.v1 import Session
from tensorflow.compat.v1.saved_model import builder as builder
from tensorflow.compat.v1.saved_model import loader as loader
from tensorflow.compat.v1.saved_model import signature_constants as signature_constants
from tensorflow.compat.v1.saved_model import signature_def_utils as signature_def_utils
from tensorflow.compat.v1.saved_model import tag_constants as tag_constants
from tensorflow.compat.v1.saved_model.builder import SavedModelBuilder
from tensorflow.compat.v1.saved_model.signature_constants import *
from tensorflow.compat.v1.saved_model.tag_constants import *

Builder = SavedModelBuilder

def simple_save(
    session: Session,
    export_dir: str,
    inputs: Mapping[str, tf.Tensor],
    outputs: Mapping[str, tf.Tensor],
    legacy_init_op: tf.Operation | None = None,
) -> None: ...
