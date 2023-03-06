# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Grappler Remapper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from absl.testing import parameterized

from tensorflow.core.protobuf import config_pb2
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.grappler import tf_optimizer
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import sysconfig
from tensorflow.python.platform import test
from tensorflow.python.util import _pywrap_utils
import tensorflow as tf
import numpy as np

os.environ['TF_CUDNN_USE_FRONTEND'] = '1'
os.environ['TF_CUDNN_USE_RUNTIME_FUSION'] = '1'

@tf.function(jit_compile=True)
def model(a):
  out = tf.keras.layers.Dropout(.2, input_shape=(2,))(a, training=True)
  return out

tf.random.set_seed(0)
data = np.arange(1000000000000).reshape(500000000000, 2).astype(np.float32)

outputs = model(data)
