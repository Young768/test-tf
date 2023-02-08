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

os.environ['TF_CUDNN_USE_FRONTEND'] = '1'
os.environ['TF_CUDNN_USE_RUNTIME_FUSION'] = '1'

def _input(shape):
  """Generates an input of a given shape."""
  return variables.Variable(random_ops.truncated_normal(shape, seed=0))


def _weight(shape):
  """Generates a weight of a given shape."""
  # Note that the lambda is needed to allow construction inside loops.
  return variables.Variable(lambda: init_ops.glorot_uniform_initializer(seed=0)
                            (shape))


def _bias(shape):
  """Generates a bias of a given shape."""
  return constant_op.constant(0.1, shape=shape)


def _get_config(remapping_on=False):
  """Returns a CongfigProto with remapper optimizer on/off."""
  rewrite_config = rewriter_config_pb2.RewriterConfig(
      remapping=rewriter_config_pb2.RewriterConfig
      .ON if remapping_on else rewriter_config_pb2.RewriterConfig.OFF)
  rewrite_config.min_graph_nodes = -1
  graph_options = config_pb2.GraphOptions(rewrite_options=rewrite_config)
  config = config_pb2.ConfigProto(graph_options=graph_options)
  return config




def test_conv2d_biasadd_act_fusion():
    """Test Conv2D+BiasAdd+Relu fusion."""

    N, H, W, C = (5, 3, 3, 8)  # pylint: disable=invalid-name
    # The runtime fusion requires the output dims to be 32-bit aligned.

    act_fn = nn.elu
    act_name = 'Elu'

    #if test_util.is_gpu_available(
    #    cuda_only=True, min_cuda_compute_capability=(8, 0)):
      #act_fns += [nn.elu, nn.relu6, nn.leaky_relu]
      #act_names += [b'Elu', b'Relu6', b'LeakyRelu']

    #for precision in ('float16'):
    #for act_fn, act_name in zip(act_fns, act_names):
    use_fp16 = True
        # The runtime fusion (when the activation is not relu) only supports
        # fp16 at this moment.
        #if not use_fp16 and act_name != b'Relu':
        #  continue

    x_shape = [N, C, H, W]
    x_format, b_format = ('NCHW', 'NC..')
    #if use_fp16:
    x_shape = [N, H, W, C]
    x_format, b_format = ('NHWC', 'N..C')

    x = _input(x_shape)
    w = _weight([2, 2, C, C])
    b = _bias([C])

       # if use_fp16:
    x = math_ops.cast(x, dtypes.float16)
    w = math_ops.cast(w, dtypes.float16)
    b = math_ops.cast(b, dtypes.float16)

    @tf.function(jit_compile=True)
    def model():
        y = nn_ops.conv2d(
                x, w, strides=(1, 1), padding='SAME', data_format=x_format)
        z = nn.bias_add(y, b, data_format=b_format)
        out = act_fn(z)
        out = array_ops.identity(out)
        return out
    for i in range(20):
      output = model()
    epilog_ops = [b'BiasAdd', act_name]
    fused_op = ['_FusedConv2D']

test_conv2d_biasadd_act_fusion()
