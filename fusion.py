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
from tensorflow.python.platform import sysconfig as sysconfig_lib
from tensorflow.python.platform import test
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.client import device_lib
import tensorflow as tf
from tensorflow.python.framework import gpu_util
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import tf_logging as logging

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

def is_gpu_available(cuda_only=False, min_cuda_compute_capability=None):
  """Returns whether TensorFlow can access a GPU.
  Warning: if a non-GPU version of the package is installed, the function would
  also return False. Use `tf.test.is_built_with_cuda` to validate if TensorFlow
  was build with CUDA support.
  For example,
  >>> gpu_available = tf.test.is_gpu_available()
  >>> is_cuda_gpu_available = tf.test.is_gpu_available(cuda_only=True)
  >>> is_cuda_gpu_min_3 = tf.test.is_gpu_available(True, (3,0))
  Args:
    cuda_only: limit the search to CUDA GPUs.
    min_cuda_compute_capability: a (major,minor) pair that indicates the minimum
      CUDA compute capability required, or None if no requirement.
  Note that the keyword arg name "cuda_only" is misleading (since routine will
  return true when a GPU device is available irrespective of whether TF was
  built with CUDA support or ROCm support. However no changes here because
  ++ Changing the name "cuda_only" to something more generic would break
     backward compatibility
  ++ Adding an equivalent "rocm_only" would require the implementation check
     the build type. This in turn would require doing the same for CUDA and thus
     potentially break backward compatibility
  ++ Adding a new "cuda_or_rocm_only" would not break backward compatibility,
     but would require most (if not all) callers to update the call to use
     "cuda_or_rocm_only" instead of "cuda_only"
  Returns:
    True if a GPU device of the requested kind is available.
  """

  # This was needed earlier when we had support for SYCL in TensorFlow.
  del cuda_only

  try:
    for local_device in device_lib.list_local_devices():
      if local_device.device_type == "GPU":
        gpu_info = gpu_util.compute_capability_from_device_desc(local_device)
        cc = gpu_info.compute_capability or (0, 0)
        if not min_cuda_compute_capability or cc >= min_cuda_compute_capability:
          return True
    return False
  except errors_impl.NotFoundError as e:
    if not all(x in str(e) for x in ["CUDA", "not find"]):
      raise e
    else:
      logging.error(str(e))
      return False
@tf.function(jit_compile=True)
def test_conv2d_biasadd_act_fusion():
    """Test Conv2D+BiasAdd+Relu fusion."""
    if not test_util.is_gpu_available():
      print('No GPU available')

    N, H, W, C = (5, 3, 3, 8)  # pylint: disable=invalid-name
    # The runtime fusion requires the output dims to be 32-bit aligned.
    ##self.assertEqual(C % 2, 0)

    act_fns = [nn.relu]
    act_names = [b'Relu']

    if test_util.is_gpu_available(
        cuda_only=True, min_cuda_compute_capability=(8, 0)):
      act_fns += [nn.elu, nn.relu6, nn.leaky_relu]
      act_names += [b'Elu', b'Relu6', b'LeakyRelu']

    for precision in ('float16', 'float32'):
      for act_fn, act_name in zip(act_fns, act_names):
        use_fp16 = precision == 'float16'
        # The runtime fusion (when the activation is not relu) only supports
        # fp16 at this moment.
        if not use_fp16 and act_name != b'Relu':
          continue

        ops.reset_default_graph()
        x_shape = [N, C, H, W]
        x_format, b_format = ('NCHW', 'NC..')
        if use_fp16:
          x_shape = [N, H, W, C]
          x_format, b_format = ('NHWC', 'N..C')

        x = _input(x_shape)
        w = _weight([2, 2, C, C])
        b = _bias([C])

        if use_fp16:
          x = math_ops.cast(x, dtypes.float16)
          w = math_ops.cast(w, dtypes.float16)
          b = math_ops.cast(b, dtypes.float16)

        y = nn_ops.conv2d(
            x, w, strides=(1, 1), padding='SAME', data_format=x_format)
        z = nn.bias_add(y, b, data_format=b_format)
        out = act_fn(z)
        out = array_ops.identity(out)
        print(out)
        epilog_ops = [b'BiasAdd', act_name]
        fused_op = ['_FusedConv2D']


