import tempfile
import numpy as np
import tensorflow_datasets as tfds

import tensorflow as tf

from tensorflow.experimental import dtensor
print('TensorFlow version:', tf.__version__)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cpu", help="Using CPU.")
parser.add_argument("--ckpt", help="Enable checkpointing.")
parser.add_argument('--nepoch', nargs='?', const=3, type=int, default=3)
args = parser.parse_args()

DEVICES = []

if args.cpu:
  print("Using CPU right now.")
  def configure_virtual_cpus(ncpu):
    phy_devices = tf.config.list_physical_devices('CPU')
    tf.config.set_logical_device_configuration(phy_devices[0], [
        tf.config.LogicalDeviceConfiguration(),
      ] * ncpu)
  configure_virtual_cpus(8)
  DEVICES = [f'CPU:{i}' for i in range(8)]

  tf.config.list_logical_devices('CPU')
else:
  gpus = tf.config.list_physical_devices('GPU')
  if gpus:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

  DEVICES = [f'GPU:{i}' for i in range(8)]


train_data = tfds.load('imdb_reviews', split='train', shuffle_files=True, batch_size=64)

text_vectorization = tf.keras.layers.TextVectorization(output_mode='tf_idf', max_tokens=1200, output_sequence_length=None)
text_vectorization.adapt(data=train_data.map(lambda x: x['text']))

def vectorize(features):
  return text_vectorization(features['text']), features['label']

train_data_vec = train_data.map(vectorize)

class Dense(tf.Module):

  def __init__(self, input_size, output_size,
               init_seed, weight_layout, activation=None):
    super().__init__()

    random_normal_initializer = tf.function(tf.random.stateless_normal)

    self.weight = dtensor.DVariable(
        dtensor.call_with_layout(
            random_normal_initializer, weight_layout,
            shape=[input_size, output_size],
            seed=init_seed
            ))
    if activation is None:
      activation = lambda x:x
    self.activation = activation

    # bias is sharded the same way as the last axis of weight.
    bias_layout = weight_layout.delete([0])

    self.bias = dtensor.DVariable(
        dtensor.call_with_layout(tf.zeros, bias_layout, [output_size]))

  def __call__(self, x):
    y = tf.matmul(x, self.weight) + self.bias
    y = self.activation(y)

    return y


class BatchNorm(tf.Module):

  def __init__(self):
    super().__init__()

  def __call__(self, x, training=True):
    if not training:
      # This branch is not used in the Tutorial.
      pass
    mean, variance = tf.nn.moments(x, axes=[0])
    return tf.nn.batch_normalization(x, mean, variance, 0.0, 1.0, 1e-5)

def make_keras_bn(bn_layout):
  return tf.keras.layers.BatchNormalization(gamma_layout=bn_layout,
                                            beta_layout=bn_layout,
                                            moving_mean_layout=bn_layout,
                                            moving_variance_layout=bn_layout,
                                            fused=False)

from typing import Tuple

class MLP(tf.Module):

  def __init__(self, dense_layouts: Tuple[dtensor.Layout, dtensor.Layout]):
    super().__init__()

    self.dense1 = Dense(
        1200, 48, (1, 2), dense_layouts[0], activation=tf.nn.relu)
    self.bn = BatchNorm()
    self.dense2 = Dense(48, 2, (3, 4), dense_layouts[1])

  def __call__(self, x):
    y = x
    y = self.dense1(y)
    y = self.bn(y)
    y = self.dense2(y)
    return y

class MLPStricter(tf.Module):

  def __init__(self, mesh, input_mesh_dim, inner_mesh_dim1, output_mesh_dim):
    super().__init__()

    self.dense1 = Dense(
        1200, 48, (1, 2), dtensor.Layout([input_mesh_dim, inner_mesh_dim1], mesh),
        activation=tf.nn.relu)
    self.bn = BatchNorm()
    self.dense2 = Dense(48, 2, (3, 4), dtensor.Layout([inner_mesh_dim1, output_mesh_dim], mesh))


  def __call__(self, x):
    y = x
    y = self.dense1(y)
    y = self.bn(y)
    y = self.dense2(y)
    return y


WORLD = dtensor.create_mesh([("world", 8)], devices=DEVICES)

model = MLP([dtensor.Layout.replicated(WORLD, rank=2),
             dtensor.Layout.replicated(WORLD, rank=2)])

sample_x, sample_y = train_data_vec.take(1).get_single_element()
sample_x = dtensor.copy_to_mesh(sample_x, dtensor.Layout.replicated(WORLD, rank=2))
print(model(sample_x))


def repack_local_tensor(x, layout):
  """Repacks a local Tensor-like to a DTensor with layout.

  This function assumes a single-client application.
  """
  x = tf.convert_to_tensor(x)
  sharded_dims = []

  # For every sharded dimension, use tf.split to split the along the dimension.
  # The result is a nested list of split-tensors in queue[0].
  queue = [x]
  for axis, dim in enumerate(layout.sharding_specs):
    if dim == dtensor.UNSHARDED:
      continue
    num_splits = layout.shape[axis]
    queue = tf.nest.map_structure(lambda x: tf.split(x, num_splits, axis=axis), queue)
    sharded_dims.append(dim)

  # Now we can build the list of component tensors by looking up the location in
  # the nested list of split-tensors created in queue[0].
  components = []
  for locations in layout.mesh.local_device_locations():
    t = queue[0]
    for dim in sharded_dims:
      split_index = locations[dim]  # Only valid on single-client mesh.
      t = t[split_index]
    components.append(t)

  return dtensor.pack(components, layout)

mesh = dtensor.create_mesh([("batch", 8)], devices=DEVICES)

#CPU_DEVICES = [f'CPU:{i}' for i in range(8)]
#cpu_mesh = dtensor.create_mesh([("batch", 8)], devices=CPU_DEVICES)

model = MLP([dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh),
             dtensor.Layout([dtensor.UNSHARDED, dtensor.UNSHARDED], mesh),])

def repack_batch(x, y, mesh):
  x = repack_local_tensor(x, layout=dtensor.Layout(['batch', dtensor.UNSHARDED], mesh))
  y = repack_local_tensor(y, layout=dtensor.Layout(['batch'], mesh))
  return x, y

sample_x, sample_y = train_data_vec.take(1).get_single_element()
sample_x, sample_y = repack_batch(sample_x, sample_y, mesh)

print('x', sample_x[:, 0])
print('y', sample_y)


# Refer to the CTL (custom training loop guide)
@tf.function
def train_step(model, x, y, learning_rate=tf.constant(1e-4)):
  with tf.GradientTape() as tape:
    logits = model(x)
    # tf.reduce_sum sums the batch sharded per-example loss to a replicated
    # global loss (scalar).
    loss = tf.reduce_sum(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=y))
  parameters = model.trainable_variables
  gradients = tape.gradient(loss, parameters)
  for parameter, parameter_gradient in zip(parameters, gradients):
    parameter.assign_sub(learning_rate * parameter_gradient)

  # Define some metrics
  accuracy = 1.0 - tf.reduce_sum(tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int64) != y, tf.float32)) / x.shape[0]
  loss_per_sample = loss / len(x)
  return {'loss': loss_per_sample, 'accuracy': accuracy}

CHECKPOINT_DIR = tempfile.mkdtemp()

def start_checkpoint_manager(mesh, model):
  ckpt = dtensor.DTensorCheckpoint(mesh, root=model)
  manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=3)

  if manager.latest_checkpoint:
    print("Restoring a checkpoint")
    ckpt.restore(manager.latest_checkpoint).assert_consumed()
  else:
    print("new training")
  return manager


num_epochs = args.nepoch
manager = start_checkpoint_manager(mesh, model)

for epoch in range(num_epochs):
  step = 0
  pbar = tf.keras.utils.Progbar(target=int(train_data_vec.cardinality()), stateful_metrics=[])
  metrics = {'epoch': epoch}
  for x,y in train_data_vec:

    x, y = repack_batch(x, y, mesh)

    metrics.update(train_step(model, x, y, 1e-2))

    pbar.update(step, values=metrics.items(), finalize=False)
    step += 1
  if args.ckpt:
    manager.save()
  pbar.update(step, values=metrics.items(), finalize=True)
