import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.experimental import dtensor

layers = tf.keras.layers

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  logical_gpus = tf.config.list_logical_devices('GPU')
  print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

DEVICES = [f'GPU:{i}' for i in range(8)]
mesh = dtensor.create_mesh([("batch", 8)], devices=DEVICES)
tf.keras.backend.experimental.enable_tf_random_generator()
tf.keras.utils.set_random_seed(1337)

train_data = tfds.load('imdb_reviews', split='train', shuffle_files=True, batch_size=64)

text_vectorization = tf.keras.layers.TextVectorization(output_mode='tf_idf', max_tokens=1200, output_sequence_length=None)
text_vectorization.adapt(data=train_data.map(lambda x: x['text']))

def vectorize(features):
  return text_vectorization(features['text']), features['label']

train_data_vec = train_data.map(vectorize)
unsharded_layout_2d = dtensor.Layout.replicated(mesh, 2)
unsharded_layout_1d = dtensor.Layout.replicated(mesh, 1)

model = tf.keras.models.Sequential([
  tf.keras.layers.Input(1200),
  tf.keras.layers.Dense(128,
                        activation='relu',
                        name='d1',
                        kernel_layout=unsharded_layout_2d,
                        bias_layout=unsharded_layout_1d),
  tf.keras.layers.BatchNormalization(name="b",
                                     gamma_layout=unsharded_layout_1d,
                                     beta_layout=unsharded_layout_1d,
                                     moving_mean_layout=unsharded_layout_1d,
                                     moving_variance_layout=unsharded_layout_1d,                                     ),
  tf.keras.layers.Dense(2,
                        activation='relu',
                        name='d2',
                        kernel_layout=unsharded_layout_2d,
                        bias_layout=unsharded_layout_1d)
])

sample_x, sample_y = train_data_vec.take(1).get_single_element()
sample_x = dtensor.copy_to_mesh(sample_x, dtensor.Layout.replicated(mesh, rank=2))

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

def repack_batch(x, y, mesh):
  x = repack_local_tensor(x, layout=dtensor.Layout(['batch', dtensor.UNSHARDED], mesh))
  y = repack_local_tensor(y, layout=dtensor.Layout(['batch'], mesh))
  return x, y

sample_x, sample_y = train_data_vec.take(1).get_single_element()
sample_x, sample_y = repack_batch(sample_x, sample_y, mesh)
optimizer = tf.keras.dtensor.experimental.optimizers.Adam(0.01, mesh=mesh)

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
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  for parameter, parameter_gradient in zip(parameters, gradients):
    parameter.assign_sub(learning_rate * parameter_gradient)

  # Define some metrics
  accuracy = 1.0 - tf.reduce_sum(tf.cast(tf.argmax(logits, axis=-1, output_type=tf.int64) != y, tf.float32)) / x.shape[0]
  loss_per_sample = loss / len(x)
  return {'loss': loss_per_sample, 'accuracy': accuracy}

num_epochs = 10

for epoch in range(num_epochs):
  step = 0
  pbar = tf.keras.utils.Progbar(target=int(train_data_vec.cardinality()), stateful_metrics=[])
  metrics = {'epoch': epoch}
  for x,y in train_data_vec:

    x, y = repack_batch(x, y, mesh)

    metrics.update(train_step(model, x, y, 1e-2))

    pbar.update(step, values=metrics.items(), finalize=False)
    step += 1
  pbar.update(step, values=metrics.items(), finalize=True)
