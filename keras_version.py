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

mesh = dtensor.create_mesh([("batch", 8)], devices=DEVICES)

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

batch_size = 128

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(batch_size)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(batch_size)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)


@tf.function
def train_step(model, x, y, optimizer, metrics):
  with tf.GradientTape() as tape:
    logits = model(x, training=True)
    # tf.reduce_sum sums the batch sharded per-example loss to a replicated
    # global loss (scalar).
    loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(
        y, logits, from_logits=True))

  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  for metric in metrics.values():
    metric.update_state(y_true=y, y_pred=logits)

  loss_per_sample = loss / len(x)
  results = {'loss': loss_per_sample}
  return results


@tf.function
def eval_step(model, x, y, metrics):
  logits = model(x, training=False)
  loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(
        y, logits, from_logits=True))

  for metric in metrics.values():
    metric.update_state(y_true=y, y_pred=logits)

  loss_per_sample = loss / len(x)
  results = {'eval_loss': loss_per_sample}
  return results

def pack_dtensor_inputs(images, labels, image_layout, label_layout):
  num_local_devices = image_layout.mesh.num_local_devices()
  images = tf.split(images, num_local_devices)
  labels = tf.split(labels, num_local_devices)
  images = dtensor.pack(images, image_layout)
  labels = dtensor.pack(labels, label_layout)
  return  images, labels

optimizer = tf.keras.dtensor.experimental.optimizers.Adam(0.01, mesh=mesh)
metrics = {'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(mesh=mesh)}
eval_metrics = {'eval_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(mesh=mesh)}


#
unsharded_layout_2d = dtensor.Layout.replicated(mesh, 2)
unsharded_layout_1d = dtensor.Layout.replicated(mesh, 1)

layout_map = tf.keras.dtensor.experimental.LayoutMap(mesh=mesh)
layout_map['feature.*kernel'] = unsharded_layout_2d
layout_map['feature.*bias'] = unsharded_layout_1d
layout_map['test.*beta'] = unsharded_layout_1d
layout_map['test.*gamma'] = unsharded_layout_1d
layout_map['test.*moving_variance'] = unsharded_layout_1d
layout_map['test.*moving_mean'] = unsharded_layout_1d

bn_axis = 1
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

with tf.keras.dtensor.experimental.layout_map_scope(layout_map):
  inputs = tf.keras.Input((28,28))
  f = tf.keras.layers.Flatten()(inputs)
  x = tf.keras.layers.Dense(128, activation='relu', name='feature')(f)
  x = tf.keras.layers.BatchNormalization(axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name="test")(x)
  output = tf.keras.layers.Dense(10, name='feature2')(x)
  output = tf.keras.layers.BatchNormalization(axis=bn_axis,
      momentum=BATCH_NORM_DECAY,
      epsilon=BATCH_NORM_EPSILON,
      name="test2")(output)
  model = tf.keras.Model(inputs, output)
  for weight in model.weights:
      print(f'Weight name: {weight.name} with layout: {weight.layout}')




num_epochs = 1000
image_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=4)
label_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=1)

def _split(value, splits, axis=0):
  children = tf.split(value, splits[0], axis=axis)
  if len(splits) > 1:
    splits = splits[1:]
    children = [tf.split(child, splits, axis + 1) for child in children]
  return tf.stack(children)

def pack_tf_tensor(value, layout):
  sharded_tensor = _split(
      value, [layout.num_shards(i) for i in range(layout.rank)])
  flattened = [np.ndarray([])] * layout.mesh.size
  for offset, shard in enumerate(layout.offset_to_shard()):
    flattened[offset] = sharded_tensor[tuple(shard)]
  return dtensor.pack(flattened, layout)

def repack_batch(x, y, x_layout, y_layout):
  x = pack_tf_tensor(x, x_layout)
  y = pack_tf_tensor(y, y_layout)
  return x, y

for epoch in range(num_epochs):
  print("============================")
  print("Epoch: ", epoch)
  for metric in metrics.values():
    metric.reset_state()
  step = 0
  results = {}
  pbar = tf.keras.utils.Progbar(target=None, stateful_metrics=[])

  for input in ds_train:
    images, labels = input[0], input[1]
    images, labels = repack_batch(
        images, labels, image_layout, label_layout)
    #print(images.layout, labels.layout)
    results.update(train_step(model, images, labels, optimizer, metrics))
    for metric_name, metric in metrics.items():
        results[metric_name] = metric.result()

    pbar.update(step, values=results.items(), finalize=False)
    step += 1
  pbar.update(step, values=results.items(), finalize=True)

  for metric in eval_metrics.values():
    metric.reset_state()
  for input in ds_test:
    images, labels = input[0], input[1]
    images, labels = repack_batch(
        images, labels, image_layout, label_layout)
    results.update(eval_step(model, images, labels, eval_metrics))

  for metric_name, metric in eval_metrics.items():
    results[metric_name] = metric.result()

  for metric_name, metric in results.items():
    print(f"{metric_name}: {metric.numpy()}")

