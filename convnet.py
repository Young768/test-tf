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

batch_size = 512

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


#@tf.function
#def train_step(model, x, y, optimizer, metrics):
#  with tf.GradientTape() as tape:
#    logits = model(x, training=True)
#    # tf.reduce_sum sums the batch sharded per-example loss to a replicated
    # global loss (scalar).
#    loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(
#        y, logits, from_logits=True))

#  gradients = tape.gradient(loss, model.trainable_variables)
#  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

#  for metric in metrics.values():
#    metric.update_state(y_true=y, y_pred=logits)

#  loss_per_sample = loss / len(x)
#  results = {'loss': loss_per_sample}
#  return results


#@tf.function
#def eval_step(model, x, y, metrics):
#  logits = model(x, training=False)
#  loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(
#        y, logits, from_logits=True))

#  for metric in metrics.values():
#    metric.update_state(y_true=y, y_pred=logits)

#  loss_per_sample = loss / len(x)
#  results = {'eval_loss': loss_per_sample}
#  return results

@tf.function
def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_func(labels, predictions)
        loss += tf.reduce_sum(model.losses)
        loss_copy = loss

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))


    return loss_copy


@tf.function
def valid_step(inputs):
    images, labels = inputs
    predictions = model(images, training=False)
    #loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = tf.reduce_sum(tf.keras.losses.sparse_categorical_crossentropy(
        labels, predictions))
    loss_per_sample = loss / len(images)
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
unsharded_layout_4d = dtensor.Layout.replicated(mesh, 4)
unsharded_layout_2d = dtensor.Layout.replicated(mesh, 2)
unsharded_layout_1d = dtensor.Layout.replicated(mesh, 1)

layout_map = tf.keras.dtensor.experimental.LayoutMap(mesh=mesh)
layout_map['conv.*kernel'] = unsharded_layout_4d
layout_map['conv.*bias'] = unsharded_layout_1d
layout_map['feature.*kernel'] = unsharded_layout_2d
layout_map['feature.*bias'] = unsharded_layout_1d
layout_map['test.*beta'] = unsharded_layout_1d
layout_map['test.*gamma'] = unsharded_layout_1d
layout_map['test.*moving_variance'] = unsharded_layout_1d
layout_map['test.*moving_mean'] = unsharded_layout_1d

bn_axis = 1
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
total_ = 200

with tf.keras.dtensor.experimental.layout_map_scope(layout_map):
  inputs = tf.keras.Input((28, 28, 1))
  f = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', name='conv1')(inputs)
  x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(f)
  x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
  x = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
  for i in range(total_):
    layer_name = 'conv_' + str(i)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="Same", activation='relu', name=layer_name)(x)
  x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv3')(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.Dense(64, activation='relu', name='feature')(x)
  output = tf.keras.layers.Dense(10, name='feature2')(x)

  model = tf.keras.Model(inputs, output)
  for weight in model.weights:
      print(f'Weight name: {weight.name} with layout: {weight.layout}')




num_epochs = 300
num_steps = num_epochs
image_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=4)
label_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=1)

for epoch in range(num_epochs):
  print("============================")
  print("Epoch: ", epoch)
  for metric in metrics.values():
    metric.reset_state()
  step = 0
  results = {}
  pbar = tf.keras.utils.Progbar(target=None, stateful_metrics=[])
  data = iter(ds_train)
  input = next(data)
  for _ in range(num_steps):
    images, labels = input[0], input[1]
    images, labels = pack_dtensor_inputs(
        images, labels, image_layout, label_layout)
    #print(images.layout, labels.layout)
    t_input = (images, labels)
    #results.update(train_step(model, images, labels, optimizer, metrics))
    for metric_name, metric in metrics.items():
      results[metric_name] = metric.result()

    pbar.update(step, values=results.items(), finalize=False)
    step += 1
  pbar.update(step, values=results.items(), finalize=True)

  for metric in eval_metrics.values():
    metric.reset_state()
  for input in ds_test:
    images, labels = input[0], input[1]
    images, labels = pack_dtensor_inputs(
        images, labels, image_layout, label_layout)
    v_input = (images, labels)
    valid_step(v_input)
    #results.update(eval_step(model, images, labels, eval_metrics))

  for metric_name, metric in eval_metrics.items():
    results[metric_name] = metric.result()

  for metric_name, metric in results.items():
    print(f"{metric_name}: {metric.numpy()}")
