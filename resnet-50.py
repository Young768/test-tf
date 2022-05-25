import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.experimental import dtensor
from tensorflow.keras import backend
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import models

import argparse

layers = tf.keras.layers

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  logical_gpus = tf.config.list_logical_devices('GPU')
  print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

DEVICES = [f'GPU:{i}' for i in range(8)]
mesh = dtensor.create_mesh([("batch", 8)], devices=DEVICES)

NUM_CLASSES = 1000

tf.keras.backend.experimental.enable_tf_random_generator()
tf.keras.utils.set_random_seed(1337)

mesh = dtensor.create_mesh([("batch", 8)], devices=DEVICES)
batch_size = 128

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

layers = tf.keras.layers

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

DEVICES = [f'GPU:{i}' for i in range(8)]
mesh = dtensor.create_mesh([("batch", 8)], devices=DEVICES)


def _gen_l2_regularizer(use_l2_regularizer=True):
    return regularizers.l2(L2_WEIGHT_DECAY) if use_l2_regularizer else None


def identity_block(input_tensor,
                   kernel_size,
                   filters,
                   stage,
                   block,
                   use_l2_regularizer=True):
    """The identity block is the block that has no conv layer at shortcut.
    Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.
    Returns:
      Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters1, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(
        input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        fused=False,
        name=bn_name_base + '2a')(
        x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(
        x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        fused=False,
        name=bn_name_base + '2b')(
        x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2c')(
        x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        fused=False,
        name=bn_name_base + '2c')(
        x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2),
               use_l2_regularizer=True):
    """A block that has a conv layer at shortcut.
    Note that from stage 3,
    the second conv layer at main path is with strides=(2, 2)
    And the shortcut should have strides=(2, 2) as well
    Args:
      input_tensor: input tensor
      kernel_size: default 3, the kernel size of middle conv layer at main path
      filters: list of integers, the filters of 3 conv layer at main path
      stage: integer, current stage label, used for generating layer names
      block: 'a','b'..., current block label, used for generating layer names
      strides: Strides for the second conv layer in the block.
      use_l2_regularizer: whether to use L2 regularizer on Conv layer.
    Returns:
      Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(
        filters1, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2a')(
        input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        fused=False,
        name=bn_name_base + '2a')(
        x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters2,
        kernel_size,
        strides=strides,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2b')(
        x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        fused=False,
        name=bn_name_base + '2b')(
        x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters3, (1, 1),
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '2c')(
        x)
    x = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        fused=False,
        name=bn_name_base + '2c')(
        x)

    shortcut = layers.Conv2D(
        filters3, (1, 1),
        strides=strides,
        use_bias=False,
        kernel_initializer='he_normal',
        kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
        name=conv_name_base + '1')(
        input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        fused=False,
        name=bn_name_base + '1')(
        shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)
    return x


def resnet50(num_classes,
             batch_size=None,
             use_l2_regularizer=True,
             rescale_inputs=False):
    """Instantiates the ResNet50 architecture.
    Args:
      num_classes: `int` number of classes for image classification.
      batch_size: Size of the batches for each step.
      use_l2_regularizer: whether to use L2 regularizer on Conv/Dense layer.
      rescale_inputs: whether to rescale inputs from 0 to 1.
    Returns:
        A Keras model instance.
    """

    unsharded_layout_4d = dtensor.Layout.replicated(mesh, 4)
    unsharded_layout_2d = dtensor.Layout.replicated(mesh, 2)
    unsharded_layout_1d = dtensor.Layout.replicated(mesh, 1)

    layout_map = tf.keras.dtensor.experimental.LayoutMap(mesh=mesh)

    layout_map['conv1.*kernel'] = unsharded_layout_4d
    layout_map['res.*kernel'] = unsharded_layout_4d
    layout_map['fc1000.*kernel'] = unsharded_layout_2d
    layout_map['fc1000.*bias'] = unsharded_layout_1d
    layout_map['bn.*beta'] = unsharded_layout_1d
    layout_map['bn.*gamma'] = unsharded_layout_1d
    layout_map['bn_conv1.*beta'] = unsharded_layout_1d
    layout_map['bn_conv1.*gamma'] = unsharded_layout_1d
    layout_map['bn.*moving_variance'] = unsharded_layout_1d
    layout_map['bn.*moving_mean'] = unsharded_layout_1d
    layout_map['bn_conv1.*moving_variance'] = unsharded_layout_1d
    layout_map['bn_conv1.*moving_mean'] = unsharded_layout_1d

    with tf.keras.dtensor.experimental.layout_map_scope(layout_map):
        input_shape = (224, 224, 3)
        img_input = layers.Input(shape=input_shape, batch_size=batch_size)
        if rescale_inputs:
            # Hub image modules expect inputs in the range [0, 1]. This rescales these
            # inputs to the range expected by the trained model.
            x = layers.Lambda(
                lambda x: x * 255.0 - backend.constant(
                    image_processing.CHANNEL_MEANS,
                    shape=[1, 1, 3],
                    dtype=x.dtype),
                name='rescale')(
                img_input)
        else:
            x = img_input

        if backend.image_data_format() == 'channels_first':
            x = layers.Lambda(
                lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
                name='transpose')(x)
            bn_axis = 1
        else:  # channels_last
            bn_axis = 3

        x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
        x = layers.Conv2D(
            64, (7, 7),
            strides=(2, 2),
            padding='valid',
            use_bias=False,
            kernel_initializer='he_normal',
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name='conv1')(
            x)
        x = layers.BatchNormalization(
            axis=bn_axis,
            momentum=BATCH_NORM_DECAY,
            epsilon=BATCH_NORM_EPSILON,
            fused=False,
            name='bn_conv1')(
            x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

        x = conv_block(
            x,
            3, [64, 64, 256],
            stage=2,
            block='a',
            strides=(1, 1),
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [64, 64, 256],
            stage=2,
            block='b',
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [64, 64, 256],
            stage=2,
            block='c',
            use_l2_regularizer=use_l2_regularizer)

        x = conv_block(
            x,
            3, [128, 128, 512],
            stage=3,
            block='a',
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [128, 128, 512],
            stage=3,
            block='b',
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [128, 128, 512],
            stage=3,
            block='c',
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [128, 128, 512],
            stage=3,
            block='d',
            use_l2_regularizer=use_l2_regularizer)

        x = conv_block(
            x,
            3, [256, 256, 1024],
            stage=4,
            block='a',
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [256, 256, 1024],
            stage=4,
            block='b',
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [256, 256, 1024],
            stage=4,
            block='c',
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [256, 256, 1024],
            stage=4,
            block='d',
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [256, 256, 1024],
            stage=4,
            block='e',
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [256, 256, 1024],
            stage=4,
            block='f',
            use_l2_regularizer=use_l2_regularizer)

        x = conv_block(
            x,
            3, [512, 512, 2048],
            stage=5,
            block='a',
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [512, 512, 2048],
            stage=5,
            block='b',
            use_l2_regularizer=use_l2_regularizer)
        x = identity_block(
            x,
            3, [512, 512, 2048],
            stage=5,
            block='c',
            use_l2_regularizer=use_l2_regularizer)

        rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
        x = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x)
        x = layers.Dense(
            num_classes,
            kernel_initializer=initializers.RandomNormal(stddev=0.01),
            kernel_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            bias_regularizer=_gen_l2_regularizer(use_l2_regularizer),
            name='fc1000')(
            x)

        # A softmax that is followed by the model loss must be done cannot be done
        # in float16 due to numeric issues. So we pass dtype=float32.
        x = layers.Activation('softmax', dtype='float32')(x)

        # Create model.
        ret = models.Model(img_input, x, name='resnet50')

    return ret


def image_set(filenames, batch_size, height, width, training=False,
              distort_color=False, num_threads=10, nsummary=10,
              deterministic=False, use_dali=None, idx_filenames=None):
    if use_dali:
        if idx_filenames is None:
            raise ValueError("Must provide idx_filenames if Dali is enabled")

        preprocessor = DALIPreprocessor(
            filenames,
            idx_filenames,
            height, width,
            batch_size,
            num_threads,
            dali_cpu=True if use_dali == 'CPU' else False,
            deterministic=deterministic, training=training)
        return preprocessor
    else:
        shuffle_buffer_size = 10000
        num_readers = 10
        ds = tf.data.Dataset.from_tensor_slices(filenames)

        # AUTOTUNE can give better perf for non-horovod cases
        thread_config = num_threads

        # shard should be before any randomizing operations

        # disabled this for DTensor
        # if training:
        #  ds = ds.shard(hvd.size(), hvd.rank())

        # read up to num_readers files and interleave their records
        ds = ds.interleave(
            tf.data.TFRecordDataset, cycle_length=num_readers)

        if training:
            # Improve training performance when training data is in remote storage and
            # can fit into worker memory.
            ds = ds.cache()

        if training:
            # shuffle data before repeating to respect epoch boundaries
            ds = ds.shuffle(shuffle_buffer_size)
            ds = ds.repeat()

        preproc_func = (lambda record:
                        _parse_and_preprocess_image_record(record, height, width,
                                                           deterministic=deterministic, random_crop=training,
                                                           distort_color=distort_color))
        ds = ds.map(preproc_func,
                    num_parallel_calls=thread_config)

        ds = ds.batch(batch_size, drop_remainder=True)

        # prefetching
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        options = tf.data.Options()
        options.experimental_slack = True
        ds = ds.with_options(options)

        return ds


model = resnet50(NUM_CLASSES)


num_epochs = 3
image_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=4)
label_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=1)

dali_mode=False
image_width=224
image_height=224
distort_color=False
train_idx_files = None
valid_idx_files = None


def parse_cmdline(init_vals):
  f = argparse.ArgumentDefaultsHelpFormatter
  p = argparse.ArgumentParser(formatter_class=f)

  p.add_argument('--data_dir',
                 default=init_vals.get('data_dir'),
                 required=False,
                 help="""Path to dataset in TFRecord format (aka Example
                 protobufs). Files should be named 'train-*' and
                 'validation-*'.""")

  vals = init_vals
  vals['data_dir'] = FLAGS.data_dir

  return vals

default_args = {
    'data_dir' : None,
}
args = parse_cmdline(default_args)
data_dir = args['data_dir']

file_format = os.path.join(data_dir, '%s-*')
train_files = sorted(tf.io.gfile.glob(file_format % 'train'))
valid_files = sorted(tf.io.gfile.glob(file_format % 'validation'))

num_preproc_threads = 4 if dali_mode else 10
train_input = image_set(train_files, batch_size,
        image_height, image_width, training=True, distort_color=distort_color,
        deterministic=False, num_threads=num_preproc_threads,
        use_dali=dali_mode, idx_filenames=train_idx_files)

valid_input = image_set(valid_files, batch_size,
        image_height, image_width, training=False, distort_color=False,
        deterministic=False, num_threads=num_preproc_threads,
        use_dali=dali_mode, idx_filenames=valid_idx_files)

for epoch in range(num_epochs):
  print("============================")
  print("Epoch: ", epoch)
  for metric in metrics.values():
    metric.reset_state()
  step = 0
  results = {}
  pbar = tf.keras.utils.Progbar(target=None, stateful_metrics=[])
  train_iter = iter(train_input)
  nstep_per_epoch = 100
  for _ in nstep_per_epoch:
    x = next(train_iter)
    images, labels = x
    images, labels = pack_dtensor_inputs(
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
  nstep_per_valid = 10
  for _ in nstep_per_valid:
    y = next(valid_input)
    images, labels = y
    images, labels = pack_dtensor_inputs(
        images, labels, image_layout, label_layout)
    results.update(eval_step(model, images, labels, eval_metrics))

  for metric_name, metric in eval_metrics.items():
    results[metric_name] = metric.result()

  for metric_name, metric in results.items():
    print(f"{metric_name}: {metric.numpy()}")
