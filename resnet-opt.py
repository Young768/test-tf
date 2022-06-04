import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.experimental import dtensor
from tensorflow.keras import backend
from tensorflow.keras import regularizers
from tensorflow.keras import initializers
from tensorflow.keras import models
import os
import sys
import time
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

tf.keras.backend.experimental.enable_tf_random_generator()
tf.keras.utils.set_random_seed(1337)


opt = tf.keras.dtensor.experimental.optimizers.SGD(0.01, mesh=mesh)
loss_func = tf.keras.losses.SparseCategoricalCrossentropy()
train_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1,
                                                              name='train_top1')
train_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,
                                                              name='train_top5')
val_loss = tf.keras.metrics.Mean(name='val_loss', dtype=tf.float32)
val_top1 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1,
                                                            name='val_top1')
val_top5 = tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,
                                                            name='val_top5')

layers = tf.keras.layers

L2_WEIGHT_DECAY = 1e-4
BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


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

def _decode_jpeg(imgdata, channels=3):
  return tf.image.decode_jpeg(imgdata, channels=channels,
                              fancy_upscaling=False,
                              dct_method='INTEGER_FAST')

def _crop_and_resize_image(image, original_bbox, height, width, deterministic=False, random_crop=False):
  with tf.name_scope('random_crop_and_resize'):
    eval_crop_ratio = 0.8
    if random_crop:
      bbox_begin, bbox_size, bbox = \
          tf.image.sample_distorted_bounding_box(
              tf.shape(image),
              bounding_boxes=tf.zeros(shape=[1,0,4]), # No bounding boxes
              min_object_covered=0.1,
              aspect_ratio_range=[0.8, 1.25],
              area_range=[0.1, 1.0],
              max_attempts=100,
              seed=7 * (1+hvd.rank()) if deterministic else 0,
              use_image_if_no_bounding_boxes=True)
      image = tf.slice(image, bbox_begin, bbox_size)
    else:
      # Central crop
      image = tf.image.central_crop(image, eval_crop_ratio)
    image = tf.compat.v1.image.resize_images(
        image,
        [height, width],
        tf.image.ResizeMethod.BILINEAR,
        align_corners=False)
    image.set_shape([height, width, 3])
    return image

def _distort_image_color(image, order=0):
  with tf.name_scope('distort_color'):
    image = tf.math.multiply(image, 1. / 255.)
    brightness = lambda img: tf.image.random_brightness(img, max_delta=32. / 255.)
    saturation = lambda img: tf.image.random_saturation(img, lower=0.5, upper=1.5)
    hue        = lambda img: tf.image.random_hue(img, max_delta=0.2)
    contrast   = lambda img: tf.image.random_contrast(img, lower=0.5, upper=1.5)
    if order == 0: ops = [brightness, saturation, hue, contrast]
    else:          ops = [brightness, contrast, saturation, hue]
    for op in ops:
      image = op(image)
    # The random_* ops do not necessarily clamp the output range
    image = tf.clip_by_value(image, 0.0, 1.0)
    # Restore the original scaling
    image = tf.multiply(image, 255.)
    return image

def _deserialize_image_record(record):
  feature_map = {
      'image/encoded':          tf.io.FixedLenFeature([ ], tf.string, ''),
      'image/class/label':      tf.io.FixedLenFeature([1], tf.int64,  -1),
      'image/class/text':       tf.io.FixedLenFeature([ ], tf.string, ''),
      'image/object/bbox/xmin': tf.io.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymin': tf.io.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/xmax': tf.io.VarLenFeature(dtype=tf.float32),
      'image/object/bbox/ymax': tf.io.VarLenFeature(dtype=tf.float32)
  }
  with tf.name_scope('deserialize_image_record'):
    obj = tf.io.parse_single_example(record, feature_map)
    imgdata = obj['image/encoded']
    label   = tf.cast(obj['image/class/label'], tf.int32)
    bbox    = tf.stack([obj['image/object/bbox/%s'%x].values
                        for x in ['ymin', 'xmin', 'ymax', 'xmax']])
    bbox = tf.transpose(tf.expand_dims(bbox, 0), [0,2,1])
    text    = obj['image/class/text']
    return imgdata, label, bbox, text

def _parse_and_preprocess_image_record(record, height, width,
                                       deterministic=False, random_crop=False,
                                       distort_color=False):
  imgdata, label, bbox, text = _deserialize_image_record(record)
  label -= 1 # Change to 0-based (don't use background class)
  with tf.name_scope('preprocess_train'):
    try:    image = _decode_jpeg(imgdata, channels=3)
    except: image = tf.image.decode_png(imgdata, channels=3)

    image = _crop_and_resize_image(image, bbox, height, width, deterministic, random_crop)

    # image comes out of crop as float32, which is what distort_color expects
    if distort_color:
      image = _distort_image_color(image)
    image = tf.cast(image, tf.float32)
    if random_crop:
      image = tf.image.random_flip_left_right(image,
                    seed=11 * (1 + hvd.rank()) if deterministic else None)
    return image, label


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
  p.add_argument('-i', '--num_iter', type=int,
                 default=init_vals.get('num_iter'),
                 required=False,
                 help="""Number of batches or epochs to run.""")
  p.add_argument('-b', '--batch_size', type=int,
                 default=init_vals.get('batch_size'),
                 required=False,
                 help="""Size of each minibatch.""")
  p.add_argument('--precision', choices=['fp32', 'fp16'],
                 default=init_vals.get('precision'),
                 required=False,
                 help="""Select single or half precision arithmetic.""")

  FLAGS, unknown_args = p.parse_known_args()

  vals = init_vals
  vals['data_dir'] = FLAGS.data_dir
  vals['num_iter'] = FLAGS.num_iter
  vals['batch_size'] = FLAGS.batch_size
  vals['precision'] = FLAGS.precision

  return vals

default_args = {
    'data_dir' : None,
    'num_iter' : 300,
    'batch_size' : 128,
    'precision' : 'fp32',
}


args = parse_cmdline(default_args)
data_dir = args['data_dir']
num_epochs = args['num_iter']
batch_size = args['batch_size']
precision = args['precision']
loss_scale = 128.0
nstep_per_epoch = num_epochs

if precision == 'fp16':
    policy = tf.keras.experimental.mixed_precision.Policy('mixed_float16')
    tf.keras.experimental.mixed_precision.set_global_policy(policy)

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


'''training loop'''

global_steps = 0
log_steps = 10
image_format='channels_last'

backend.set_image_data_format(image_format)
NUM_CLASSES = 1000
model = resnet50(NUM_CLASSES)

image_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=4)
label_layout = dtensor.Layout.batch_sharded(mesh, 'batch', rank=2)


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


for epoch in range(num_epochs):
  print("============================")
  print("Epoch: ", epoch)
  total_loss = 0.0
  num_batches = 0
  #train_top1.reset_states()
  #train_top5.reset_states()
  epoch_start = time.time()
  train_iter = iter(train_input)
  valid_iter = iter(valid_input)
  for _ in range(nstep_per_epoch):
    global_steps += 1
    if global_steps == 1:
        start_time = time.time()
    x = next(train_iter)
    images, labels = x
    images, labels = pack_dtensor_inputs(
        images, labels, image_layout, label_layout)
    t_x = (images, labels)
    total_loss += train_step(t_x)

    if global_steps % log_steps == 0:
        timestamp = time.time()
        elapsed_time = timestamp - start_time
        examples_per_second = \
            (batch_size * log_steps) / elapsed_time
        print("global_step: %d images_per_sec: %.1f" % (global_steps,
                                                        examples_per_second))
        start_time = timestamp
    num_batches += 1

  train_loss = total_loss / num_batches

  epoch_run_time = time.time() - epoch_start
  print("epoch: %d time_taken: %.1f" % (epoch, epoch_run_time))

  y = next(valid_iter)
  y_images, y_labels = y
  y_images, y_labels = pack_dtensor_inputs(
      y_images, y_labels, image_layout, label_layout)
  y_x = (y_images, y_labels)
  valid_step(y_x)
