import tensorflow as tf
import tensorflow_datasets as tfds
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

layers = tf.keras.layers

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
  logical_gpus = tf.config.list_logical_devices('GPU')
  print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")


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

optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

tf.config.optimizer.set_experimental_options({"cudagraph": True})

@tf.function
def train_step(inputs):
    images, labels = inputs

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        #loss = loss_fn(labels, predictions)
        #loss += tf.reduce_sum(model.losses)
        #loss_copy = loss

    #gradients = tape.gradient(loss, model.trainable_variables)
    #optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    return predictions


inputs = tf.keras.Input((28, 28, 1))
x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(inputs)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
output = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
model = tf.keras.Model(inputs, output)




num_epochs = 20
num_steps = 2
data = iter(ds_train)
input = next(data)

for epoch in range(num_epochs):
  print("============================")
  print("Epoch: ", epoch)
  step = 0

  for _ in range(num_steps):
    loss_val = train_step(input)
    print(loss_val.device)
    print("done!    ")
