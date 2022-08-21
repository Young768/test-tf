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



inputs = tf.keras.Input((28, 28, 1))
x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(inputs)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
output = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
model = tf.keras.Model(inputs, output)

@tf.function
def foo(inputs):
  return model(inputs)

input = tf.constant(value=1, shape=(1, 28, 28, 1))

num_epochs = 10

for epoch in range(num_epochs):
  out = foo(input)
