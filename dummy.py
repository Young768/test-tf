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



@tf.function
def step(tensor):
    out = tf.add(tensor, 1.0)
    out = tf.multiply(out, 2.0)
    out = tf.add(out, 2.0)
    out = tf.multiply(out, 2.0)
    out = tf.multiply(out, 2.0)
    return out

input = tf.constant(value=1)

num_epochs = 10

for epoch in range(num_epochs):
  output = step(input)
