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

rank_2_tensor = tf.constant([[1, 2],
                             [3, 4],
                             [5, 6]], dtype=tf.float16)


@tf.function
def step(tensor):
    out = tf.add(tensor, 1.0, name="Node_0")
    out = tf.add(out, 1.0, name="Node_1")
    out = tf.add(out, 1.0, name="Node_2")
    out = tf.add(out, 1.0, name="Node_3")
    out = tf.add(out, 1.0, name="Node_4")
    out = tf.add(out, 2.0, name="Node_5")
    out = tf.add(out, 2.0, name="Node_6")
    out = tf.add(out, 2.0, name="Node_7")
    out = tf.add(out, 2.0, name="Node_8")
    out = tf.add(out, 2.0, name="Node_9")
    return out

input = tf.constant(value=1.0)

num_epochs = 10

for epoch in range(num_epochs):
  output = step(rank_2_tensor)
