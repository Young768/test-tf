import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

model = keras.models.Sequential([
    keras.layers.MaxPooling2D((2, 2), name='pool1'),
    keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    keras.layers.MaxPooling2D((2, 2), name='pool2'),
    #keras.layers.BatchNormalization()
])

@tf.function
def step(tensor):
    output = model(tensor)
    d1 = model.layers[1]

    #output = model.evaluate(tensor)
    return d1, output

for i in range(40):
    inp = tf.constant(value=1.0, shape=(1, 28, 28, 1))
    #dataset = tf.data.Dataset.from_tensor_slices(inp).repeat().batch(1)
    d, output_tensor = step(inp)
    print(d)
    if i == 0:
        prev = output_tensor
    if i > 0:
        tf.debugging.assert_near(output_tensor, prev)
        prev = output_tensor

    #print("output: {}".format(i), output_tensor)
    # input_tensor.assign_add(output_tensor[1])
