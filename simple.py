import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

model = keras.models.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    #keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3)),
    keras.layers.BatchNormalization(),
    #keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    #keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
    #keras.layers.BatchNormalization(),
    #keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    #keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    #keras.layers.BatchNormalization(),
    #keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    #keras.layers.BatchNormalization(),
    #keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
    #keras.layers.BatchNormalization(),
    #keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    #keras.layers.Flatten(),
    #keras.layers.Dense(4096, activation='relu'),
    #keras.layers.Dropout(0.5),
    #keras.layers.Dense(4096, activation='relu'),
    #keras.layers.Dropout(0.5),
    #keras.layers.Dense(10, activation='softmax')
])

@tf.function
def step(tensor):
    output = model(tensor)
    return output

for i in range(40):
    inp = tf.constant(value=1.0, shape=(1, 227,227,3))
    output_tensor = step(inp)
    if i == 0:
        prev = output_tensor
    if i > 0:
        tf.debugging.assert_near(output_tensor, prev)
        prev = output_tensor

    #print("output: {}".format(i), output_tensor)
    # input_tensor.assign_add(output_tensor[1])
