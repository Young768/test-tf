import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

model = tf.keras.Sequential([
    # layer 1
    tf.keras.layers.Conv2D(filters=96,
                            kernel_size=(11, 11),
                            strides=4,
                            padding="valid",
                            activation=tf.keras.activations.relu,
                            input_shape=(227, 227, 3)),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                strides=2,
                                padding="valid"),
    tf.keras.layers.BatchNormalization(),
    # layer 2
    tf.keras.layers.Conv2D(filters=256,
                            kernel_size=(5, 5),
                            strides=1,
                            padding="same",
                            activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                strides=2,
                                padding="same"),
    tf.keras.layers.BatchNormalization(),
    # layer 3
    tf.keras.layers.Conv2D(filters=384,
                            kernel_size=(3, 3),
                            strides=1,
                            padding="same",
                            activation=tf.keras.activations.relu),
    # layer 4
    tf.keras.layers.Conv2D(filters=384,
                            kernel_size=(3, 3),
                            strides=1,
                            padding="same",
                            activation=tf.keras.activations.relu),
    # layer 5
    tf.keras.layers.Conv2D(filters=256,
                            kernel_size=(3, 3),
                            strides=1,
                            padding="same",
                            activation=tf.keras.activations.relu),
    tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                strides=2,
                                padding="same"),
    tf.keras.layers.BatchNormalization(),
    # layer 6
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=4096,
                            activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(rate=0.2),
    # layer 7
    tf.keras.layers.Dense(units=4096,
                            activation=tf.keras.activations.relu),
    tf.keras.layers.Dropout(rate=0.2),
    # layer 8
    tf.keras.layers.Dense(units=10,
                            activation=tf.keras.activations.softmax)
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
