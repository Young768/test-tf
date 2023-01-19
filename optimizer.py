import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time

(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)

train_ds = (train_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))

test_ds = (test_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))

validation_ds = (validation_ds
                  .map(process_images)
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=32, drop_remainder=True))

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

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.legacy.Adagrad(learning_rate=0.0001), metrics=['accuracy'])
#model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.experimental.Adagrad(learning_rate=0.0001), metrics=['accuracy'])
train_begin = time.time()
model.fit(train_ds,
          epochs=50,
          steps_per_epoch=20)
elapsed = time.time() - train_begin
print("Training time elasped:", elapsed)