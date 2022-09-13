import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import time
from keras.callbacks import Callback

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

model = keras.models.Sequential([
    keras.layers.MaxPooling2D((2, 2), name='pool1'),
    keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2'),
    keras.layers.MaxPooling2D((2, 2), name='pool2'),
    keras.layers.BatchNormalization()
])

optimizer = tf.keras.optimizers.SGD()
model.compile(optimizer)
test_layer = model.layers[3]
tf.print("debugggggggggggggggggg",test_layer)
class CustomCallback(tf.keras.callbacks.Callback):
   def __init__(self, data):
        self.data = data
   def on_predict_batch_end(self, epoch, logs=None):
        out = model.layers[0](self.data)
        out = model.layers[1](out)
        out = model.layers[2](out)
        encoder_outputs = test_layer[3](out)
        tf.print('output --> ', encoder_outputs)

@tf.function
def step(tensor):
    output = model.evaluate(tensor)
    return output


for i in range(1):
    inp = tf.constant(value=1.0, shape=(1, 28, 28, 1))
    dataset = tf.data.Dataset.from_tensor_slices(inp).repeat().batch(1)
    output_tensor = model.predict(dataset, steps=40, callbacks=[CustomCallback(inp)])
    if i == 0:
        prev = output_tensor
    if i > 0:
        tf.debugging.assert_near(output_tensor, prev)
        prev = output_tensor

    #print("output: {}".format(i), output_tensor)
    # input_tensor.assign_add(output_tensor[1])
