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
    #keras.layers.BatchNormalization()
])

class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_train_batch_end(self, batch, logs=None):
        for i in range(len(model_inception.layers)):
            get_layer_output = K.function(inputs=self.model.layers[i].input, outputs=self.model.layers[i].output)
            print('\n Training: output of the layer {} is {} ends at {}'.format(i, get_layer_output.outputs, datetime.datetime.now().time()))

@tf.function
def step(tensor):
    #output = model(tensor)
    output = model.evaluate(tensor)
    return output

optimizer = tf.keras.optimizers.SGD()
model.compile(optimizer)
for i in range(2):
    inp = tf.constant(value=1.0, shape=(1, 28, 28, 1))
    dataset = tf.data.Dataset.from_tensor_slices(inp).repeat().batch(1)
    #output_tensor = step(dataset)
    output_tensor = model.predict(dataset, steps=40)
    if i == 0:
        prev = output_tensor
    if i > 0:
        tf.debugging.assert_near(output_tensor, prev)
        prev = output_tensor

    #print("output: {}".format(i), output_tensor)
    # input_tensor.assign_add(output_tensor[1])
