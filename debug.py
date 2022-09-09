import tensorflow as tf
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"


inputs = tf.keras.Input((28, 28, 1))
x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(inputs)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
output = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
model = tf.keras.Model(inputs, output)
@tf.function
def step(tensor):
    output = model(tensor)
    return output


for i in range(40):
    inp = tf.constant(value=1.0, shape=(1, 28, 28, 1))
    output_tensor = step(inp)
    if i == 0:
        prev = output_tensor
    if i > 0:
        tf.debugging.assert_near(output_tensor, prev)
        prev = output_tensor
    #print("output: {}".format(i), output_tensor)
    # input_tensor.assign_add(output_tensor[1])
