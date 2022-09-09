import tensorflow as tf
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

@tf.function
def step(tensor):
    inputs = tf.keras.Input((10, 1))(tensor)
    x = tf.keras.layers.MaxPooling2D((2, 2), name='pool1')(inputs)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    output = tf.keras.layers.MaxPooling2D((2, 2), name='pool2')(x)
    return output


for i in range(40):
    inp = tf.constant(value=1.0, shape=(1, 10, 1))
    output_tensor = step(inp)
    print("{}".format(i), output_tensor)
    # input_tensor.assign_add(output_tensor[1])
