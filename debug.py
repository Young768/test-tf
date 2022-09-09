import tensorflow as tf
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

@tf.function
def step(tensor):
    out = tf.multiply(tensor, 2.0)
    out = tf.multiply(out, 2.0)
    out = tf.multiply(out, 2.0)
    out = tf.multiply(out, 2.0)
    out = tf.multiply(out, 2.0)
    out = tf.multiply(out, 2.0)
    return out

# input_tensor = tf.Variable(initial_value=1.0)
for i in range(40):
    input_tensor = tf.Variable(initial_value=1.0)
    output_tensor = step(input_tensor)
    print("{}".format(i), output_tensor)
    # input_tensor.assign_add(output_tensor[1])
