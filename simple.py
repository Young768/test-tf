import tensorflow as tf

@tf.function
def step(tensor):
    out = tf.add(tensor, 1.0)
    out = tf.multiply(out, 2.0)
    return out

@tf.function
def step_1(tensor):
    with tf.GradientTape() as tape:
        out = step(tensor) * 10
    grads = tape.gradient(out, tensor)
    tensor.assign_add(grads)
    return out, grads

input_tensor = tf.Variable(initial_value=1.0)
for i in range(4):
    # input_tensor = tf.constant(i, dtype=tf.float32)
    output_tensor = step_1(input_tensor)
    print("{}".format(i), output_tensor)
