import tensorflow as tf

inputs = tf.keras.Input((28, 28, 1))
x = tf.keras.layers.BatchNormalization()(inputs)
x = tf.keras.layers.BatchNormalization()(x)
output = tf.keras.layers.BatchNormalization()(x)
model = tf.keras.Model(inputs, output)

def step_1(tensor):
    with tf.GradientTape() as tape:
        out = model(tensor)
    gradients = tape.gradient(out, tensor)
    return gradients

inp = tf.constant(value=1.0, shape=(1, 28, 28, 1))
for i in range(40):
    output_tensor = step_1(inp)
print("Done.")