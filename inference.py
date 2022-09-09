import tensorflow as tf

inputs = tf.keras.Input((28, 28, 1))
x = tf.keras.layers.BatchNormalization()(inputs)
x = tf.keras.layers.BatchNormalization()(x)
output = tf.keras.layers.BatchNormalization()(x)
model = tf.keras.Model(inputs, output)

@tf.function
def foo(inputs):
  return model(inputs)

inp = tf.constant(value=1.0, shape=(1, 28, 28, 1))
for i in range(40):
    output_tensor = foo(inp)
    print("output:", output_tensor)