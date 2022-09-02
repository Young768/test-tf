import tensorflow as tf

inputs = tf.keras.Input((28, 28, 1))
output = tf.keras.layers.BatchNormalization()(inputs)
model = tf.keras.Model(inputs, output)

@tf.function
def foo(inputs):
  return model(inputs)

inp = tf.constant(value=1.0, shape=(1, 28, 28, 1))
for i in range(40):
    output_tensor = foo(inp)
print("Done.")