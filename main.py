import tensorflow as tf

hello = tf.constant("Hello world")

output = hello.numpy()
print(output.decode())