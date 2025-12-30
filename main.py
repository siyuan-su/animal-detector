import tensorflow as tf
tensor = tf.constant("Michael Dont Leave Me Here")
print(tensor.numpy().decode())

hello = tf.constant("Hello world")

output = hello.numpy()
print(output.decode())