import tensorflow as tf

x = tf.constant(2)
y1 = x + 300
y2 = x - 66
y = y1 + y2

with tf.Session() as sess:
    result = sess.run(y)
    print(result)
