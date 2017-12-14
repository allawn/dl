import tensorflow as tf


cluster = tf.train.ClusterSpec({"local": ["localhost:2222", "localhost:2223"]})

x = tf.constant(2)


with tf.device("/job:local/task:1"):
    y2 = x - 66

with tf.device("/job:local/task:0"):
    y1 = x + 300
    y = y1 + y2


with tf.Session("grpc://localhost:2222") as sess:
    result = sess.run(y)
    print(result)