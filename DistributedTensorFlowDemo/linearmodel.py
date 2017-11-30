import tensorflow as tf

def inference(x):
    W = tf.Variable([.3], dtype=tf.float32)
    b = tf.Variable([-.3], dtype=tf.float32)
    logist = W*x + b
    return logist,W,b

def loss(logist,y):
    loss = tf.reduce_sum(tf.square(logist - y))
    return loss