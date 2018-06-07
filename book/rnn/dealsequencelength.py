import tensorflow as tf

def length(sequence):
    used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
    length = tf.reduce_sum(used, 1)
    length = tf.cast(length, tf.int32)
    return length


max_length = 100
frame_size = 64
num_hidden = 200

sequence = tf.placeholder(
    tf.float32, [None, max_length, frame_size])
output, state = tf.nn.dynamic_rnn(
    tf.contrib.rnn.GRUCell(num_hidden),
    sequence,
    dtype=tf.float32,
    sequence_length=length(sequence),
)

def cost(output, target):
    # Compute cross entropy for each frame.
    cross_entropy = target * tf.log(output)
    cross_entropy = -tf.reduce_sum(cross_entropy, 2)
    mask = tf.sign(tf.reduce_max(tf.abs(target), 2))
    cross_entropy *= mask
    # Average over actual sequence lengths.
    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

num_classes = 10

last = last_relevant(output)
weight = tf.Variable(
    tf.truncated_normal([num_hidden, num_classes], stddev=0.1))
bias = tf.Variable(tf.constant(0.1, shape=[num_classes]))
prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)

