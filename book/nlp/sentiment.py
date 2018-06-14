import  tensorflow as tf
import numpy as np

num_units=100
num_layers=2
dropout=0.5

timesteps=5;
batch_size=2
tag_class_num=10
inputs=tf.placeholder(tf.float32, [batch_size, timesteps, 1], name='input_x')
#one hot label
label_data = tf.placeholder(tf.int32, [batch_size, timesteps, tag_class_num])
sequence_length=[2,5]

words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(inputs), reduction_indices=2))
words_used_num = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)

cell = tf.nn.rnn_cell.BasicRNNCell
cells_fw = [cell(num_units) for _ in range(num_layers)]
cells_bw = [cell(num_units) for _ in range(num_layers)]
if dropout > 0.0:
    cells_fw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_fw]
    cells_bw = [tf.contrib.rnn.DropoutWrapper(cell) for cell in cells_bw]
outputs, _, _ = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(
    cells_fw=cells_fw,
    cells_bw=cells_bw,
    inputs=inputs,
    sequence_length=sequence_length,
    dtype=tf.float32,
    scope="birnn")

def last_relevant(output, sequence_length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    one=np.ones(shape=[len(sequence_length)])
    index = tf.range(0, batch_size) * max_length + (sequence_length - one)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.gather(flat, index)
    return relevant

lastOut=last_relevant(outputs,sequence_length)
print(lastOut)