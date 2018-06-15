import  tensorflow as tf
from tensorflow.contrib.crf import crf_log_likelihood
from tensorflow.contrib.crf import viterbi_decode
from tensorflow.contrib.crf import crf_decode
import numpy as np

num_units=100
num_layers=2
dropout=0.5

timesteps=5;
batch_size=2
tag_class_num=10
inputs=tf.placeholder(tf.float32, [batch_size, timesteps, 1], name='input_x')
#one hot label
label_data = tf.placeholder(tf.int32, [batch_size, timesteps])
sequence_length=[2,5]
sequence_length=tf.constant(sequence_length,tf.int32)
print(sequence_length)

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
print(outputs)
outputs=tf.reshape(outputs,[-1,2*num_units])
print(outputs)
weight = tf.truncated_normal([2*num_units, tag_class_num], stddev=0.01)
bias = tf.constant(0.1, shape=[tag_class_num])
logist=tf.matmul(outputs, weight) + bias
logits = tf.reshape(logist, [batch_size, timesteps, tag_class_num])

log_likelihood,transition_params=crf_log_likelihood(inputs=logits,tag_indices=label_data,sequence_lengths=sequence_length)
loss = -tf.reduce_mean(log_likelihood)

decode_tags,best_score=crf_decode(logits,transition_params,sequence_length)
correct = tf.equal(decode_tags, label_data)
used = tf.sign(tf.abs(label_data))
used = tf.cast(used, tf.bool)
correct = tf.reduce_sum(tf.cast(tf.logical_and(correct, used), tf.int32))
accuracy = tf.cast(correct, tf.float32)/tf.cast(tf.reduce_sum(sequence_length), tf.float32, name='accuracy')
print(accuracy)