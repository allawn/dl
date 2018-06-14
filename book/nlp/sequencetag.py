import  tensorflow as tf

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
print(outputs)
outputs=tf.reshape(outputs,[-1,2*num_units])
print(outputs)
weight = tf.truncated_normal([2*num_units, tag_class_num], stddev=0.01)
bias = tf.constant(0.1, shape=[tag_class_num])
logist=tf.matmul(outputs, weight) + bias
print(logist)
prediction = tf.nn.softmax(logist)
print(prediction)
prediction = tf.reshape(prediction, [batch_size, timesteps, tag_class_num])

#method1 for compute loss
cross_entropy = tf.cast(label_data,tf.float32) * tf.log(prediction)
cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
mask = tf.sign(tf.reduce_max(tf.abs(label_data), reduction_indices=2))
cross_entropy *= tf.cast(mask,tf.float32)
cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
cross_entropy /= tf.cast(words_used_num, tf.float32)

#method2 for compute loss
label_data2 = tf.placeholder(tf.int32, [batch_size, timesteps])
label_data2=tf.reshape(label_data2,[-1])
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logist, labels=label_data2)
print(losses)
mask2 = tf.sequence_mask(sequence_length)
mask2=tf.reshape(mask2,[-1])
print(mask2)
# apply mask
losses = tf.boolean_mask(losses, mask2)
loss = tf.reduce_mean(losses)

# tf.boolean_mask(logist,)
