import tensorflow as tf
import numpy as np

vocab_size=1000
embeddingsize=128
cellHiddenSize=200
batch_size=32
time_step=5
inputWords=tf.placeholder(shape=[batch_size,time_step],dtype=tf.int32)
labelWords=tf.placeholder(shape=[batch_size,time_step],dtype=tf.int32)
embedding = tf.get_variable("embedding", [vocab_size, embeddingsize], dtype=tf.float32)
inputs = tf.nn.embedding_lookup(embedding, inputWords)

cell=tf.nn.rnn_cell.BasicLSTMCell(cellHiddenSize)
outputs,finalStatus=tf.nn.dynamic_rnn(inputs=inputs,cell=cell,dtype=tf.float32)
output = tf.reshape(outputs, [-1, cellHiddenSize])

softmax_w = tf.get_variable("softmax_w", [cellHiddenSize, vocab_size], dtype=tf.float32)
softmax_b = tf.get_variable("softmax_b", [vocab_size], dtype=tf.float32)
logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)
# Reshape logits to be a 3-D tensor for sequence loss
logits = tf.reshape(logits, [batch_size, time_step, vocab_size])

# Use the contrib sequence loss and average over the batches
loss = tf.contrib.seq2seq.sequence_loss(
    logits,
    labelWords,
    tf.ones([batch_size, time_step], dtype=tf.float32),
    average_across_timesteps=False,
    average_across_batch=True)

_cost = tf.reduce_sum(loss)
#output perplexity
np.exp(_cost / time_step)