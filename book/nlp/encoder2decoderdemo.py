import tensorflow as tf
from tensorflow.contrib.seq2seq import *

max_encoder_time=10
max_decoder_time=10
batch_size=2
src_vocab_size=100
target_vocab_size=60
embedding_size=20
num_units=30

encoder_inputs=tf.placeholder(shape=[max_encoder_time, batch_size],dtype=tf.int32)
embedding_encoder = tf.get_variable("embedding_encoder", [src_vocab_size, embedding_size],dtype=tf.float32)
encoder_emb_inp = tf.nn.embedding_lookup( embedding_encoder, encoder_inputs)
source_sequence_length=tf.placeholder(shape=[batch_size],dtype=tf.int32)
print(encoder_emb_inp)

encoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_emb_inp,
    sequence_length=source_sequence_length, time_major=True,dtype=tf.float32)

decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
decoder_inputs=tf.placeholder(dtype=tf.int32,shape=[max_decoder_time, batch_size])
embedding_decoder = tf.get_variable("embedding_decoder", [target_vocab_size, embedding_size],dtype=tf.float32)
decoder_emb_inp=tf.nn.embedding_lookup(embedding_decoder,decoder_inputs)
decoder_lengths=tf.placeholder(shape=[batch_size],dtype=tf.int32)
helper = TrainingHelper( decoder_emb_inp, decoder_lengths, time_major=True)
projection_layer = tf.layers.Dense(target_vocab_size, use_bias=False)

decoder = BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)

print(encoder_outputs)

outputs, final_state, final_sequence_lengths= dynamic_decode(decoder)
print(outputs)
print(final_state)
print(final_sequence_lengths)
logits = outputs.rnn_output
print(logits)

#inference
helper2 = GreedyEmbeddingHelper(embedding_decoder, tf.fill([batch_size], 0), 1)
decoder2 = tf.contrib.seq2seq.BasicDecoder(decoder_cell, helper, encoder_state, output_layer=projection_layer)
maximum_iterations = tf.round(tf.reduce_max(source_sequence_length) * 2)
outputs, _,_ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=maximum_iterations)
translations = outputs.sample_id
print(translations)
