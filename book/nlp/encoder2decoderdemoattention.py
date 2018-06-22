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

attention_states = tf.transpose(encoder_outputs, [1, 0, 2])
attention_mechanism = LuongAttention(
    num_units, attention_states,
    memory_sequence_length=source_sequence_length)


decoder_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
decoder_cell = AttentionWrapper(decoder_cell, attention_mechanism, attention_layer_size=num_units)

decoder_inputs=tf.placeholder(dtype=tf.int32,shape=[max_decoder_time, batch_size])
embedding_decoder = tf.get_variable("embedding_decoder", [target_vocab_size, embedding_size],dtype=tf.float32)
decoder_emb_inp=tf.nn.embedding_lookup(embedding_decoder,decoder_inputs)
decoder_lengths=tf.placeholder(shape=[batch_size],dtype=tf.int32)
helper = TrainingHelper( decoder_emb_inp, decoder_lengths, time_major=True)
projection_layer = tf.layers.Dense(target_vocab_size, use_bias=False)

decoder = BasicDecoder(decoder_cell, helper, decoder_cell.zero_state(batch_size,tf.float32).clone(cell_state=encoder_state), output_layer=projection_layer)

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


#beam search inference
# BEAM SEARCH TILE
BEAM_WIDTH=5
start_tokens = tf.fill([batch_size], 0)
end_token = 1

encoder_outputs=tf.transpose(encoder_outputs, [1, 0, 2])
encoder_out = tile_batch(encoder_outputs, multiplier=BEAM_WIDTH)
X_seq_len = tile_batch(source_sequence_length, multiplier=BEAM_WIDTH)
encoder_state = tile_batch(encoder_state, multiplier=BEAM_WIDTH)

# ATTENTION (PREDICTING)
attention_mechanism = LuongAttention(
    num_units = num_units,
    memory = encoder_out,
    memory_sequence_length = X_seq_len)

decoder_cell = tf.contrib.seq2seq.AttentionWrapper(
    cell = tf.nn.rnn_cell.BasicLSTMCell(num_units),
    attention_mechanism = attention_mechanism,
    attention_layer_size = num_units)

# DECODER (PREDICTING)
predicting_decoder = BeamSearchDecoder(
    cell = decoder_cell,
    embedding = embedding_decoder,
    start_tokens = tf.tile(tf.constant([1], dtype=tf.int32), [batch_size]),
    end_token = 2,
    initial_state = decoder_cell.zero_state(batch_size * BEAM_WIDTH,tf.float32).clone(cell_state=encoder_state),
    beam_width = BEAM_WIDTH,
    output_layer = projection_layer,
    length_penalty_weight = 0.0)
predicting_decoder_output, _, _ = dynamic_decode(
    decoder = predicting_decoder,
    impute_finished = False,
    maximum_iterations = 2 * tf.reduce_max(source_sequence_length))

