import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units * 2
lstm_layers = 2

# using dynamic_rnn
def inference(encoder_inputs, embeddings_inputs, classNum, isTrainModel=True):
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_inputs, encoder_inputs)

    if isTrainModel:
        encoder_cell_fw = tf.contrib.rnn.MultiRNNCell([tf.contrib.rnn.DropoutWrapper(
            tf.contrib.rnn.BasicLSTMCell(num_units=encoder_hidden_units), output_keep_prob=0.5) for _ in
            range(lstm_layers)])
    else:
        encoder_cell_fw = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(num_units=encoder_hidden_units) for _ in range(lstm_layers)])

    encoder_outputs, final_state = tf.nn.dynamic_rnn(encoder_cell_fw, encoder_inputs_embedded, dtype=tf.float32)

    output = tf.transpose(encoder_outputs, [1, 0, 2])
    # print("------------",output)
    print(output.get_shape()[0])
    last = tf.gather(output, tf.shape(output)[0] - 1)
    print(last)
    # xxx

    after_dp = tf.layers.dropout(encoder_outputs, rate=0.5, training=isTrainModel)
    output_logits = tf.contrib.layers.fully_connected(last, num_outputs=classNum, activation_fn=tf.identity)
    return output_logits


# using bidirectional_dynamic_rnn
def inference2(encoder_inputs, embeddings_inputs, classNum, isTrainModel=True):
    encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings_inputs, encoder_inputs)
    encoder_cell_fw = LSTMCell(encoder_hidden_units)
    encoder_cell_bw = LSTMCell(encoder_hidden_units)

    ((encoder_fw_outputs,
      encoder_bw_outputs),
     (encoder_fw_final_state,
      encoder_bw_final_state)) = (
        tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell_fw,
                                        cell_bw=encoder_cell_bw,
                                        inputs=encoder_inputs_embedded,
                                        dtype=tf.float32, time_major=False)
    )
    output = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)
    output = tf.transpose(output, [1, 0, 2])
    print("------------",output)
    print(output.get_shape()[0])
    #get last cell output to classify
    last = tf.gather(output, tf.shape(output)[0] - 1)
    print(last)
    # xxx
    after_dp = tf.layers.dropout(last, rate=0.5, training=isTrainModel)
    output_logits = tf.contrib.layers.fully_connected(after_dp, num_outputs=classNum, activation_fn=tf.identity)
    print(output_logits)
    # xxx
    return output_logits


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels,
        logits=logits,
        name='cross_entropy'
    )
    loss = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    return loss


def train(loss, learning_rate, global_step):
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads = opt.compute_gradients(loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    return apply_gradient_op
