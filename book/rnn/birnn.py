import  tensorflow as tf
hidden_units=20
forwardCell=tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units)
backwordCell=tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units)
timesteps=5;
batch_size=2
input=tf.placeholder(tf.float32, [batch_size, timesteps, 1], name='input_x')

outputs,output_states=tf.nn.bidirectional_dynamic_rnn(inputs=input,cell_fw=forwardCell,cell_bw=backwordCell,dtype=tf.float32,time_major=False)

print(outputs)
print(output_states)



def stacked_bidirectional_rnn(num_units, num_layers, inputs, seq_lengths, batch_size):
    _inputs = inputs
    for _ in range(num_layers):
        with tf.variable_scope(None, default_name="bidirectional-rnn"):
            rnn_cell_fw = tf.nn.rnn_cell.BasicRNNCell(num_units)
            rnn_cell_bw = tf.nn.rnn_cell.BasicRNNCell(num_units)
            initial_state_fw = rnn_cell_fw.zero_state(batch_size, dtype=tf.float32)
            initial_state_bw = rnn_cell_bw.zero_state(batch_size, dtype=tf.float32)
            (output, state) = tf.nn.bidirectional_dynamic_rnn(rnn_cell_fw, rnn_cell_bw, _inputs, seq_lengths,
                                                              initial_state_fw, initial_state_bw, dtype=tf.float32)
            _inputs = tf.concat(output, 2)
    return _inputs



def rnn_layers(inputs,lengths,num_units,num_layers,dropout):
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
        sequence_length=lengths,
        dtype=tf.float32,
        scope="birnn")
    return outputs

