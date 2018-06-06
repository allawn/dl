import tensorflow as tf

hidden_units=20
rnnLayerNum=2
rnnCells=[]
for i in range(rnnLayerNum):
    rnnCells.append(tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units))

multiRnnCell=tf.nn.rnn_cell.MultiRNNCell(rnnCells)
timesteps=5;
input=tf.placeholder(tf.float32, [None, timesteps, 1], name='input_x')
outputs, final_state =tf.nn.dynamic_rnn(multiRnnCell,input,dtype=tf.float32)
print(outputs)
print(final_state)
