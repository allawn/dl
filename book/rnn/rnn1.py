import tensorflow as tf
import numpy as np

hidden_units=20
rnnLayerNum=1
rnnCells=[]
for i in range(rnnLayerNum):
    rnnCells.append(tf.nn.rnn_cell.BasicRNNCell(num_units=hidden_units))

multiRnnCell=tf.nn.rnn_cell.MultiRNNCell(rnnCells)
timesteps=5;
input=tf.placeholder(tf.float32, [1, timesteps, 1], name='input_x')
sequence_length=[2]
initial_state=multiRnnCell.zero_state(batch_size=1,dtype=tf.float32)

outputs, final_state =tf.nn.dynamic_rnn(multiRnnCell,input,sequence_length=sequence_length,initial_state=initial_state,dtype=tf.float32)
print(outputs)
print(final_state)
outputsList=tf.unstack(outputs,timesteps, 1)

initopt=tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(initopt)
    inputVal=np.ones([1,5,1],dtype=np.float32)
    print(sess.run(outputsList[1], feed_dict={input:inputVal}))
    print(sess.run(final_state, feed_dict={input:inputVal}))
