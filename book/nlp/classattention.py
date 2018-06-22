import tensorflow as tf

hidden_units=20
attention_size=hidden_units
cell=tf.nn.rnn_cell.BasicLSTMCell(num_units=hidden_units)
timesteps=5;
batch_size=2
input=tf.placeholder(tf.float32, [batch_size, timesteps, 1], name='input_x')
outputs, states=tf.nn.dynamic_rnn(cell=cell,inputs=input,dtype=tf.float32)
outputs=tf.reshape(outputs,shape=[timesteps,batch_size,hidden_units])
attention_w = tf.Variable(tf.truncated_normal([hidden_units, attention_size], stddev=0.1), name='attention_w')
attention_b = tf.Variable(tf.constant(0.1, shape=[attention_size]), name='attention_b')
u_list = []
for t in range(timesteps):
    u_t = tf.tanh(tf.matmul(outputs[t], attention_w) + attention_b)
    u_list.append(u_t)
u_w = tf.Variable(tf.truncated_normal([attention_size, 1], stddev=0.1), name='attention_uw')
attn_z = []
for t in range(timesteps):
    z_t = tf.matmul(u_list[t], u_w)
    attn_z.append(z_t)
# transform to batch_size * sequence_length
attn_zconcat = tf.concat(attn_z, axis=1)
alpha = tf.nn.softmax(attn_zconcat)
print(alpha)
# transform to sequence_length * batch_size * 1 , same rank as outputs
alpha_trans = tf.reshape(tf.transpose(alpha, [1,0]), [timesteps, -1, 1])
final_output = tf.reduce_sum(outputs * alpha_trans, 0)
