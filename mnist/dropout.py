import tensorflow as tf
#��������ռλ�������Խ���dropout�ĸ���
dropout = tf.placeholder(tf.float32)
#�����������������ģ����Ԫ�����������Ϊ5 *5 ��С�ľ�������Ԫ��Ϊ1
x = tf.Variable(tf.ones([5, 5]))
#����dropout����
y = tf.nn.dropout(x, dropout)
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
#����dropout����0.4, ������dropout����
print(sess.run(y, feed_dict={dropout: 0.4}))
