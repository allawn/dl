import tensorflow as tf
# ��ʾ�������ݣ�����ά��Ϊ1 * 2��С�ĸ���������
x = tf.placeholder(tf.float32, shape = [1, 2])
# ������Ԫ�е�Ȩ��ϵ����ƫ�ã���Ϊ���������������Ȩ��ϵ������ά��Ϊ2 * 1��ƫ��ά��Ϊ1
W = tf.Variable(tf.random_normal([2, 1]), name = "weight")
b = tf.Variable(tf.random_normal([1]), name = "biases")
# ������Ԫ�ļ������������sigmoidΪ�������tf.matmul(x, W) + bΪ�������ݵļ�Ȩ���
y = tf.sigmoid(tf.matmul(x, W) + b)
#�����������ݶ�Ӧ��Ŀ��ֵ
target = 0
#�����Ż�������Ϊ����֤���򴫲��㷨����ѧϰ������Ϊ1
opt = tf.train.GradientDescentOptimizer(learning_rate = 1)
# ������ʧ���������ݹ�ʽ�Ķ��壬��ʧ������Ӧ�������ֵ�ĵ���Ϊ(y - target)
loss = 1 / 2 * (y - target) ** 2
#���򴫲��㷨�У��ݶ��Զ�����
grads_and_vars = opt.compute_gradients(loss)
#ʹ���ݶ��½���������ģ�Ͳ���
applyGrads = opt.apply_gradients(grads_and_vars)
#�ռ�ģ���в�����Ӧ���ݶ�ֵ���������Ƚ���֤���򴫲��㷨
grads = []
for grad, var in grads_and_vars:
    grads.append(grad)
# ����ģ�����ݵĳ�ʼ������
init = tf.global_variables_initializer()
with tf.Session() as sess:
sess.run(init)
#����ģ��lossֵ�����������ֵy���Լ�ģ�Ͳ�����Ӧ���ݶ�ֵ����������x: [[1, 2]]Ϊ��������
    _, loss, y, grads = sess.run([applyGrads, loss, y, grads], feed_dict = {x: [[1, 2]]})
    print("loss:", loss)
    print("y:", y)
print("grads", grads)
