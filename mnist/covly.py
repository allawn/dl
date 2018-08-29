import tensorflow as tf
import cv2
#ͨ��opencv�⣬��ȡͼƬ����Ϊ���飬����Ϊά����ϢΪ28 * 28
img = cv2.imread("1.png", 0)
#����TensorFlowͼ���������ݣ�features������Ϊ��ͼƬ����������
features = tf.Variable(initial_value=img, dtype=tf.float32)

# ��������㣬�ı���������ά��Ϊ 4-D tensor: [batch_size, width, height, channels]��ͼ������Ϊ 28 * 28���ش�С, ����Ϊ��ͨ��
input_layer = tf.reshape(features, [-1, 28, 28, 1])

# ���������1������˴�СΪ5 * 5�����������Ϊ32�������ʹ��relu������Tensorά��Ϊ[batch_size, 28, 28, 1]�����Tensorά��Ϊ[batch_size, 28, 28, 32]
conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

# �����ػ���1��
# ����2x2ά�ȵ���󻯳ػ�����������Ϊ2
# ����Tensorά��: [batch_size, 28, 28, 32]
# ���Tensoά��: [batch_size, 14, 14, 32]
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# ���������2������˴�СΪ5 * 5�����������Ϊ64�������ʹ��relu������Tensorά��Ϊ[batch_size, 14, 14, 32]�����Tensorά��Ϊ[batch_size, 14, 14, 64]
conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)

# �����ػ���2������2 * 2ά�ȵ���󻯳ػ�����������Ϊ2������Tensorά��Ϊ[batch_size, 14, 14, 64]�����Tensorά��Ϊ[batch_size, 7, 7, 64]
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# չ���ػ���2���TensorΪһ������������Tensorά��Ϊ[batch_size, 7, 7, 64]�����Tensorά��Ϊ [batch_size, 7 * 7 * 64]
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

# ����ȫ���Ӳ㣬��ȫ���Ӳ����1024��Ԫ����Ԫʹ��relu�����������Tensorά��Ϊ[batch_size, 7 * 7 * 64]�����Tensorά��Ϊ[batch_size, 1024]
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

# ��ȫ���Ӳ�����ݼ���dropout��������ֹ����ϣ�ѵ����ʱ����Ԫ����40%�ĸ��ʱ�dropout
dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=True)

# Logits�㣬��dropout������Tensor��ִ�з������������Tensorά��ά��Ϊ[batch_size, 1024]�� ���Tensorά��Ϊ[batch_size, 10]
logits = tf.layers.dense(inputs=dropout, units=10)
print(logits)
