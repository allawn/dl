import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
tf.app.flags.DEFINE_string('data_dir', '.', """the default data dirs""")

FLAGS=tf.app.flags.FLAGS
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

IMAGE_SIZE = 28
NUM_CHANNELS = 1
BATCH_SIZE=32
num_epochs=1

train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
eval_data = mnist.test.images
eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
train_size = train_labels.shape[0]
checkPointPath="C:\\tmp\\mnistckp\\model.ckp"

#not add 正则化
def inference(input, l2_regularizer=None):

    input_layer = tf.reshape(input, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        kernel_regularizer=l2_regularizer,
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        kernel_regularizer=l2_regularizer,
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4)
    logits = tf.layers.dense(inputs=dropout, units=10)
    return logits


def train():

    train_data_node = tf.placeholder(tf.float32, shape=(BATCH_SIZE, IMAGE_SIZE*IMAGE_SIZE* NUM_CHANNELS))
    train_labels_node = tf.placeholder(tf.int64, shape=(BATCH_SIZE,10))
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.0)
    logits = inference(train_data_node, regularizer)
    l2_loss = tf.losses.get_regularization_loss()
    entropyloss=tf.nn.softmax_cross_entropy_with_logits(labels=train_labels_node, logits=logits)
    loss = tf.reduce_mean([l2_loss+entropyloss])
    learning_rate=0.01
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9).minimize(loss)
    labels = tf.argmax(train_labels_node, 1)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(top_k_op, "float"), name="accuracy")
    saver = tf.train.Saver()
    initAll=tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(initAll)
        for step in range(int(num_epochs * train_size) // BATCH_SIZE):
            offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
            batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
            batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
            feed_dict = {train_data_node: batch_data,
                         train_labels_node: batch_labels}
            _, lossVal, accuracyVal=sess.run([optimizer, loss, accuracy], feed_dict=feed_dict)
            print('Iter %d, lossVal %.3f, accuracyVal %.3f' % (step, lossVal, accuracyVal))

        saver.save(sess=sess,save_path=checkPointPath)

if __name__ == '__main__':
    train()