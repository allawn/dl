import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim

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
pbPath="C:\\tmp\\mnistckp\\model.pb"
tf.logging.set_verbosity(tf.logging.INFO)

def inference(input, l2_regularizer=None):
    input_layer = tf.reshape(input, [-1, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], name="inputLayer")
    with slim.arg_scope([slim.conv2d],weights_regularizer=l2_regularizer):
        net=slim.conv2d(input_layer,32,[5,5])
        net=slim.max_pool2d(net, [2,2], stride=2)
        net=slim.conv2d(net,64,[5,5],)
        net=slim.max_pool2d(net,[2,2],stride=2)
        net=slim.flatten(net)
        net=slim.fully_connected(net, 1024)
        net=slim.dropout(net, 0.4)
        logits = slim.fully_connected(net, 10)
    return logits


def train():
    image, label=tf.train.slice_input_producer([train_data,train_labels])
    images,labels=tf.train.batch([image,label], BATCH_SIZE)

    train_data_node=tf.reshape(images, shape=[-1, IMAGE_SIZE*IMAGE_SIZE* NUM_CHANNELS],name="inputdataName")
    train_data_node=tf.cast(train_data_node,tf.float32)
    train_labels_node=tf.reshape(labels,shape=[-1,10])
    train_labels_node=tf.cast(train_labels_node,tf.int64)

    regularizer = slim.l2_regularizer(scale=0.0)
    logits = inference(train_data_node, regularizer)
    prediction = tf.nn.softmax(logits,name="predictionName")
    slim.losses.softmax_cross_entropy(onehot_labels=train_labels_node, logits=logits)
    total_loss=slim.losses.get_total_loss()

    learning_rate=0.01
    optimizer = tf.train.MomentumOptimizer(learning_rate,momentum=0.9)
    global_step=slim.train.create_global_step()
    train_op = slim.learning.create_train_op(total_loss, optimizer,global_step)

    labels = tf.argmax(train_labels_node, 1)
    top_k_op = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(top_k_op, "float"), name="accuracy")
    saver = tf.train.Saver()

    graph_def = tf.get_default_graph().as_graph_def()
    with tf.gfile.GFile(pbPath, 'wb') as f:
        f.write(graph_def.SerializeToString())

    slim.learning.train(train_op,checkPointPath,number_of_steps=500,log_every_n_steps=1)


#     with tf.Session() as sess:
#         coord = tf.train.Coordinator()
#         threads = tf.train.start_queue_runners(sess, coord)
#         sess.run(initAll)
#         for step in range(int(num_epochs * train_size) // BATCH_SIZE):
#             _, lossVal, accuracyVal=sess.run([train_op, total_loss, accuracy])
#             print('Iter %d, lossVal %.3f, accuracyVal %.3f' % (step, lossVal, accuracyVal))
#
#         saver.save(sess=sess,save_path=checkPointPath)
#
#         coord.request_stop()
#         coord.join(threads)


if __name__ == '__main__':
    train()