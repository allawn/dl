
from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
import dataload
import tensorflow as tf
import model

vocabulary_size = 50000
embedding_size=128

maxlen = 150  # cut texts after this number of words (among top max_features most common words)
batchSize = 32
classNum=2
learning_rate=0.001
epochs=100
ckpt_dir=""

path='C:\\wuwei\\work\\github\\data\\imdb.pkl'

print('Loading data...')
trainData, testData = dataload.load_data(path=path, nb_words=vocabulary_size)
print(len(trainData), 'train sequences')
print(len(testData), 'test sequences')

trainBatches=dataload.get_batches(data=trainData)
testData=dataload.get_batches(data=testData)

def train():
    graph = tf.Graph()
    with graph.as_default():

        global_step = tf.contrib.framework.get_or_create_global_step()

        encoder_inputs = tf.placeholder(shape=(batchSize, None), dtype=tf.int32, name='encoder_inputs')

        class_targets = tf.placeholder(shape=(batchSize,), dtype=tf.int32, name='class_targets')

        embeddings_inputs = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        logist = model.inference(encoder_inputs, embeddings_inputs, classNum)

        loss = model.loss(logits=logist, labels=class_targets)

        train_op = model.train(loss, learning_rate, global_step)

        top_k_op = tf.nn.in_top_k(logist, class_targets, 1)
        accuracy = tf.reduce_mean(tf.cast(top_k_op, "float"), name="accuracy")

        saver = tf.train.Saver()

    iteration = 1

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            for x, y in trainBatches:
                feed = {
                        encoder_inputs: x,
                        class_targets: y,
                       }

                lossVal, accuracyVal, _ = sess.run([loss, accuracy, train_op], feed_dict=feed)
                print("Iteration: {}".format(iteration), "\tlossVal: {:.3f}".format(lossVal),
                      "\taccuracyVal: {:.3f}".format(accuracyVal))
                iteration=iteration+1

        # saver.save(sess, ckpt_dir)



def main(argv=None):
    train()


if __name__ == '__main__':
    tf.app.run()