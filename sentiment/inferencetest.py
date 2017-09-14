from six.moves import cPickle
import gzip
import tensorflow as tf
import preProcessData
import os
import string
import dataload
import model
import numpy as np

def loadDic(path):
    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    dictionary = cPickle.load(f)
    f.close()

    return dictionary

classNum=2
batchSize=1
vocabulary_size = 50000
embedding_size=128

ckpt_dir = "ckp.ckpt"

def inference():
    dicPath="C:\\wuwei\\work\\github\\data\\imdb.dict.pkl"
    inferenceFile="C:\\wuwei\\work\\github\\dl\\sentiment\\inferencedata\\pos1.txt"
    dic = loadDic(dicPath)
    inference_data = preProcessData.preProcess_inference_data(inferenceFile, dic)
    print(inference_data)
    inference_data=dataload.load_inference_data(inference_data)
    print(inference_data)
    graph = tf.Graph()
    with graph.as_default():

        encoder_inputs = tf.placeholder(shape=(batchSize, None), dtype=tf.int32, name='encoder_inputs')

        embeddings_inputs = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

        logist = model.inference2(encoder_inputs, embeddings_inputs,classNum)

        inputs_batch_major = np.zeros(shape=[batchSize, len(inference_data[0])], dtype=np.int32)
        for i in range(len(inference_data[0])):
            inputs_batch_major[0, i] = inference_data[0][i]

        saver = tf.train.Saver()

    with tf.Session(graph=graph) as sess:
        # saver.restore(sess,ckpt_dir)
        sess.run(tf.global_variables_initializer())

        feed = {encoder_inputs: inputs_batch_major}

        logist_val=sess.run([logist], feed_dict=feed)
        results = np.squeeze(logist_val)
        top_1 = results.argsort()[-1:][::-1]

        print("The logist val is: ",results)
        print("The predict class is: {}".format(top_1[0]))





def aatestaa():
    dic = loadDic("C:\\wuwei\\work\\github\\data\\imdb.dict.pkl")
    print(dic.keys())
    print(dic['serbian'])


def main(argv=None):
    inference()

if __name__ == '__main__':
    tf.app.run()