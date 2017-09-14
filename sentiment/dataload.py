from __future__ import absolute_import
from six.moves import cPickle
import gzip
from six.moves import zip
import numpy as np
import random


def load_inference_data(X, nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):


    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        for x in X:
            if len(x) < maxlen:
                new_X.append(x)
        X = new_X
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                                      'Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX


    inferenceData=[]

    for index in range(len(X)):
        inferenceData.append(X[index])

    return inferenceData

def load_data(path="imdb.pkl", nb_words=None, skip_top=0,
              maxlen=None, test_split=0.2, seed=113,
              start_char=1, oov_char=2, index_from=3):

    #path = get_file(path, origin="https://s3.amazonaws.com/text-datasets/imdb.pkl")

    if path.endswith(".gz"):
        f = gzip.open(path, 'rb')
    else:
        f = open(path, 'rb')

    X, labels = cPickle.load(f)
    f.close()

    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    if start_char is not None:
        X = [[start_char] + [w + index_from for w in x] for x in X]
    elif index_from:
        X = [[w + index_from for w in x] for x in X]

    if maxlen:
        new_X = []
        new_labels = []
        for x, y in zip(X, labels):
            if len(x) < maxlen:
                new_X.append(x)
                new_labels.append(y)
        X = new_X
        labels = new_labels
    if not X:
        raise Exception('After filtering for sequences shorter than maxlen=' +
                        str(maxlen) + ', no sequence was kept. '
                                      'Increase maxlen.')
    if not nb_words:
        nb_words = max([max(x) for x in X])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters: 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        X = [[oov_char if (w >= nb_words or w < skip_top) else w for w in x] for x in X]
    else:
        nX = []
        for x in X:
            nx = []
            for w in x:
                if (w >= nb_words or w < skip_top):
                    nx.append(w)
            nX.append(nx)
        X = nX

    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])

    X_test = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])

    trainData=[]
    testData=[]
    for index in range(len(X_train)):
        trainData.append((X_train[index],y_train[index]))

    for index in range(len(X_test)):
        testData.append((X_test[index], y_test[index]))

    return trainData, testData


def batch(inputs, max_sequence_length=None):

    sequence_lengths = [len(seq) for seq in inputs]
    batch_size = len(inputs)

    if max_sequence_length is None:
        max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length], dtype=np.int32)  # == PAD

    for i, seq in enumerate(inputs):
        for j, element in enumerate(seq):
            inputs_batch_major[i, j] = element

    # print(inputs_batch_major)
    # [batch_size, max_time] -> [max_time, batch_size]
    # inputs_time_major = inputs_batch_major.swapaxes(0, 1)

    return inputs_batch_major


def get_batches(data, batch_size=32):
    for index in range(batch_size):
        data.append(data[random.randint(0,(len(data)-1))])
    n_batches = len(data) // batch_size
    dataList = data[:n_batches * batch_size]
    for dataIndex in range(0, len(dataList), batch_size):
        inputs=[]
        for row in range(batch_size):
            inputs.append(dataList[dataIndex+row][0])

        x =batch(inputs)

        y = np.zeros(shape=[batch_size],dtype=np.int32)
        for row in range(batch_size):
            y[row]=dataList[dataIndex+row][1]

        yield  x, y



def aatest():
    (X_train, y_train), (X_test, y_test)=load_data(path='C:\\wuwei\\work\\github\\data\\imdb.pkl')
    print(X_train[0])
    print(y_train[0])

# aatest()