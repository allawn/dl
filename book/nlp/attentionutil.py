import tensorflow as tf
from tensorflow.python.framework import dtypes
import numpy as np
from tensorflow.python.ops import array_ops

rnnHiddenSize = 50
attention_size = 50

def do_attention(outputs, seq_length, name):
    atten_w = tf.get_variable(name=name + "_h", dtype=tf.float32,
                              initializer=tf.random_normal_initializer(stddev=0.1),
                              shape=[rnnHiddenSize, attention_size])

    atten_h = tf.tensordot(outputs, atten_w, axes=1)
    # atten_h_bias = tf.get_variable(name=name + "_h_bias", dtype=tf.float32,
    #                                shape=[config.attention_size],
    #                                initializer=tf.random_normal_initializer(stddev=0.1))
    # atten_h = tf.tanh(tf.tensordot(outputs, atten_w, axes=1) + atten_h_bias)

    atten_v = tf.get_variable(name=name + "_v", dtype=tf.float32, shape=[attention_size],
                              initializer=tf.random_normal_initializer(stddev=0.1))
    values = tf.reduce_sum(tf.multiply(atten_h, atten_v), axis=-1)
    # values = tf.tensordot(atten_h, atten_v, axes=1)

    # consider the seq_length, set value=-np.inf for the index of outputs greater then seq_length
    values_mask_value = dtypes.as_dtype(tf.float32).as_numpy_dtype(-np.inf)
    values_mask = array_ops.sequence_mask(
        seq_length, maxlen=array_ops.shape(values)[1])
    values_mask_values = values_mask_value * array_ops.ones_like(values)
    values = array_ops.where(values_mask, values, values_mask_values)

    score = tf.nn.softmax(values, axis=-1)
    score = tf.expand_dims(score, axis=-1)

    atten_output = tf.multiply(outputs, score)

    atten_output = tf.reduce_sum(atten_output, axis=1)

    return atten_output
