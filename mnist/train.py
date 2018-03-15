import tensorflow as tf
import argparse
import sys

from tensorflow.examples.tutorials.mnist import input_data
tf.app.flags.DEFINE_string('data_dir', '.', """the default data dirs""")

FLAGS=tf.app.flags.FLAGS
mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

def inference():
    bb=2

def train():
    aa=1

if __name__ == '__main__':
    train()