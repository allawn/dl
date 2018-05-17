from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Make TFGAN models and TF-Slim models discoverable.
import sys
import os
# This is needed since the notebook is stored in the `tensorflow/models/gan` folder.
sys.path.append('..')
sys.path.append(os.path.join('..', 'slim'))

import matplotlib.pyplot as plt
import numpy as np
import time
import functools
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

# Main TFGAN library.
import tensorflow.contrib.gan as tfgan
# tfgan = tf.contrib.gan

# TFGAN MNIST examples from `tensorflow/models`.
from mnist import data_provider
from mnist import util

# TF-Slim data provider.
from datasets import download_and_convert_mnist

# Shortcuts for later.
queues = tf.contrib.slim.queues
layers = tf.contrib.layers
ds = tf.contrib.distributions
framework = tf.contrib.framework

leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.01)


def visualize_training_generator(train_step_num, start_time, data_np):
    """Visualize generator outputs during training.

    Args:
        train_step_num: The training step number. A python integer.
        start_time: Time when training started. The output of `time.time()`. A
            python float.
        data: Data to plot. A numpy array, most likely from an evaluated TensorFlow
            tensor.
    """
    print('Training step: %i' % train_step_num)
    time_since_start = (time.time() - start_time) / 60.0
    print('Time since start: %f m' % time_since_start)
    print('Steps per min: %f' % (train_step_num / time_since_start))
    plt.axis('off')
    plt.imshow(np.squeeze(data_np), cmap='gray')
    plt.show()


def visualize_digits(tensor_to_visualize):
    """Visualize an image once. Used to visualize generator before training.

    Args:
        tensor_to_visualize: An image tensor to visualize. A python Tensor.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with queues.QueueRunners(sess):
            images_np = sess.run(tensor_to_visualize)
    plt.axis('off')
    plt.imshow(np.squeeze(images_np), cmap='gray')
    plt.show()


def evaluate_tfgan_loss(gan_loss, name=None):
    """Evaluate GAN losses. Used to check that the graph is correct.

    Args:
        gan_loss: A GANLoss tuple.
        name: Optional. If present, append to debug output.
    """
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        with queues.QueueRunners(sess):
            gen_loss_np = sess.run(gan_loss.generator_loss)
            dis_loss_np = sess.run(gan_loss.discriminator_loss)
    if name:
        print('%s generator loss: %f' % (name, gen_loss_np))
        print('%s discriminator loss: %f' % (name, dis_loss_np))
    else:
        print('Generator loss: %f' % gen_loss_np)
        print('Discriminator loss: %f' % dis_loss_np)


MNIST_DATA_DIR = '/tmp/mnist-data'

if not tf.gfile.Exists(MNIST_DATA_DIR):
    tf.gfile.MakeDirs(MNIST_DATA_DIR)

download_and_convert_mnist.run(MNIST_DATA_DIR)

tf.reset_default_graph()

# Define our input pipeline. Pin it to the CPU so that the GPU can be reserved
# for forward and backwards propogation.
batch_size = 32
with tf.device('/cpu:0'):
    real_images, _, _ = data_provider.provide_data(
        'train', batch_size, MNIST_DATA_DIR)

# Sanity check that we're getting images.
check_real_digits = tfgan.eval.image_reshaper(real_images[:20,...], num_cols=10)
visualize_digits(check_real_digits)


def infogan_generator(inputs, categorical_dim, weight_decay=2.5e-5,
                      is_training=True):
    """InfoGAN discriminator network on MNIST digits.

    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.

    Args:
        inputs: A 3-tuple of Tensors (unstructured_noise, categorical structured
            noise, continuous structured noise). `inputs[0]` and `inputs[2]` must be
            2D, and `inputs[1]` must be 1D. All must have the same first dimension.
        categorical_dim: Dimensions of the incompressible categorical noise.
        weight_decay: The value of the l2 weight decay.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population
            statistics.

    Returns:
        A generated image in the range [-1, 1].
    """
    unstructured_noise, cat_noise, cont_noise = inputs
    cat_noise_onehot = tf.one_hot(cat_noise, categorical_dim)
    all_noise = tf.concat([unstructured_noise, cat_noise_onehot, cont_noise], axis=1)

    with framework.arg_scope(
            [layers.fully_connected, layers.conv2d_transpose],
            activation_fn=tf.nn.relu, normalizer_fn=layers.batch_norm,
            weights_regularizer=layers.l2_regularizer(weight_decay)), \
         framework.arg_scope([layers.batch_norm], is_training=is_training):
        net = layers.fully_connected(all_noise, 1024)
        net = layers.fully_connected(net, 7 * 7 * 128)
        net = tf.reshape(net, [-1, 7, 7, 128])
        net = layers.conv2d_transpose(net, 64, [4, 4], stride=2)
        net = layers.conv2d_transpose(net, 32, [4, 4], stride=2)
        # Make sure that generator output is in the same range as `inputs`
        # ie [-1, 1].
        net = layers.conv2d(net, 1, 4, normalizer_fn=None, activation_fn=tf.tanh)

        return net


def infogan_discriminator(img, unused_conditioning, weight_decay=2.5e-5,
                          categorical_dim=10, continuous_dim=2, is_training=True):
    """InfoGAN discriminator network on MNIST digits.

    Based on a paper https://arxiv.org/abs/1606.03657 and their code
    https://github.com/openai/InfoGAN.

    Args:
        img: Real or generated MNIST digits. Should be in the range [-1, 1].
        unused_conditioning: The TFGAN API can help with conditional GANs, which
            would require extra `condition` information to both the generator and the
            discriminator. Since this example is not conditional, we do not use this
            argument.
        weight_decay: The L2 weight decay.
        categorical_dim: Dimensions of the incompressible categorical noise.
        continuous_dim: Dimensions of the incompressible continuous noise.
        is_training: If `True`, batch norm uses batch statistics. If `False`, batch
            norm uses the exponential moving average collected from population
            statistics.

    Returns:
        Logits for the probability that the image is real, and a list of posterior
        distributions for each of the noise vectors.
    """
    with framework.arg_scope(
            [layers.conv2d, layers.fully_connected],
            activation_fn=leaky_relu, normalizer_fn=None,
            weights_regularizer=layers.l2_regularizer(weight_decay),
            biases_regularizer=layers.l2_regularizer(weight_decay)):
        net = layers.conv2d(img, 64, [4, 4], stride=2)
        net = layers.conv2d(net, 128, [4, 4], stride=2)
        net = layers.flatten(net)
        net = layers.fully_connected(net, 1024, normalizer_fn=layers.layer_norm)

        logits_real = layers.fully_connected(net, 1, activation_fn=None)

        # Recognition network for latent variables has an additional layer
        with framework.arg_scope([layers.batch_norm], is_training=is_training):
            encoder = layers.fully_connected(
                net, 128, normalizer_fn=layers.batch_norm)

        # Compute logits for each category of categorical latent.
        logits_cat = layers.fully_connected(
            encoder, categorical_dim, activation_fn=None)
        q_cat = ds.Categorical(logits_cat)

        # Compute mean for Gaussian posterior of continuous latents.
        mu_cont = layers.fully_connected(
            encoder, continuous_dim, activation_fn=None)
        sigma_cont = tf.ones_like(mu_cont)
        q_cont = ds.Normal(loc=mu_cont, scale=sigma_cont)

        return logits_real, [q_cat, q_cont]


# Dimensions of the structured and unstructured noise dimensions.
cat_dim, cont_dim, noise_dims = 10, 2, 64

generator_fn = functools.partial(infogan_generator, categorical_dim=cat_dim)
discriminator_fn = functools.partial(
    infogan_discriminator, categorical_dim=cat_dim,
    continuous_dim=cont_dim)
unstructured_inputs, structured_inputs = util.get_infogan_noise(
    batch_size, cat_dim, cont_dim, noise_dims)

infogan_model = tfgan.infogan_model(
    generator_fn=generator_fn,
    discriminator_fn=discriminator_fn,
    real_data=real_images,
    unstructured_generator_inputs=unstructured_inputs,
    structured_generator_inputs=structured_inputs)

infogan_loss = tfgan.gan_loss(
    infogan_model,
    gradient_penalty_weight=1.0,
    mutual_information_penalty_weight=1.0)

# Sanity check that we can evaluate our losses.
evaluate_tfgan_loss(infogan_loss)


generator_optimizer = tf.train.AdamOptimizer(0.001, beta1=0.5)
discriminator_optimizer = tf.train.AdamOptimizer(0.00009, beta1=0.5)
gan_train_ops = tfgan.gan_train_ops(
    infogan_model,
    infogan_loss,
    generator_optimizer,
    discriminator_optimizer)

# Set up images to evaluate MNIST score.
images_to_eval = 500
assert images_to_eval % cat_dim == 0

unstructured_inputs = tf.random_normal([images_to_eval, noise_dims-cont_dim])
cat_noise = tf.constant(list(range(cat_dim)) * (images_to_eval // cat_dim))
cont_noise = tf.random_uniform([images_to_eval, cont_dim], -1.0, 1.0)

with tf.variable_scope(infogan_model.generator_scope, reuse=True):
    eval_images = infogan_model.generator_fn(
        (unstructured_inputs, cat_noise, cont_noise))

MNIST_CLASSIFIER_FROZEN_GRAPH = './mnist/data/classify_mnist_graph_def.pb'
eval_score = util.mnist_score(
    eval_images, MNIST_CLASSIFIER_FROZEN_GRAPH)

# Generate three sets of images to visualize the effect of each of the structured noise
# variables on the output.
rows = 2
categorical_sample_points = np.arange(0, 10)
continuous_sample_points = np.linspace(-1.0, 1.0, 10)
noise_args = (rows, categorical_sample_points, continuous_sample_points,
              noise_dims-cont_dim, cont_dim)

display_noises = []
display_noises.append(util.get_eval_noise_categorical(*noise_args))
display_noises.append(util.get_eval_noise_continuous_dim1(*noise_args))
display_noises.append(util.get_eval_noise_continuous_dim2(*noise_args))

display_images = []
for noise in display_noises:
    with tf.variable_scope('Generator', reuse=True):
        display_images.append(infogan_model.generator_fn(noise, is_training=False))

display_img = tfgan.eval.image_reshaper(
    tf.concat(display_images, 0), num_cols=10)

global_step = tf.train.get_or_create_global_step()
train_step_fn = tfgan.get_sequential_train_steps()
loss_values, mnist_score_values  = [], []

with tf.train.SingularMonitoredSession() as sess:
    start_time = time.time()
    for i in xrange(6001):
        cur_loss, _ = train_step_fn(
            sess, gan_train_ops, global_step, train_step_kwargs={})
        loss_values.append((i, cur_loss))
        if i % 1000 == 0:
            mnist_score_np, display_img_np = sess.run([eval_score, display_img])
            mnist_score_values.append((i, mnist_score_np))
            visualize_training_generator(i, start_time, display_img_np)
            print('Current loss: %f' % cur_loss)
            print('Current MNIST score: %f' % mnist_score_values[-1][1])