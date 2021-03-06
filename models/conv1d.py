"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
slim = tf.contrib.slim


def conv1d_args_scope(
        weight_decay=0.0001,
        batch_norm_decay=0.997,
        batch_norm_epsilon=1e-5,
        batch_norm_scale=True,
        activation_fn=tf.nn.elu,
        use_batch_norm=False,
        data_format='NCW'):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': None,  # Use fused batch norm if possible.
    }
    with slim.arg_scope(
            [slim.convolution, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer(),
            activation_fn=activation_fn):
        with slim.arg_scope(
                [slim.convolution],
                data_format=data_format,
                normalizer_fn=slim.batch_norm if use_batch_norm else None,
                normalizer_params=batch_norm_params,
                padding='SAME'):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                # The following implies padding='SAME' for pool1, which makes feature
                # alignment easier for dense prediction tasks. This is also used in
                # https://github.com/facebook/fb.resnet.torch. However the accompanying
                # code of 'Deep Residual Learning for Image Recognition' uses
                # padding='VALID' for pool1. You can switch to that choice by setting
                # slim.arg_scope([slim.max_pool2d], padding='VALID').
                with slim.arg_scope(
                        [slim.pool],
                        padding='SAME',
                        data_format=data_format) as arg_sc:
                    return arg_sc


def create_conv1d_basic3_model(
        waveform_input, model_settings, dropout_prob=1.0, is_training=True):
    """Convolutional 1d model.
    """
    num_samples = model_settings['desired_samples']
    input_3d = tf.reshape(
        waveform_input, [-1, 1, num_samples])
    print(input_3d.shape)

    with slim.arg_scope(conv1d_args_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net = slim.convolution(input_3d, 64, 3, stride=3)
            net = slim.convolution(net, 64, 3)
            net = slim.pool(net, 3, 'MAX', stride=3)
            print(net.shape)
            net = slim.convolution(net, 128, 3)
            net = slim.pool(net, 3, 'MAX', stride=3)
            net = slim.convolution(net, 128, 3)
            net = slim.pool(net, 3, 'MAX', stride=3)
            net = slim.convolution(net, 256, 3)
            net = slim.pool(net, 3, 'MAX', stride=3)
            net = slim.convolution(net, 256, 3)
            net = slim.pool(net, 3, 'MAX', stride=3)
            net = slim.convolution(net, 256, 3)
            net = slim.pool(net, 3, 'MAX', stride=3)
            net = slim.convolution(net, 512, 3)
            net = slim.convolution(net, 512, 3)
            print(net.shape)
            net_global_avg = tf.reduce_mean(net, [2])
            net_global_max = tf.reduce_max(net, [2])
            print(net.shape)
            net = 0.5 * (net_global_avg + net_global_max)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024)
            net = slim.dropout(net, dropout_prob)
            final_fc = slim.fully_connected(
                net, model_settings['label_count'], activation_fn=None)
            print(final_fc.shape)

    return final_fc


def create_conv1d_basic2_model(
        waveform_input, model_settings, dropout_prob=1.0, is_training=True):
    """Convolutional 1d model.
    """
    num_samples = model_settings['desired_samples']
    input_3d = tf.reshape(
        waveform_input, [-1, 1, num_samples])
    print(input_3d.shape)

    with slim.arg_scope(conv1d_args_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net = slim.convolution(input_3d, 64, 3, stride=2)
            net = slim.convolution(net, 64, 3, stride=2)
            net = slim.pool(net, 3, 'MAX', stride=2)
            print(net.shape)
            net = slim.convolution(net, 128, 3)
            net = slim.convolution(net, 128, 3)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 256, 3)
            net = slim.convolution(net, 256, 3)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 256, 3)
            net = slim.convolution(net, 256, 3)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 512, 3)
            net = slim.convolution(net, 512, 3)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 1024, 3)
            net = slim.convolution(net, 1024, 3)
            print(net.shape)
            net_global_avg = tf.reduce_mean(net, [2])
            net_global_max = tf.reduce_max(net, [2])
            print(net.shape)
            net = 0.5 * (net_global_avg + net_global_max)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024)
            net = slim.dropout(net, dropout_prob)
            final_fc = slim.fully_connected(
                net, model_settings['label_count'], activation_fn=None)
            print(final_fc.shape)

    return final_fc


def create_conv1d_a_model(
        waveform_input, model_settings, dropout_prob=1.0, is_training=True):
    """Convolutional 1d model.
    """
    num_samples = model_settings['desired_samples']
    input_3d = tf.reshape(
        waveform_input, [-1, 1, num_samples])
    print(input_3d.shape)

    with slim.arg_scope(conv1d_args_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net = slim.convolution(input_3d, 64, 3, stride=2)
            net = slim.convolution(net, 64, 3, stride=2)
            print(net.shape)
            net = slim.pool(net, 3, 'MAX', stride=2)
            print(net.shape)
            net = slim.convolution(net, 128, 3, rate=1)
            net = slim.convolution(net, 128, 3, rate=2)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 256, 3, rate=4)
            net = slim.convolution(net, 256, 3, rate=8)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 512, 3, rate=16)
            net = slim.convolution(net, 512, 3, rate=32)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 1024, 3, rate=64)
            net = slim.convolution(net, 1024, 3, rate=128)
            print(net.shape)
            net_global_avg = tf.reduce_mean(net, [2])
            net_global_max = tf.reduce_max(net, [2])
            net = 0.5 * (net_global_avg + net_global_max)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024)
            net = slim.dropout(net, dropout_prob)
            final_fc = slim.fully_connected(
                net, model_settings['label_count'], activation_fn=None)
            print(final_fc.shape)

    return final_fc


def create_conv1d_b_model(
        waveform_input, model_settings, dropout_prob=1.0, is_training=True):
    """Convolutional 1d model.
    """
    num_samples = model_settings['desired_samples']
    input_3d = tf.reshape(
        waveform_input, [-1, 1, num_samples])
    print(input_3d.shape)

    with slim.arg_scope(conv1d_args_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net = slim.convolution(input_3d, 32, 3, stride=2)
            net = slim.convolution(net, 32, 3)
            print(net.shape)
            net = slim.pool(net, 3, 'MAX', stride=2)
            print(net.shape)
            net = slim.convolution(net, 64, 3, rate=2)
            net = slim.convolution(net, 64, 1)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 128, 3, rate=4)
            net = slim.convolution(net, 128, 1)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 256, 3, rate=8)
            net = slim.convolution(net, 256, 1)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 512, 3, rate=16)
            net = slim.convolution(net, 512, 1, rate=32)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 1024, 3, rate=64)
            net = slim.convolution(net, 1024, 1, rate=128)
            print(net.shape)
            net_global_avg = tf.reduce_mean(net, [2])
            net_global_max = tf.reduce_max(net, [2])
            net = tf.concat([net_global_avg, net_global_max], axis=1)
            print('concat', net.shape)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024)
            print('fc1', net.shape)
            net = slim.dropout(net, dropout_prob)
            final_fc = slim.fully_connected(
                net, model_settings['label_count'], activation_fn=None)
            print(final_fc.shape)

    return final_fc


def create_conv1d_c_model(
        waveform_input, model_settings, dropout_prob=1.0, is_training=True):
    """Convolutional 1d model.
    """
    num_samples = model_settings['desired_samples']
    input_3d = tf.reshape(
        waveform_input, [-1, 1, num_samples])
    print(input_3d.shape)

    with slim.arg_scope(conv1d_args_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net = slim.convolution(input_3d, 64, 3, stride=2, padding='SAME')
            net = slim.convolution(net, 64, 3, padding='SAME')
            print(net.shape)
            net = slim.pool(net, 3, 'MAX', stride=2)
            print(net.shape)
            net = slim.convolution(net, 128, 3, rate=2)
            net = slim.convolution(net, 128, 1)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 256, 3, rate=4)
            net = slim.convolution(net, 256, 1)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 512, 3, rate=8)
            net = slim.convolution(net, 512, 1)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 1024, 3, rate=16)
            net = slim.convolution(net, 1024, 3, rate=32)
            net = slim.pool(net, 3, 'MAX', stride=2)
            net = slim.convolution(net, 2048, 3, rate=64)
            net = slim.convolution(net, 2048, 3, rate=128)
            print(net.shape)
            net_global_avg = tf.reduce_mean(net, [2])
            net_global_max = tf.reduce_max(net, [2])
            net = 0.5 * (net_global_avg + net_global_max)
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1024)
            print('fc1', net.shape)
            net = slim.dropout(net, dropout_prob)
            final_fc = slim.fully_connected(
                net, model_settings['label_count'], activation_fn=None)
            print('final', final_fc.shape)

    return final_fc


def create_conv1d_d_model(
        waveform_input, model_settings, dropout_prob=1.0, is_training=True):
    """Convolutional 1d model.
    """
    num_samples = model_settings['desired_samples']
    input_3d = tf.reshape(
        waveform_input, [-1, 1, num_samples])
    print(input_3d.shape)

    with slim.arg_scope(conv1d_args_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net = slim.convolution(input_3d, 64, 3, stride=2)
            net = slim.convolution(net, 128, 3, stride=2)
            net = slim.pool(net, 3, 'MAX', stride=2)
            print(net.shape)

            dilated = []
            net = slim.convolution(net, 256, 3, rate=1)
            dilated.append(slim.convolution(net, 256, 1, activation_fn=None))

            def _layer(x, rate):
                shortcut = tf.identity(x)
                x = slim.convolution(x, 256, 3, rate=rate)
                dilated.append(slim.convolution(x, 256, 1, activation_fn=None))
                x = x + shortcut
                return x

            net = _layer(net, rate=2)
            net = _layer(net, rate=4)
            net = _layer(net, rate=8)
            net = _layer(net, rate=16)
            net = _layer(net, rate=32)
            net = _layer(net, rate=64)
            net = _layer(net, rate=128)
            net = _layer(net, rate=256)
            net = _layer(net, rate=512)
            print(net.shape, len(dilated))

            net = sum(dilated)
            net = tf.nn.elu(net)
            #net = net[:, :, net.shape[2]//2]
            net = slim.convolution(net[:, :, 250:], 512, 1, stride=250)
            #net = slim.pool(net, 1, 'MAX', stride=250)
            print(net.shape)

            #net = slim.convolution(net, 1024, 1, stride=2)
            #print('1x1', net.shape)

            net_global_avg = tf.reduce_mean(net, [2])
            net_global_max = tf.reduce_max(net, [2])
            net = 0.5 * (net_global_avg + net_global_max)
            net = slim.flatten(net)
            print(net.shape)

            net = slim.fully_connected(net, 1024)
            print('fc1', net.shape)

            net = slim.dropout(net, dropout_prob)
            final_fc = slim.fully_connected(
                net, model_settings['label_count'], activation_fn=None)
            print('final', final_fc.shape)

    return final_fc