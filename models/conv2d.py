"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
slim = tf.contrib.slim


def conv2d_args_scope(weight_decay=0.0001,
                      batch_norm_decay=0.997,
                      batch_norm_epsilon=1e-5,
                      batch_norm_scale=True,
                      activation_fn=tf.nn.elu,
                      use_batch_norm=True):

    batch_norm_params = {
        'decay': batch_norm_decay,
        'epsilon': batch_norm_epsilon,
        'scale': batch_norm_scale,
        'updates_collections': tf.GraphKeys.UPDATE_OPS,
        'fused': None,  # Use fused batch norm if possible.
    }
    with slim.arg_scope(
            [slim.conv2d, slim.fully_connected],
            weights_regularizer=slim.l2_regularizer(weight_decay),
            weights_initializer=slim.variance_scaling_initializer()):
        with slim.arg_scope(
                [slim.conv2d],
                activation_fn=activation_fn,
                normalizer_fn=slim.batch_norm if use_batch_norm else None,
                normalizer_params=batch_norm_params):
            with slim.arg_scope([slim.batch_norm], **batch_norm_params):
                # The following implies padding='SAME' for pool1, which makes feature
                # alignment easier for dense prediction tasks. This is also used in
                # https://github.com/facebook/fb.resnet.torch. However the accompanying
                # code of 'Deep Residual Learning for Image Recognition' uses
                # padding='VALID' for pool1. You can switch to that choice by setting
                # slim.arg_scope([slim.max_pool2d], padding='VALID').
                with slim.arg_scope([slim.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc


def create_conv2d_model(
        fingerprint_input, model_settings, dropout_prob=1.0, is_training=True):
    """Convolutional model 2.
    """
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(
        fingerprint_input, [-1, input_time_size, input_frequency_size, 1])

    with slim.arg_scope(conv2d_args_scope()):
        with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
            net = slim.conv2d(fingerprint_4d, 64, (7, 21), padding='SAME')
            net = slim.conv2d(net, 64, (5, 11), padding='SAME')
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 128, (3, 7), padding='SAME')
            net = slim.conv2d(net, 128, (3, 7), padding='SAME')
            net = slim.max_pool2d(net, 2)
            net = slim.conv2d(net, 256, (1, 3), padding='SAME')
            net = slim.conv2d(net, 256, (1, 3), padding='SAME')
            net = tf.reduce_mean(net, [2])
            print(net.shape)
            net = slim.dropout(net, dropout_prob)
            net = slim.flatten(net)
            final_fc = slim.fully_connected(net, model_settings['label_count'])
            print(final_fc.shape)

    return final_fc
