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
        use_batch_norm=True,
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
            weights_regularizer=slim.l2_regularizer(weight_decay)):
        with slim.arg_scope(
                [slim.convolution],
                data_format=data_format,
                weights_initializer=slim.variance_scaling_initializer(),
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
                with slim.arg_scope(
                        [slim.pool],
                        padding='SAME',
                        data_format=data_format) as arg_sc:
                    return arg_sc


def create_conv1d_model(
        waveform_input, model_settings, dropout_prob=1.0, is_training=True):
    """Convolutional 1d model.
    """
    num_samples = model_settings['desired_samples']
    input_3d = tf.reshape(
        waveform_input, [-1, 1, num_samples])
    print(input_3d.shape)

    with slim.arg_scope([slim.batch_norm, slim.dropout], is_training=is_training):
        net = slim.convolution(input_3d, 64, 9, padding='SAME')
        net = slim.convolution(net, 64, 7, padding='SAME')
        net = slim.pool(net, 2, 'MAX')
        net = slim.convolution(net, 128, 5, padding='SAME')
        net = slim.convolution(net, 128, 5, padding='SAME')
        net = slim.pool(net, 2, 'MAX')
        net = slim.convolution(net, 256, 3, padding='SAME')
        net = slim.convolution(net, 256, 3, padding='SAME')
        net = slim.pool(net, 2, 'MAX')
        net = slim.convolution(net, 512, 3, padding='SAME')
        net = slim.convolution(net, 512, 3, padding='SAME')
        net = slim.pool(net, 2, 'MAX')
        net = slim.convolution(net, 1024, 3, padding='SAME')
        net = slim.convolution(net, 1024, 3, padding='SAME')
        print(net.shape)
        net_global_avg = tf.reduce_mean(net, [2])
        net_global_max = tf.reduce_max(net, [2])
        print(net.shape)
        net = 0.5 * (net_global_avg + net_global_max)
        net = slim.flatten(net)
        net = slim.fully_connected(net, 1024)
        net = slim.dropout(net, dropout_prob)
        final_fc = slim.fully_connected(net, model_settings['label_count'])
        print(final_fc.shape)

    return final_fc
