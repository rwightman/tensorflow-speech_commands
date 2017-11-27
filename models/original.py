# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Model definitions for simple speech recognition.

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
slim = tf.contrib.slim



def create_conv_model(
        fingerprint_input, model_settings, dropout_prob=1.0, is_training=False):
    """Builds a standard convolutional model.

    This is roughly the network labeled as 'cnn-trad-fpool3' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

    Here's the layout of the graph:

    (fingerprint_input)
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MaxPool]
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MaxPool]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v

    This produces fairly good quality results, but can involve a large number of
    weight parameters and computations. For a cheaper alternative from the same
    paper with slightly less accuracy, see 'low_latency_conv' below.

    During training, dropout nodes are introduced after each relu, controlled by a
    placeholder.

    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(
        fingerprint_input, [-1, input_time_size, input_frequency_size, 1])

    first_filter_width = 8
    first_filter_height = 20
    first_filter_count = 64
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(
        fingerprint_4d, first_weights, [1, 1, 1, 1], 'SAME') + first_bias
    first_relu = tf.nn.relu(first_conv)
    first_dropout = tf.layers.dropout(first_relu, dropout_prob, training=is_training)
    max_pool = tf.nn.max_pool(first_dropout, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')

    second_filter_width = 4
    second_filter_height = 10
    second_filter_count = 64
    second_weights = tf.Variable(
        tf.truncated_normal(
            [
                second_filter_height, second_filter_width, first_filter_count,
                second_filter_count
            ],
            stddev=0.01))
    second_bias = tf.Variable(tf.zeros([second_filter_count]))
    second_conv = tf.nn.conv2d(
        max_pool, second_weights, [1, 1, 1, 1], 'SAME') + second_bias
    second_relu = tf.nn.relu(second_conv)
    second_dropout = tf.layers.dropout(second_relu, dropout_prob, training=is_training)
    second_conv_shape = second_dropout.get_shape()
    second_conv_output_width = second_conv_shape[2]
    second_conv_output_height = second_conv_shape[1]
    second_conv_element_count = int(
        second_conv_output_width * second_conv_output_height *
        second_filter_count)
    flattened_second_conv = tf.reshape(
        second_dropout, [-1, second_conv_element_count])

    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_conv_element_count, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(flattened_second_conv, final_fc_weights) + final_fc_bias
    return final_fc


def create_low_latency_conv_model(
        fingerprint_input, model_settings, dropout_prob=1.0, is_training=False):
    """Builds a convolutional model with low compute requirements.

    This is roughly the network labeled as 'cnn-one-fstride4' in the
    'Convolutional Neural Networks for Small-footprint Keyword Spotting' paper:
    http://www.isca-speech.org/archive/interspeech_2015/papers/i15_1478.pdf

    Here's the layout of the graph:

    (fingerprint_input)
            v
        [Conv2D]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v

    This produces slightly lower quality results than the 'conv' model, but needs
    fewer weight parameters and computations.

    During training, dropout nodes are introduced after the relu, controlled by a
    placeholder.

    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.
    """
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']
    fingerprint_4d = tf.reshape(fingerprint_input,
                                [-1, input_time_size, input_frequency_size, 1])
    first_filter_width = 8
    first_filter_height = input_time_size
    first_filter_count = 186
    first_filter_stride_x = 1
    first_filter_stride_y = 1
    first_weights = tf.Variable(
        tf.truncated_normal(
            [first_filter_height, first_filter_width, 1, first_filter_count],
            stddev=0.01))
    first_bias = tf.Variable(tf.zeros([first_filter_count]))
    first_conv = tf.nn.conv2d(fingerprint_4d, first_weights, [
        1, first_filter_stride_y, first_filter_stride_x, 1
    ], 'VALID') + first_bias
    first_relu = tf.nn.relu(first_conv)
    first_dropout = tf.layers.dropout(first_relu, dropout_prob, training=is_training)
    first_conv_output_width = math.floor(
        (input_frequency_size - first_filter_width + first_filter_stride_x) /
        first_filter_stride_x)
    first_conv_output_height = math.floor(
        (input_time_size - first_filter_height + first_filter_stride_y) /
        first_filter_stride_y)
    first_conv_element_count = int(
        first_conv_output_width * first_conv_output_height * first_filter_count)
    flattened_first_conv = tf.reshape(
        first_dropout, [-1, first_conv_element_count])

    first_fc_output_channels = 128
    first_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_conv_element_count, first_fc_output_channels], stddev=0.01))
    first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
    first_fc = tf.matmul(flattened_first_conv, first_fc_weights) + first_fc_bias
    second_fc_input = tf.layers.dropout(first_fc, dropout_prob, training=is_training)

    second_fc_output_channels = 128
    second_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
    second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
    second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
    final_fc_input = tf.layers.dropout(second_fc, dropout_prob, training=is_training)

    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_fc_output_channels, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    return final_fc


def create_low_latency_svdf_model(
        fingerprint_input, model_settings,
        runtime_settings, dropout_prob=1.0, is_training=False):
    """Builds an SVDF model with low compute requirements.

    This is based in the topology presented in the 'Compressing Deep Neural
    Networks using a Rank-Constrained Topology' paper:
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43813.pdf

    Here's the layout of the graph:

    (fingerprint_input)
            v
          [SVDF]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
          [Relu]
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v
        [MatMul]<-(weights)
            v
        [BiasAdd]<-(bias)
            v

    This model produces lower recognition accuracy than the 'conv' model above,
    but requires fewer weight parameters and, significantly fewer computations.

    During training, dropout nodes are introduced after the relu, controlled by a
    placeholder.

    Args:
      fingerprint_input: TensorFlow node that will output audio feature vectors.
      The node is expected to produce a 2D Tensor of shape:
        [batch, model_settings['dct_coefficient_count'] *
                model_settings['spectrogram_length']]
      with the features corresponding to the same time slot arranged contiguously,
      and the oldest slot at index [:, 0], and newest at [:, -1].
      model_settings: Dictionary of information about the model.
      is_training: Whether the model is going to be used for training.
      runtime_settings: Dictionary of information about the runtime.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.

    Raises:
        ValueError: If the inputs tensor is incorrectly shaped.
    """
    input_frequency_size = model_settings['dct_coefficient_count']
    input_time_size = model_settings['spectrogram_length']

    # Validation.
    input_shape = fingerprint_input.get_shape()
    if len(input_shape) != 2:
        raise ValueError('Inputs to `SVDF` should have rank == 2.')
    if input_shape[-1].value is None:
        raise ValueError('The last dimension of the inputs to `SVDF` '
                         'should be defined. Found `None`.')
    if input_shape[-1].value % input_frequency_size != 0:
        raise ValueError('Inputs feature dimension %d must be a multiple of '
                         'frame size %d', fingerprint_input.shape[-1].value,
                         input_frequency_size)

    # Set number of units (i.e. nodes) and rank.
    rank = 2
    num_units = 1280
    # Number of filters: pairs of feature and time filters.
    num_filters = rank * num_units
    # Create the runtime memory: [num_filters, batch, input_time_size]
    batch = 1
    memory = tf.Variable(tf.zeros([num_filters, batch, input_time_size]),
                         trainable=False, name='runtime-memory')

    is_training_value = tf.contrib.util.constant_value(is_training)
    # Determine the number of new frames in the input, such that we only operate
    # on those. For training we do not use the memory, and thus use all frames
    # provided in the input.
    # new_fingerprint_input: [batch, num_new_frames*input_frequency_size]
    if is_training_value:
        num_new_frames = input_time_size
    else:
        window_stride_ms = int(model_settings['window_stride_samples'] * 1000 /
                               model_settings['sample_rate'])
        num_new_frames = tf.cond(
            tf.equal(tf.count_nonzero(memory), 0),
            lambda: input_time_size,
            lambda: int(runtime_settings['clip_stride_ms'] / window_stride_ms))

    new_fingerprint_input = fingerprint_input[
                            :, -num_new_frames * input_frequency_size:]
    # Expand to add input channels dimension.
    new_fingerprint_input = tf.expand_dims(new_fingerprint_input, 2)

    # Create the frequency filters.
    weights_frequency = tf.Variable(
        tf.truncated_normal([input_frequency_size, num_filters], stddev=0.01))
    # Expand to add input channels dimensions.
    # weights_frequency: [input_frequency_size, 1, num_filters]
    weights_frequency = tf.expand_dims(weights_frequency, 1)
    # Convolve the 1D feature filters sliding over the time dimension.
    # activations_time: [batch, num_new_frames, num_filters]
    activations_time = tf.nn.conv1d(
        new_fingerprint_input, weights_frequency, input_frequency_size, 'VALID')
    # Rearrange such that we can perform the batched matmul.
    # activations_time: [num_filters, batch, num_new_frames]
    activations_time = tf.transpose(activations_time, perm=[2, 0, 1])

    # Runtime memory optimization.
    if not is_training_value:
        # We need to drop the activations corresponding to the oldest frames, and
        # then add those corresponding to the new frames.
        new_memory = memory[:, :, num_new_frames:]
        new_memory = tf.concat([new_memory, activations_time], 2)
        tf.assign(memory, new_memory)
        activations_time = new_memory

    # Create the time filters.
    weights_time = tf.Variable(
        tf.truncated_normal([num_filters, input_time_size], stddev=0.01))
    # Apply the time filter on the outputs of the feature filters.
    # weights_time: [num_filters, input_time_size, 1]
    # outputs: [num_filters, batch, 1]
    weights_time = tf.expand_dims(weights_time, 2)
    outputs = tf.matmul(activations_time, weights_time)
    # Split num_units and rank into separate dimensions (the remaining
    # dimension is the input_shape[0] -i.e. batch size). This also squeezes
    # the last dimension, since it's not used.
    # [num_filters, batch, 1] => [num_units, rank, batch]
    outputs = tf.reshape(outputs, [num_units, rank, -1])
    # Sum the rank outputs per unit => [num_units, batch].
    units_output = tf.reduce_sum(outputs, axis=1)
    # Transpose to shape [batch, num_units]
    units_output = tf.transpose(units_output)

    # Appy bias.
    bias = tf.Variable(tf.zeros([num_units]))
    first_bias = tf.nn.bias_add(units_output, bias)

    # Relu.
    first_relu = tf.nn.relu(first_bias)

    first_dropout = tf.layers.dropout(first_relu, dropout_prob, training=is_training)

    first_fc_output_channels = 256
    first_fc_weights = tf.Variable(
        tf.truncated_normal([num_units, first_fc_output_channels], stddev=0.01))
    first_fc_bias = tf.Variable(tf.zeros([first_fc_output_channels]))
    first_fc = tf.matmul(first_dropout, first_fc_weights) + first_fc_bias
    second_fc_input = tf.layers.dropout(first_fc, dropout_prob, training=is_training)

    second_fc_output_channels = 256
    second_fc_weights = tf.Variable(
        tf.truncated_normal(
            [first_fc_output_channels, second_fc_output_channels], stddev=0.01))
    second_fc_bias = tf.Variable(tf.zeros([second_fc_output_channels]))
    second_fc = tf.matmul(second_fc_input, second_fc_weights) + second_fc_bias
    final_fc_input = tf.layers.dropout(second_fc, dropout_prob, training=is_training)

    label_count = model_settings['label_count']
    final_fc_weights = tf.Variable(
        tf.truncated_normal(
            [second_fc_output_channels, label_count], stddev=0.01))
    final_fc_bias = tf.Variable(tf.zeros([label_count]))
    final_fc = tf.matmul(final_fc_input, final_fc_weights) + final_fc_bias
    return final_fc
