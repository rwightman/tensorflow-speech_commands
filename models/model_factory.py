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

from models.conv1d import *
from models.conv2d import create_conv2d_model
from models.original import create_conv_model, create_low_latency_conv_model, \
    create_low_latency_svdf_model
from models.vggish_slim import create_vggish_slim
from models.nasnet.nasnet import create_nasnetm
from models.resnet_v2 import create_resnet_v2_50, create_resnet_v2_xx


def prepare_model_settings(
        label_count,
        sample_rate,
        clip_duration_ms,
        input_format='spectrogram',
        window_size_ms=30,
        window_stride_ms=10,
        lower_frequency_limit=20,
        upper_frequency_limit=4000,
        filterbank_channel_count=40,
        dct_coefficient_count=40):
    """Calculates common settings needed for all models.

    Args:
      label_count: How many classes are to be recognized.
      sample_rate: Number of audio samples per second.
      clip_duration_ms: Length of each audio clip to be analyzed.
      window_size_ms: Duration of frequency analysis window.
      window_stride_ms: How far to move in time between frequency windows.
      dct_coefficient_count: Number of frequency bins to use for analysis.

    Returns:
      Dictionary containing common settings.
    """
    desired_samples = int(sample_rate * clip_duration_ms / 1000)
    if input_format == 'spectrogram':
        window_size_samples = int(sample_rate * window_size_ms / 1000)
        window_stride_samples = int(sample_rate * window_stride_ms / 1000)
        length_minus_window = (desired_samples - window_size_samples)
        if length_minus_window < 0:
            spectrogram_length = 0
        else:
            spectrogram_length = 1 + int(length_minus_window / window_stride_samples)
        fingerprint_size = dct_coefficient_count * spectrogram_length
        return {
            'desired_samples': desired_samples,
            'window_size_samples': window_size_samples,
            'window_stride_samples': window_stride_samples,
            'spectrogram_length': spectrogram_length,
            'lower_frequency_limit': lower_frequency_limit,
            'upper_frequency_limit': upper_frequency_limit,
            'filterbank_channel_count': filterbank_channel_count,
            'dct_coefficient_count': dct_coefficient_count,
            'sample_size': fingerprint_size,
            'label_count': label_count,
            'sample_rate': sample_rate,
            'input_format': input_format,
        }
    elif input_format == 'raw':
        return {
            'desired_samples': desired_samples,
            'label_count': label_count,
            'sample_rate': sample_rate,
            'sample_size': desired_samples,
            'input_format': input_format
        }
    else:
        assert False, "Invalid input format"


def create_model(sample_input, model_settings, model_architecture,
                 dropout_prob=1.0, is_training=False, runtime_settings=None):
    """Builds a model of the requested architecture compatible with the settings.

    There are many possible ways of deriving predictions from a spectrogram
    input, so this function provides an abstract interface for creating different
    kinds of models in a black-box way. You need to pass in a TensorFlow node as
    the 'fingerprint' input, and this should output a batch of 1D features that
    describe the audio. Typically this will be derived from a spectrogram that's
    been run through an MFCC, but in theory it can be any feature vector of the
    size specified in model_settings['fingerprint_size'].

    The function will build the graph it needs in the current TensorFlow graph,
    and return the tensorflow output that will contain the 'logits' input to the
    softmax prediction process. If training flag is on, it will also return a
    placeholder node that can be used to control the dropout amount.

    See the implementations below for the possible model architectures that can be
    requested.

    Args:
      sample_input: TensorFlow node that will output audio feature vectors.
      model_settings: Dictionary of information about the model.
      model_architecture: String specifying which kind of model to create.
      is_training: Whether the model is going to be used for training.
      runtime_settings: Dictionary of information about the runtime.

    Returns:
      TensorFlow node outputting logits results, and optionally a dropout
      placeholder.

    Raises:
      Exception: If the architecture type isn't recognized.
    """
    if model_architecture == 'conv':
        return create_conv_model(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'conv1d_basic2':
        return create_conv1d_basic2_model(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'conv1d_basic3':
        return create_conv1d_basic3_model(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'conv1d_a':
        return create_conv1d_a_model(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'conv1d_b':
        return create_conv1d_b_model(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'conv1d_c':
        return create_conv1d_c_model(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'conv1d_d':
        return create_conv1d_d_model(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'conv2d':
        return create_conv2d_model(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'vggish' or model_architecture == 'vggish_slim':
        return create_vggish_slim(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'resnetxx':
        return create_resnet_v2_xx(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'resnet50':
        return create_resnet_v2_50(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'nasnetm':
        return create_nasnetm(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'low_latency_conv':
        return create_low_latency_conv_model(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training)
    elif model_architecture == 'low_latency_svdf':
        return create_low_latency_svdf_model(
            sample_input, model_settings,
            dropout_prob=dropout_prob, is_training=is_training,
            runtime_settings=runtime_settings)
    else:
        raise Exception('model_architecture argument "' + model_architecture +
                        '" not recognized, should be one of "single_fc", "conv",' +
                        ' "low_latency_conv, or "low_latency_svdf"')


def load_variables_from_checkpoint(sess, start_checkpoint):
    """Utility function to centralize checkpoint restoration.

    Args:
      sess: TensorFlow session.
      start_checkpoint: Path to saved checkpoint on disk.
    """
    saver = tf.train.Saver(tf.global_variables())
    saver.restore(sess, start_checkpoint)

