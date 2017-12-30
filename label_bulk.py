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
r"""Runs a trained audio graph against a WAVE file and reports the results.

The model, labels and .wav file specified in the arguments will be loaded, and
then the predictions from running the model against the audio data will be
printed to the console. This is a useful script for sanity checking trained
models, and as an example of how to use an audio model from Python.

Here's an example of running it:

python tensorflow/examples/speech_commands/label_wav.py \
--graph=/tmp/my_frozen_graph.pb \
--labels=/tmp/speech_commands_train/conv_labels.txt \
--wav=/tmp/speech_dataset/left/a5d485dc_nohash_0.wav

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os
import re
from collections import OrderedDict

import tensorflow as tf
from tensorflow.python.ops import io_ops
import numpy as np
import pandas as pd
import input_data
from models.model_factory import *

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio

# pylint: enable=unused-import

FLAGS = None


def find_inputs(
        folder,
        types=['.wav'],
        sort=True):

    filenames = []
    for root, subdirs, files in os.walk(folder, topdown=False):
        for f in files:
            base, ext = os.path.splitext(f)
            if ext.lower() in types:
                filenames.append(os.path.join(root, f))
    if sort:
        filenames = sorted(filenames)
    return filenames


def _load_sample(
        wav_filename,
        model_settings):
    """Creates an audio model with the nodes needed for inference.

    Uses the supplied arguments to create a model, and inserts the input and
    output nodes that are needed to use the graph for inference.

    Args:
      wanted_words: Comma-separated list of the words we're trying to recognize.
      sample_rate: How many samples per second are in the input audio files.
      clip_duration_ms: How many samples to analyze for the audio pattern.
      clip_stride_ms: How often to run recognition. Useful for models with cache.
      window_size_ms: Time slice duration to estimate frequencies from.
      window_stride_ms: How far apart time slices should be.
      dct_coefficient_count: Number of frequency bands to analyze.
      model: Name of the kind of model to generate.
    """
    wav_loader = io_ops.read_file(wav_filename)

    decoded_sample_data = contrib_audio.decode_wav(
        wav_loader,
        desired_channels=1,
        desired_samples=model_settings['desired_samples'],
        name='decoded_sample_data')

    if model_settings['input_format'] == 'raw':
        print(decoded_sample_data.audio.shape)
        reshaped_input = tf.reshape(decoded_sample_data.audio, [
            -1, model_settings['desired_samples']
        ])
        print(reshaped_input.shape)
    else:
        spectrogram = contrib_audio.audio_spectrogram(
            decoded_sample_data.audio,
            window_size=model_settings['window_size_samples'],
            stride=model_settings['window_stride_samples'],
            magnitude_squared=True)

        fingerprint_input = contrib_audio.mfcc(
            spectrogram,
            decoded_sample_data.sample_rate,
            lower_frequency_limit=model_settings['lower_frequency_limit'],
            upper_frequency_limit=model_settings['upper_frequency_limit'],
            filterbank_channel_count=model_settings['filterbank_channel_count'],
            dct_coefficient_count=model_settings['dct_coefficient_count'])
        fingerprint_frequency_size = model_settings['dct_coefficient_count']
        fingerprint_time_size = model_settings['spectrogram_length']

        reshaped_input = tf.reshape(fingerprint_input, [
            -1, fingerprint_time_size * fingerprint_frequency_size
        ])

    return reshaped_input


def get_inputs(folder, model_settings, batch_size=256, num_threads=1):
    def _parse_data(filename):
        fingerprint = _load_sample(filename, model_settings)
        return fingerprint

    filenames = find_inputs(folder)
    dataset = tf.data.Dataset.from_tensor_slices(np.array(filenames))
    dataset = dataset.map(_parse_data, num_parallel_calls=num_threads)
    dataset = dataset.prefetch(num_threads * batch_size)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(1)

    return dataset, [os.path.basename(f) for f in filenames]


def load_labels(filename):
    """Read in labels, one label per line."""
    return [line.strip().replace('_', '') for line in tf.gfile.GFile(filename)]


def _prepare_model_settings(num_words):
    if any(n in FLAGS.model for n in ['conv1d']):
        model_settings = prepare_model_settings(
            num_words,
            FLAGS.sample_rate,
            FLAGS.clip_duration_ms,
            input_format='raw')
    elif any(n in FLAGS.model for n in ['vggish', 'nasnet', 'resnet']):
        model_settings = prepare_model_settings(
            num_words,
            FLAGS.sample_rate,
            FLAGS.clip_duration_ms,
            input_format='spectrogram',
            window_size_ms=FLAGS.window_size_ms,
            window_stride_ms=FLAGS.window_stride_ms,
            lower_frequency_limit=125,
            upper_frequency_limit=7500,
            filterbank_channel_count=64,
            dct_coefficient_count=64)
    else:
        model_settings = prepare_model_settings(
            num_words,
            FLAGS.sample_rate,
            FLAGS.clip_duration_ms,
            input_format='spectrogram',
            window_size_ms=FLAGS.window_size_ms,
            window_stride_ms=FLAGS.window_stride_ms,
            lower_frequency_limit=20,
            upper_frequency_limit=4000,
            filterbank_channel_count=FLAGS.dct_coefficient_count,
            dct_coefficient_count=FLAGS.dct_coefficient_count)
    return model_settings


def main(_):
    """Loads the model and labels, and runs the inference to print predictions."""
    if not FLAGS.wav or not tf.gfile.Exists(FLAGS.wav):
        tf.logging.fatal('Audio file does not exist %s', FLAGS.wav)

    if not FLAGS.labels or not tf.gfile.Exists(FLAGS.labels):
        tf.logging.fatal('Labels file does not exist %s', FLAGS.labels)

    labels_list = load_labels(FLAGS.labels)
    words_list = input_data.prepare_words_list(FLAGS.wanted_words.split(','))
    model_settings = _prepare_model_settings(len(words_list))
    runtime_settings = {'clip_stride_ms': FLAGS.clip_stride_ms}
    print('Model settings:')
    for k, v in model_settings.items():
        print(k, v)

    dataset, filenames = get_inputs(
        FLAGS.wav,
        model_settings,
        batch_size=FLAGS.batch_size)
    iterator = dataset.make_one_shot_iterator()
    inputs = iterator.get_next()

    net = create_model(
        inputs, model_settings, FLAGS.model,
        runtime_settings=runtime_settings)
    if isinstance(net, tuple):
        logits, endpoints = net
    else:
        logits, endpoints = net, {}
    predicted_indices_op = tf.argmax(logits, 1)
    predicted_probs_op = tf.reduce_max(tf.nn.softmax(logits), 1)

    with tf.Session() as sess:
        load_variables_from_checkpoint(sess, FLAGS.checkpoint)
        num_samples = len(filenames)
        labels_arr = np.array(labels_list)
        label_outputs = []
        prob_outputs = []

        try:
            for i in range(0, num_samples, FLAGS.batch_size):
                batch_predictions, batch_probs = sess.run(
                    [predicted_indices_op, predicted_probs_op])
                batch_labels = labels_arr[batch_predictions]
                curr_batch_size = min(FLAGS.batch_size, num_samples - i)
                if curr_batch_size < FLAGS.batch_size:
                    batch_labels = batch_labels[:curr_batch_size]
                    batch_probs = batch_probs[:curr_batch_size]
                label_outputs.append(batch_labels)
                prob_outputs.append(batch_probs)
                print("%d of %d" % (i, num_samples))
        except KeyboardInterrupt:
            pass

        label_outputs = np.concatenate(label_outputs, axis=0)
        prob_outputs = np.concatenate(prob_outputs, axis=0)
        print(prob_outputs.shape)
        print(label_outputs.shape, len(filenames))
        data_dict = OrderedDict([
            ('fname', filenames[:len(label_outputs)]),
            ('label', label_outputs)])
        if FLAGS.include_probs:
            data_dict['prob'] = prob_outputs
        submission = pd.DataFrame(data=data_dict)
        print(submission.shape)
        submission.to_csv('./submission.csv', index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--wav', type=str, default='', help='Audio file to be identified.')
    parser.add_argument(
        '--labels', type=str, default='', help='Path to file containing labels.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='How many items to train with at once', )
    parser.add_argument(
        '--how_many_labels',
        type=int,
        default=3,
        help='Number of results to show.')
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Expected sample rate of the wavs', )
    parser.add_argument(
        '--clip_duration_ms',
        type=int,
        default=1000,
        help='Expected duration in milliseconds of the wavs', )
    parser.add_argument(
        '--clip_stride_ms',
        type=int,
        default=30,
        help='How often to run recognition. Useful for models with cache.', )
    parser.add_argument(
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long the stride is between spectrogram timeslices', )
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint', )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model',
        type=str,
        default='conv',
        help='What model architecture to use')
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
                #'zero,one,two,three,four,five,six,seven,eight,nine,'
                #'bird,dog,cat,bed,house,tree,marvin,sheila,happy,wow',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--include_probs',
        type=bool,
        default=False,
        help='Include probability in output'
    )

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
