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
r"""Simple speech recognition to spot a limited number of keywords.

This is a self-contained example script that will train a very basic audio
recognition model in TensorFlow. It downloads the necessary training data and
runs with reasonable defaults to train within a few hours even only using a CPU.
For more information, please see
https://www.tensorflow.org/tutorials/audio_recognition.

It is intended as an introduction to using neural networks for audio
recognition, and is not a full speech recognition system. For more advanced
speech systems, I recommend looking into Kaldi. This network uses a keyword
detection style to spot discrete words from a small vocabulary, consisting of
"yes", "no", "up", "down", "left", "right", "on", "off", "stop", and "go".

To run the training process, use:

bazel run tensorflow/examples/speech_commands:train

This will write out checkpoints to /tmp/speech_commands_train/, and will
download over 1GB of open source training data, so you'll need enough free space
and a good internet connection. The default data is a collection of thousands of
one-second .wav files, each containing one spoken word. This data set is
collected from https://aiyprojects.withgoogle.com/open_speech_recording, please
consider contributing to help improve this and other models!

As training progresses, it will print out its accuracy metrics, which should
rise above 90% by the end. Once it's complete, you can run the freeze script to
get a binary GraphDef that you can easily deploy on mobile applications.

If you want to train on your own data, you'll need to create .wavs with your
recordings, all at a consistent length, and then arrange them into subfolders
organized by label. For example, here's a possible file structure:

my_wavs >
  up >
    audio_0.wav
    audio_1.wav
  down >
    audio_2.wav
    audio_3.wav
  other>
    audio_4.wav
    audio_5.wav

You'll also need to tell the script what labels to look for, using the
`--wanted_words` argument. In this case, 'up,down' might be what you want, and
the audio in the 'other' folder would be used to train an 'unknown' category.

To pull this all together, you'd run:

bazel run tensorflow/examples/speech_commands:train -- \
--data_dir=my_wavs --wanted_words=up,down

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

import input_data
from models.model_factory import *
from tensorflow.python.platform import gfile

FLAGS = None


def main(_):
    # We want to see all the logging messages for this tutorial.
    tf.logging.set_verbosity(tf.logging.INFO)

    # Start a new TensorFlow session.
    sess = tf.InteractiveSession()

    num_words = len(input_data.prepare_words_list(FLAGS.wanted_words.split(',')))
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

    print('Model settings:')
    for k, v in model_settings.items():
        print(k, v)

    audio_processor = input_data.AudioProcessor(
        FLAGS.data_dir,
        FLAGS.silence_percentage,
        FLAGS.unknown_percentage,
        FLAGS.wanted_words.split(','),
        FLAGS.validation_percentage,
        FLAGS.testing_percentage, model_settings)
    sample_size = model_settings['sample_size']
    label_count = model_settings['label_count']
    time_shift_samples = int((FLAGS.time_shift_ms * FLAGS.sample_rate) / 1000)

    # Figure out the learning rates for each training phase. Since it's often
    # effective to have high learning rates at the start of training, followed by
    # lower levels towards the end, the number of steps and learning rates can be
    # specified as comma-separated lists to define the rate at each stage. For
    # example --how_many_training_steps=10000,3000 --learning_rate=0.001,0.0001
    # will run 13,000 training loops in total, with a rate of 0.001 for the first
    # 10,000, and 0.0001 for the final 3,000.
    training_steps_list = list(map(int, FLAGS.how_many_training_steps.split(',')))
    learning_rates_list = list(map(float, FLAGS.learning_rate.split(',')))
    if len(training_steps_list) != len(learning_rates_list):
        raise Exception(
            '--how_many_training_steps and --learning_rate must be equal length '
            'lists, but are %d and %d long instead' % (len(training_steps_list),
                                                       len(learning_rates_list)))

    # Create placeholder variables
    sample_input = tf.placeholder(
        tf.float32, [None, sample_size], name='sample_input')
    ground_truth_input = tf.placeholder(
        tf.float32, [None, label_count], name='groundtruth_input')
    learning_rate_input = tf.placeholder(
        tf.float32, [], name='learning_rate_input')
    is_training_input = tf.placeholder_with_default(True, [], name='is_training_input')

    # Instantiate model graph
    net = create_model(
        sample_input,
        model_settings,
        FLAGS.model,
        dropout_prob=0.6,
        is_training=is_training_input)
    if isinstance(net, tuple):
        logits, endpoints = net
    else:
        logits, endpoints = net, {}
    print(endpoints)
    print(net)

    for variable in slim.get_model_variables():
        tf.summary.histogram(variable.op.name, variable)

    # Define loss and optimizer
    cross_entropy_op = tf.losses.softmax_cross_entropy(
        onehot_labels=ground_truth_input, logits=logits, label_smoothing=0.05)
    if 'AuxLogits' in endpoints:
        tf.losses.softmax_cross_entropy(
            onehot_labels=ground_truth_input, logits=endpoints['AuxLogits'],
            label_smoothing=0.05, weights=0.4, scope='aux_loss')

    for loss in tf.get_collection(tf.GraphKeys.LOSSES):
        tf.summary.scalar('losses/%s' % loss.op.name, loss)
    total_loss = tf.losses.get_total_loss()
    tf.summary.scalar('total_loss', total_loss)

    if FLAGS.opt == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate_input)
    else:
        optimizer = tf.train.MomentumOptimizer(
            learning_rate_input,
            momentum=0.9,
            use_nesterov=True)

    train_step = slim.learning.create_train_op(
        total_loss,
        optimizer,
        clip_gradient_norm=10.0,
        check_numerics=FLAGS.check_nans)

    predicted_indices = tf.argmax(logits, 1)
    expected_indices = tf.argmax(ground_truth_input, 1)
    correct_prediction = tf.equal(predicted_indices, expected_indices)
    confusion_matrix = tf.confusion_matrix(
        expected_indices, predicted_indices, num_classes=label_count)
    evaluation_step = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', evaluation_step)

    global_step = tf.contrib.framework.get_or_create_global_step()
    increment_global_step = tf.assign(global_step, global_step + 1)

    saver = tf.train.Saver(tf.global_variables())

    exp_name = '-'.join([
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        FLAGS.model])
    output_dir = get_outdir(FLAGS.output_dir, 'train', exp_name)

    # Merge all the summaries and write them out to /tmp/retrain_logs (by default)
    merged_summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(output_dir + '/train', sess.graph)
    validation_writer = tf.summary.FileWriter(output_dir + '/validation')

    tf.global_variables_initializer().run()

    start_step = 1

    if FLAGS.start_checkpoint:
        load_variables_from_checkpoint(sess, FLAGS.start_checkpoint)
        start_step = global_step.eval(session=sess)

    tf.logging.info('Training from step: %d ', start_step)

    # Save graph.pbtxt.
    tf.train.write_graph(sess.graph_def, output_dir, FLAGS.model + '.pbtxt')

    # Save list of words.
    with gfile.GFile(
            os.path.join(output_dir, FLAGS.model + '_labels.txt'), 'w') as f:
        f.write('\n'.join(audio_processor.words_list))

    # Training loop.
    training_steps_max = np.sum(training_steps_list)
    for training_step in range(start_step, training_steps_max + 1):
        is_last_step = (training_step == training_steps_max)

        # Figure out what the current learning rate is.
        training_steps_sum = 0
        for i in range(len(training_steps_list)):
            training_steps_sum += training_steps_list[i]
            if training_step <= training_steps_sum:
                learning_rate_value = learning_rates_list[i]
                break

        # Pull the audio samples we'll use for training.
        train_samples, train_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, 0, model_settings, FLAGS.background_frequency,
            FLAGS.background_volume, time_shift_samples, 'training', sess)

        # Run the graph with this batch of training data.
        ops = {
            'evaluation_step': evaluation_step,
            'train_step': train_step,
            'cross_entropy_op': cross_entropy_op,
            'increment_global_step': increment_global_step,
        }
        if (training_step % 500) == 0 or is_last_step:
            ops['merged_summaries'] = merged_summaries

        results = sess.run(
            ops,
            feed_dict={
                sample_input: train_samples,
                ground_truth_input: train_ground_truth,
                learning_rate_input: learning_rate_value,
                is_training_input: True,
            })
        train_accuracy = results['evaluation_step']
        cross_entropy_value = results['train_step']  #results['cross_entropy_op']
        if 'merged_summaries' in results:
            train_writer.add_summary(results['merged_summaries'], training_step)
        tf.logging.info('Step #%d: rate %f, accuracy %.1f%%, cross entropy %f' %
                        (training_step, learning_rate_value, train_accuracy * 100,
                         cross_entropy_value))

        if (training_step % FLAGS.eval_step_interval) == 0 or is_last_step:
            eval_set_size = audio_processor.set_size('validation')
            total_accuracy = 0
            total_conf_matrix = None
            for i in range(0, eval_set_size, FLAGS.batch_size):
                validation_samples, validation_ground_truth = audio_processor.get_data(
                    FLAGS.batch_size, i, model_settings, 0.0,
                    0.0, 0, 'validation', sess)

                # Run a validation step and capture training summaries for TensorBoard
                # with the `merged` op.
                validation_summary, validation_accuracy, conf_matrix = sess.run(
                    [merged_summaries, evaluation_step, confusion_matrix],
                    feed_dict={
                        sample_input: validation_samples,
                        ground_truth_input: validation_ground_truth,
                        is_training_input: False,
                    })
                validation_writer.add_summary(validation_summary, training_step)
                batch_size = min(FLAGS.batch_size, eval_set_size - i)
                total_accuracy += (validation_accuracy * batch_size) / eval_set_size
                if total_conf_matrix is None:
                    total_conf_matrix = conf_matrix
                else:
                    total_conf_matrix += conf_matrix
            tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
            tf.logging.info('Step %d: Validation accuracy = %.1f%% (N=%d)' %
                            (training_step, total_accuracy * 100, eval_set_size))

        # Save the model checkpoint periodically.
        if (training_step % FLAGS.save_step_interval == 0 or
                    training_step == training_steps_max):
            checkpoint_path = os.path.join(output_dir, FLAGS.model + '.ckpt')
            tf.logging.info('Saving to "%s-%d"', checkpoint_path, training_step)
            saver.save(sess, checkpoint_path, global_step=training_step)

    test_set_size = audio_processor.set_size('testing')
    tf.logging.info('set_size=%d', test_set_size)
    total_accuracy = 0
    total_conf_matrix = None
    for i in range(0, test_set_size, FLAGS.batch_size):
        test_samples, test_ground_truth = audio_processor.get_data(
            FLAGS.batch_size, i, model_settings, 0.0, 0.0, 0, 'testing', sess)
        test_accuracy, conf_matrix = sess.run(
            [evaluation_step, confusion_matrix],
            feed_dict={
                sample_input: test_samples,
                ground_truth_input: test_ground_truth,
                is_training_input: False,
            })
        batch_size = min(FLAGS.batch_size, test_set_size - i)
        total_accuracy += (test_accuracy * batch_size) / test_set_size
        if total_conf_matrix is None:
            total_conf_matrix = conf_matrix
        else:
            total_conf_matrix += conf_matrix
    tf.logging.info('Confusion Matrix:\n %s' % (total_conf_matrix))
    tf.logging.info('Final test accuracy = %.1f%% (N=%d)' %
                    (total_accuracy * 100, test_set_size))


def get_outdir(path, *paths, inc=False):
    outdir = os.path.join(path, *paths)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    elif inc:
        count = 1
        outdir_inc = outdir + '-' + str(count)
        while os.path.exists(outdir_inc):
            count = count + 1
            outdir_inc = outdir + '-' + str(count)
            assert count < 100
        outdir = outdir_inc
        os.makedirs(outdir)
    return outdir


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/data/f/commands/train/audio',
        help="""\
      Where to download the speech training data to.
      """)
    parser.add_argument(
        '--background_volume',
        type=float,
        default=0.2,
        help="""\
      How loud the background noise should be, between 0 and 1.
      """)
    parser.add_argument(
        '--background_frequency',
        type=float,
        default=0.8,
        help="""\
      How many of the training samples have background noise mixed in.
      """)
    parser.add_argument(
        '--silence_percentage',
        type=float,
        default=10.0,
        help="""\
      How much of the training data should be silence.
      """)
    parser.add_argument(
        '--unknown_percentage',
        type=float,
        default=9.0,
        help="""\
      How much of the training data should be unknown words.
      """)
    parser.add_argument(
        '--time_shift_ms',
        type=float,
        default=100.0,
        help="""\
      Range to randomly shift the training audio by in time.
      """)
    parser.add_argument(
        '--testing_percentage',
        type=int,
        default=0,
        help='What percentage of wavs to use as a test set.')
    parser.add_argument(
        '--validation_percentage',
        type=int,
        default=9,
        help='What percentage of wavs to use as a validation set.')
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
        '--window_size_ms',
        type=float,
        default=30.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--window_stride_ms',
        type=float,
        default=10.0,
        help='How long each spectrogram timeslice is', )
    parser.add_argument(
        '--dct_coefficient_count',
        type=int,
        default=40,
        help='How many bins to use for the MFCC fingerprint', )
    parser.add_argument(
        '--eval_step_interval',
        type=int,
        default=500,
        help='How often to evaluate the training results.')
    parser.add_argument(
        '--how_many_training_steps',
        type=str,
        default='12000, 6000', #'2000,10000,5000,3000',
        help='How many training loops to run', )
    parser.add_argument(
        '--learning_rate',
        type=str,
        default='0.0001,0.00001', #'.00001,0.001,0.0001,0.00001',
        help='How large a learning rate to use when training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=512,
        help='How many items to train with at once', )
    parser.add_argument(
        '--wanted_words',
        type=str,
        default='yes,no,up,down,left,right,on,off,stop,go',
                #'zero,one,two,three,four,five,six,seven,eight,nine,'
                #'bird,dog,cat,bed,house,tree,marvin,sheila,happy,wow',
        help='Words to use (others will be added to an unknown label)', )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data/x/commands_out',
        help='Directory to write event logs and checkpoint.')
    parser.add_argument(
        '--save_step_interval',
        type=int,
        default=500,
        help='Save model checkpoint every save_steps.')
    parser.add_argument(
        '--start_checkpoint',
        type=str,
        default='',
        help='If specified, restore this pretrained model before any training.')
    parser.add_argument(
        '--model',
        type=str,
        default='conv',
        help='What model architecture to use')
    parser.add_argument(
        '--opt',
        type=str,
        default='momentum',
        help='What optimizer to use')
    parser.add_argument(
        '--check_nans',
        type=bool,
        default=False,
        help='Whether to check for invalid numbers during processing')

    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
