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

import hashlib
import math
import os.path
import random
import re
import sys

import numpy as np
import tensorflow as tf
import librosa as lr
import scipy.io.wavfile as wf

from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
from tensorflow.python.ops import io_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat

import multiprocessing as mp
try:
    mp.set_start_method('spawn')
except RuntimeError:
    pass

MAX_NUM_WAVS_PER_CLASS = 2 ** 27 - 1  # ~134M
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
RANDOM_SEED = 59185


def prepare_words_list(wanted_words):
    """Prepends common tokens to the custom word list.

    Args:
      wanted_words: List of strings containing the custom words.

    Returns:
      List with the standard silence and unknown tokens added.
    """
    return [SILENCE_LABEL, UNKNOWN_WORD_LABEL] + wanted_words


def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.

    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)

    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(compat.as_bytes(hash_name)).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result


def spectrogram_graph(input, sample_rate, model_settings):
    assert model_settings['input_format'] == 'spectrogram'

    def _do(input):
        input = tf.reshape(input, [input.shape[0], 1])
        # Run the spectrogram and MFCC ops to get a 2D 'fingerprint' of the audio.
        spectrogram = contrib_audio.audio_spectrogram(
            input,
            window_size=model_settings['window_size_samples'],
            stride=model_settings['window_stride_samples'],
            magnitude_squared=True)

        processed_input = contrib_audio.mfcc(
            spectrogram,
            sample_rate,
            lower_frequency_limit=model_settings['lower_frequency_limit'],
            upper_frequency_limit=model_settings['upper_frequency_limit'],
            filterbank_channel_count=model_settings['filterbank_channel_count'],
            dct_coefficient_count=model_settings['dct_coefficient_count'])
        return processed_input

    return tf.map_fn(_do, input)


def spectrogram_graph2(input, sample_rate, model_settings, power=True):
    """A batch + GPU capable spectrogram impl
    """

    # `stfts` is a complex64 Tensor representing the Short-time Fourier Transform of
    # each signal in `signals`. Its shape is [batch_size, ?, fft_unique_bins]
    # where fft_unique_bins = fft_length // 2 + 1. fft_length is set to nearest
    # enclosing power of two of window size.
    stfts = tf.contrib.signal.stft(
        input,
        frame_length=model_settings['window_size_samples'],
        frame_step=model_settings['window_stride_samples'])

    if power:
        # A power spectrogram is the squared magnitude of the complex-valued STFT.
        # A float32 Tensor of shape [batch_size, ?, 513].
        spectrograms = tf.real(stfts * tf.conj(stfts))
    else:
        # An energy spectrogram is the magnitude of the complex-valued STFT.
        # A float32 Tensor of shape [batch_size, ?, 513].
        spectrograms = tf.abs(stfts)

    # Warp the linear-scale, magnitude spectrograms into the mel-scale.
    num_spectrogram_bins = spectrograms.shape[-1].value
    lower_edge_hertz = model_settings['lower_frequency_limit']
    upper_edge_hertz = model_settings['upper_frequency_limit']
    num_mel_bins = model_settings['filterbank_channel_count']
    dct_coefficient_count = model_settings['dct_coefficient_count']
    linear_to_mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sample_rate,
        lower_edge_hertz, upper_edge_hertz)
    print(spectrograms.shape, linear_to_mel_weight_matrix.shape)

    # mel_spectrograms = tf.tensordot(
    #     magnitude_spectrograms, linear_to_mel_weight_matrix, 1)
    # Note: Shape inference for `tf.tensordot` does not currently handle this case.
    # mel_spectrograms.set_shape(
    #     magnitude_spectrograms.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    # print(mel_spectrograms.shape)

    mel_spectrograms = tf.matmul(
        tf.reshape(spectrograms, [-1, spectrograms.shape[-1]]),
        linear_to_mel_weight_matrix)
    mel_spectrograms = tf.reshape(
        mel_spectrograms,
        [-1, spectrograms.shape[1], linear_to_mel_weight_matrix.shape[-1]])
    print(mel_spectrograms.shape)

    log_mel_spectrograms = tf.log(mel_spectrograms + 1e6)

    mfccs = tf.contrib.signal.mfccs_from_log_mel_spectrograms(
        log_mel_spectrograms)[..., :dct_coefficient_count]
    print(mfccs.shape)

    return mfccs


class AudioProcessorRosa(object):
    """Handles loading, partitioning, and preparing audio training data."""

    def __init__(self, data_dir, silence_percentage, unknown_percentage,
                 wanted_words, validation_percentage, testing_percentage,
                 model_settings):
        self.data_dir = data_dir
        self.prepare_data_index_(
            silence_percentage, unknown_percentage, wanted_words,
            validation_percentage, testing_percentage)
        self.prepare_background_data_()

    def prepare_data_index_(
            self, silence_percentage, unknown_percentage,
            wanted_words, validation_percentage,
            testing_percentage):
        """Prepares a list of the samples organized by set and label.

        The training loop needs a list of all the available data, organized by
        which partition it should belong to, and with ground truth labels attached.
        This function analyzes the folders below the `data_dir`, figures out the
        right
        labels for each file based on the name of the subdirectory it belongs to,
        and uses a stable hash to assign it to a data set partition.

        Args:
          silence_percentage: How much of the resulting data should be background.
          unknown_percentage: How much should be audio outside the wanted classes.
          wanted_words: Labels of the classes we want to be able to recognize.
          validation_percentage: How much of the data set to use for validation.
          testing_percentage: How much of the data set to use for testing.

        Returns:
          Dictionary containing a list of file information for each set partition,
          and a lookup map for each class to determine its numeric index.

        Raises:
          Exception: If expected files are not found.
        """
        # Make sure the shuffling and picking of unknowns is deterministic.
        random.seed(RANDOM_SEED)
        wanted_words_index = {}
        for index, wanted_word in enumerate(wanted_words):
            wanted_words_index[wanted_word] = index + 2
        self.data_index = {'validation': [], 'testing': [], 'training': []}
        unknown_index = {'validation': [], 'testing': [], 'training': []}
        all_words = {}

        # Look through all the subfolders to find audio samples
        search_path = os.path.join(self.data_dir, '*', '*.wav')
        for wav_path in gfile.Glob(search_path):
            _, word = os.path.split(os.path.dirname(wav_path))
            word = word.lower()
            # Treat the '_background_noise_' folder as a special case, since we expect
            # it to contain long audio samples we mix in to improve training.
            if word == BACKGROUND_NOISE_DIR_NAME:
                continue
            all_words[word] = True
            set_index = which_set(wav_path, validation_percentage, testing_percentage)
            # If it's a known class, store its detail, otherwise add it to the list
            # we'll use to train the unknown label.
            if word in wanted_words_index:
                self.data_index[set_index].append({'label': word, 'file': wav_path})
            else:
                unknown_index[set_index].append({'label': word, 'file': wav_path})
        if not all_words:
            raise Exception('No .wavs found at ' + search_path)
        for index, wanted_word in enumerate(wanted_words):
            if wanted_word not in all_words:
                raise Exception('Expected to find ' + wanted_word +
                                ' in labels but only found ' +
                                ', '.join(all_words.keys()))

        # We need an arbitrary file to load as the input for the silence samples.
        # It's multiplied by zero later, so the content doesn't matter.
        silence_wav_path = self.data_index['training'][0]['file']
        for set_index in ['validation', 'testing', 'training']:
            set_size = len(self.data_index[set_index])
            silence_size = int(math.ceil(set_size * silence_percentage / 100))
            for _ in range(silence_size):
                self.data_index[set_index].append({
                    'label': SILENCE_LABEL,
                    'file': silence_wav_path
                })

            # Pick some unknowns to add to each partition of the data set.
            random.shuffle(unknown_index[set_index])
            unknown_size = int(math.ceil(set_size * unknown_percentage / 100))
            self.data_index[set_index].extend(unknown_index[set_index][:unknown_size])

        # Make sure the ordering is random.
        for set_index in ['validation', 'testing', 'training']:
            random.shuffle(self.data_index[set_index])

        # Prepare the rest of the result data structure.
        self.words_list = prepare_words_list(wanted_words)
        self.word_to_index = {}
        for word in all_words:
            if word in wanted_words_index:
                self.word_to_index[word] = wanted_words_index[word]
            else:
                self.word_to_index[word] = UNKNOWN_WORD_INDEX
        self.word_to_index[SILENCE_LABEL] = SILENCE_INDEX

    def prepare_background_data_(self):
        """Searches a folder for background noise audio, and loads it into memory.

        It's expected that the background audio samples will be in a subdirectory
        named '_background_noise_' inside the 'data_dir' folder, as .wavs that match
        the sample rate of the training data, but can be much longer in duration.

        If the '_background_noise_' folder doesn't exist at all, this isn't an
        error, it's just taken to mean that no background noise augmentation should
        be used. If the folder does exist, but it's empty, that's treated as an
        error.

        Returns:
          List of raw PCM-encoded audio samples of background noise.

        Raises:
          Exception: If files aren't found in the folder.
        """
        self.background_data = []
        background_dir = os.path.join(self.data_dir, BACKGROUND_NOISE_DIR_NAME)
        if not os.path.exists(background_dir):
            return self.background_data

        search_path = os.path.join(
            self.data_dir, BACKGROUND_NOISE_DIR_NAME, '*.wav')
        for wav_path in gfile.Glob(search_path):
            wav_rate, wav_audio = wf.read(wav_path)
            self.background_data.append(wav_audio.astype(np.float32))
        if not self.background_data:
            raise Exception('No background wav files were found in ' + search_path)

    def set_size(self, mode):
        """Calculates the number of samples in the dataset partition.

        Args:
          mode: Which partition, must be 'training', 'validation', or 'testing'.

        Returns:
          Number of samples in the partition.
        """
        return len(self.data_index[mode])

    def get_data(
            self,
            how_many, offset, model_settings,
            background_frequency=0.0,
            background_volume_range=0.0,
            pitch_shift_frequency=0.0,
            pitch_shift=0.0,
            time_stretch_frequency=0.0,
            time_stretch=0.0,
            time_shift=0,
            mode='training',
            sess=None):
        """Gather samples from the data set, applying transformations as needed.

        When the mode is 'training', a random selection of samples will be returned,
        otherwise the first N clips in the partition will be used. This ensures that
        validation always uses the same samples, reducing noise in the metrics.

        Args:
          how_many: Desired number of samples to return. -1 means the entire
            contents of this partition.
          offset: Where to start when fetching deterministically.
          model_settings: Information about the current model being trained.
          background_frequency: How many clips will have background noise, 0.0 to
            1.0.
          background_volume_range: How loud the background noise will be.
          time_shift: How much to randomly shift the clips by in time.
          mode: Which partition to use, must be 'training', 'validation', or
            'testing'.
          sess: TensorFlow session that was active when processor was created.

        Returns:
          List of sample data for the transformed samples, and list of labels in
          one-hot form.
        """
        # Pick one of the partitions to choose samples from.
        candidates = self.data_index[mode]
        if how_many == -1:
            sample_count = len(candidates)
        else:
            sample_count = max(0, min(how_many, len(candidates) - offset))

        # Data and labels will be populated and returned.
        desired_samples = model_settings['desired_samples']
        data = np.zeros((sample_count, model_settings['desired_samples']))
        labels = np.zeros((sample_count, model_settings['label_count']))
        use_background = self.background_data and (mode == 'training')
        pick_deterministically = (mode != 'training')

        # Use the processing graph we created earlier to repeatedly to generate the
        # final output sample data we'll use in training.
        for i in range(offset, offset + sample_count):
            # Pick which audio sample to use.
            if how_many == -1 or pick_deterministically:
                sample_index = i
            else:
                sample_index = np.random.randint(len(candidates))
            sample = candidates[sample_index]

            # If we want silence, mute out the main sample but leave the background.
            if sample['label'] == SILENCE_LABEL:
                foreground_volume = 0
            else:
                foreground_volume = np.random.uniform(0.8, 1.0)

            if foreground_volume > 0:
                #sample_audio, sample_rate = lr.load(sample['file'], sr=model_settings['sample_rate'])
                sample_rate, sample_audio = wf.read(sample['file'])
                sample_audio = sample_audio.astype(np.float32)

                if pitch_shift > 0 and np.random.uniform(0, 1) < pitch_shift_frequency:
                    pitch_shift_amount = np.random.uniform(-pitch_shift, pitch_shift)
                else:
                    pitch_shift_amount = 0
                #print('pitch shift: ', pitch_shift_amount)
                if pitch_shift_amount != 0:
                    sample_audio = lr.effects.pitch_shift(sample_audio, sample_rate, pitch_shift_amount)
                    time_stretch_amount = 1.0
                elif time_stretch > 0 and np.random.uniform(0, 1) < time_stretch_frequency:
                    time_stretch_amount = np.random.uniform(1.0 - time_stretch, 1.0 + time_stretch)
                else:
                    time_stretch_amount = 1.0
                #print('time stretch: ', time_stretch_amount)
                if time_stretch_amount != 1.0:
                    sample_audio = lr.effects.time_stretch(sample_audio, time_stretch_amount)

                actual_samples = sample_audio.shape[0]
                # If we're time shifting, set up the offset for this sample.
                if time_shift > 0:
                    time_shift_amount = np.random.randint(-time_shift, time_shift)
                else:
                    time_shift_amount = 0
                #print('time shift: ', time_shift_amount)
                if time_shift_amount < 0:
                    crop_l = -time_shift_amount
                    pad_l = 0
                else:
                    crop_l = 0
                    pad_l = time_shift_amount
                crop_r = min(actual_samples, desired_samples + crop_l - pad_l)
                pad_r = max(0, desired_samples - (crop_r - crop_l + pad_l))
                #print('cl, cr, pl, pr:', crop_l, crop_r, pad_l, pad_r)

                sample_audio = sample_audio[crop_l:crop_r]
                if pad_l and pad_r:
                    sample_audio = np.r_[
                        np.random.uniform(-0.01, 0.01, pad_l),
                        sample_audio,
                        np.random.uniform(-0.01, 0.01, pad_r)]
                elif pad_l:
                    sample_audio = np.r_[
                        np.random.uniform(-0.01, 0.01, pad_l),
                        sample_audio]
                elif pad_r:
                    sample_audio = np.r_[
                        sample_audio,
                        np.random.uniform(-0.01, 0.01, pad_r)]
                #print(sample_audio.shape)
                sample_audio *= foreground_volume
            else:
                sample_audio = np.zeros(desired_samples, dtype=np.float32)

            # Choose a section of background noise to mix in.
            if use_background and np.random.uniform(0, 1) < background_frequency:
                background_index = np.random.randint(len(self.background_data))
                background_samples = self.background_data[background_index]
                background_offset = np.random.randint(
                    0, len(background_samples) - model_settings['desired_samples'])
                background_clipped = background_samples[
                    background_offset:background_offset + desired_samples]
                #background_reshaped = background_clipped.reshape([desired_samples, 1])
                if sample['label'] == SILENCE_LABEL:
                    background_volume = np.random.uniform(0.01, 0.3)
                else:
                    background_volume = np.random.uniform(0.01, background_volume_range)

                sample_audio = (sample_audio + background_clipped * background_volume).clip(-1.0, 1.0)

            data[i - offset, :] = sample_audio #np.expand_dims(sample_audio, axis=1)
            label_index = self.word_to_index[sample['label']]
            labels[i - offset, label_index] = 1

        return data, labels


class InputDataIterator:
    def __init__(
            self,
            processor,
            how_many, offset, model_settings,
            background_frequency=0.0,
            background_volume_range=0.0,
            pitch_shift_frequency=0.0,
            pitch_shift=0.0,
            time_stretch_frequency=0.0,
            time_stretch=0.0,
            time_shift=0,
            mode='training',
            sess=None):

        self.processor = processor
        self.get_fn = lambda: self.processor.get_data(
            how_many, offset, model_settings,
            background_frequency=background_frequency,
            background_volume_range=background_volume_range,
            pitch_shift_frequency=pitch_shift_frequency,
            pitch_shift=pitch_shift,
            time_stretch_frequency=time_stretch_frequency,
            time_stretch=time_stretch,
            time_shift=time_shift,
            mode=mode,
            sess=sess
        )

    def __next__(self):
        return self.get_fn()

    def __iter__(self):
        return self


class MpIterator:
    def __init__(self, iterator, maxsize=32):
        self.queue = mp.Queue(maxsize=maxsize)
        self.iterator = iterator
        self.is_shutdown = mp.Event()
        self.processes = [mp.Process(target=self._run, args=[i]) for i in range(4)]
        for p in self.processes:
            p.start()

    def _run(self, i):
        try:
            print("Entering run loop", i)
            np.random.seed(RANDOM_SEED + i)
            sys.stdout.flush()
            while not self.is_shutdown.is_set():
                element = next(self.iterator)
                self.queue.put(element)
            print("Exiting run loop")
        except Exception as e:
            self.queue.put(e)
            self.is_shutdown.set()
            self.queue.close()

    def shutdown(self):
        self.is_shutdown.set()
        self.queue.close()

    def __iter__(self):
        return self

    def __next__(self):
        return self.queue.get()
