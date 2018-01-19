# Kaggle Speech Recognition Challenge

This was my first crack at the Kaggle Tensorflow Speech Recognition Challenge (
https://www.kaggle.com/c/tensorflow-speech-recognition-challenge). 

It is a Tensorflow based solution that combines the Tensorflow speech commands (as the base, https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/speech_commands) with the Tensorflow models audioset vggish model (https://github.com/tensorflow/models/tree/master/research/audioset) and some custom models that I built.

The models and training/validation code were updated to the TF Slim style. 

I was never able to get great results with this code base. I suspect most of it was due to the default data handling and augmentation that I was relying on. There wasn't a whole lot of difference from model to model. I quickly surpassed all results I achieved with these models in a few days of hacking around in PyTorch in the dying days of the challenge. 

One noteworthy thing I did experiment with here before moving to PyTorch is the 1D convolution models. The performance of several custom 1D models was at or just below the performance of the 2D MFCC spectrogram based models. 'conv1d_basic3' being the most promising of those.