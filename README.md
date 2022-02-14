#speech-enhancement
Neural Speech Enhancement implemented in PyTorch with CNNs

Based on the paper: *[A Fully Convolutional Neural Network for Speech Enhancement](https://arxiv.org/abs/1609.07132)*

`denoise.py` duplicates the LibriSpeech dataset, and adds different types of noise to the audio that gets put into the model.

In the future, the UrbanSound8K dataset will be used instead of custom noises.