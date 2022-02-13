# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 13:08:17 2021

@author: Christian Konstantinov
"""
#%% Imports

import random
import numpy as np
import librosa as lb
import torch
from torch import nn
from torchaudio.datasets import LIBRISPEECH
import torch.optim as optim

#%% Data loading

DATA_PATH = '../data'

train_data = LIBRISPEECH(DATA_PATH, url='train-clean-100')
test_data = LIBRISPEECH(DATA_PATH, url='test-clean')
dev_data = LIBRISPEECH(DATA_PATH, url='test-clean')

fs = 8000

#%% Noise loading

NOISE_KITCHEN_PATH = f'{DATA_PATH}/noise/kitchen.wav'
NOISE_DOGS_PATH = f'{DATA_PATH}/noise/dogs.wav'
NOISE_MUSIC_PATH = f'{DATA_PATH}/noise/music.wav'

noise_kitchen, _ = lb.load(NOISE_KITCHEN_PATH, sr=fs)
noise_dogs, _ = lb.load(NOISE_DOGS_PATH, sr=fs)
noise_music, _ = lb.load(NOISE_MUSIC_PATH, sr=fs)

#%% Dataset creation functions

def get_rms(sig):
    return np.sqrt(np.mean(np.absolute(sig)**2))

def get_white_noise(size):
    return np.random.normal(0, 1, size).astype(np.float32)

def get_noise(size, noise):
    if type(noise) == str and noise == 'white':
        return get_white_noise(size)
    noise = np.array([0])
    i = np.random.choice(noise.size, 1)[0]
    noise = noise[i:i+size]
    pad_len = size - (noise.size - i)
    if pad_len > 0:
        pad = np.zeros((pad_len,))
        np.append(noise, pad)
    return noise.astype(np.float32)

def add_noise(input_sig, source='white'):
    output_sig = input_sig
    noise = get_noise(input_sig.size, source)
    noise_rms = get_rms(noise)
    if noise_rms:
        noise *= get_rms(input_sig) / noise_rms
    return (output_sig + noise) / 2

def get_spectrogram_tensor(input_sig):
    n_frames = int(fs / 1000)
    lmfs = lb.feature.melspectrogram(y=input_sig, sr=fs,
                                                    n_fft=256, window=torch.hamming_window,
                                                    hop_length=64)
    total_frames = lmfs.shape[1]
    pad_len = n_frames - (total_frames % n_frames)
    if pad_len != n_frames:
        pad = np.zeros((lmfs.shape[0], pad_len))
        lmfs = np.append(lmfs, pad, axis=1)
    input_vector = np.split(lmfs, lmfs.shape[1] // n_frames, axis=1)
    #input_vector = [lmfs[:, i : i+n_frames] for i in range(0, lmfs.shape[1]-n_frames)]
    return torch.tensor(input_vector)

def create_dataset(librispeech):
    dataset = []
    sources = {0: 'white', 1: noise_kitchen, 2: noise_dogs, 3: noise_music}

    for n, (waveform, _, _, _, _, _) in enumerate(librispeech):
        waveform = lb.resample(waveform.squeeze(0).numpy(), 16000, fs) # LIBRISPEECH is sampled at 16 kHz
        feature = get_spectrogram_tensor(add_noise(waveform, source=random.choice(sources))).unsqueeze(1)
        target = get_spectrogram_tensor(waveform).unsqueeze(1)
        dataset.append((feature, target))
        if n == 100:
            break
    return dataset

def save_dataset(dataset, fname):
    torch.save(dataset, f'{DATA_PATH}/{fname}.pt')

#%% Pleb Check

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#%% Tensor creation

train_data = (create_dataset(train_data))
test_data = (create_dataset(test_data))
dev_data = (create_dataset(dev_data))

#%% Tensor saving

save_dataset(train_data, 'train')
save_dataset(test_data, 'test')
save_dataset(dev_data, 'dev')

#%% Tensor loading

train_data = torch.load(f'{DATA_PATH}/train.pt')
test_data = torch.load(f'{DATA_PATH}/test.pt')
dev_data = torch.load(f'{DATA_PATH}/dev.pt')

#%% CNNs

class CR_CNN(nn.Module):
    def __init__(self):
        super(CR_CNN, self).__init__()
        self.relu = nn.ReLU(inplace=True).to(device)
        self.bn = nn.BatchNorm2d(1).to(device)
        self.conv1 = nn.Conv2d(1, 1, (9, 8), padding=4).to(device)
        self.conv2 = nn.Conv2d(1, 1, (5, 1), padding=4).to(device)
        self.conv3 = nn.Conv2d(1, 1, (9, 1), padding=4).to(device)
        self.conv4 = nn.Conv2d(1, 1, (128, 1), padding=4).to(device)
        self.block = nn.Sequential(self.conv2, self.relu, self.bn,
                                   self.conv3, self.relu, self.bn,
                                   self.conv3, self.relu, self.bn
                                  ).to(device)
        self.chonk = nn.Sequential(self.block, self.block, self.block, self.block).to(device)

    def forward(self, x):
        x = self.bn(self.relu(self.conv1(x)))
        x = self.chonk(x)
        x = self.bn(self.relu(self.conv2(x)))
        x = self.bn(self.relu(self.conv3(x)))
        x = self.bn(self.relu(self.conv4(x)))
        return x

# evaluation metric
def SDR(f, y):
    """
    Return the signal to distortion ratio.

    Parameters
    ----------
    f : numpy.ndarray
        The denoised spectrum.
    y : numpy.ndarray
        The clean spectrum.

    Returns
    -------
    numpy.float32
        The signal to distortion ratio.

    """
    return 10*np.log10(np.sqrt(y**2)**2/(np.sqrt((f-y)**2)**2))

#%% Hyperparameters

model = CR_CNN()
loss_function = nn.MSELoss()
optimizer = optim.Adam(params=model.parameters(), lr=0.0015)
patience = 4
epochs = 10
hidden_size = 10
layers = 3
dev_losses = []
no_improve = 0

#%% Training

def train(model, training_data, optimizer, loss_function):
    for epoch in range(epochs):
        print(f'Starting epoch {epoch}...')
        for feature, target in training_data:
            model.zero_grad()
            output = model(torch.FloatTensor(feature.float()).to(device))
            print(output.shape)
            print(torch.FloatTensor(target.float()).to(device).shape)
            loss = loss_function(output, torch.FloatTensor(target.float()).to(device))
            loss.backward()
            optimizer.step()

        # validate
        with torch.no_grad():
            for feature, target in dev_data:
                output = model(feature.to(device))
                loss = loss_function(output, target.to(device))
                if not dev_losses or loss >= max(dev_losses):
                    no_improve = 0
                    torch.save('../data/model.pt')
                else:
                    no_improve += 1
                dev_losses.append(loss)
                if no_improve >= patience:
                    print(f'Patience exceeded! ({patience} epochs with no accuracy improvement). Stopping early.')
                    break

train(model, train_data, optimizer, loss_function)

#%% Testing

def test(model, testing_data):
    evaluations = []
    with torch.no_grad():
        for feature, target in testing_data:
            output = model(feature.to(device))
            evaluations.append(SDR(output, target))
    return evaluations

#%% Other

def spectrogram_to_audio(spectrogram, fs):
    pass