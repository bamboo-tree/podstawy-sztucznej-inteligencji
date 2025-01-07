from scipy.io import wavfile
from scipy.fft import fft
import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt


# wczytanie listy plików
files = os.listdir('voices')
fs = 16000
seconds = 3
# uzupełnienie pustej listy
x_raw = np.zeros((len(files), fs*seconds))
# wypełnienie danymi
for i, file in enumerate(files):
    x_raw[i,:] = wavfile.read(f'./voices/{file}')[1]
y = pd.read_csv('Genders_voices.csv', sep=';')

x_fft = np.abs(fft(x_raw, axis=-1)) / x_raw.shape[1]
# utworzenie wykresu
fig, ax = plt.subplots(2, 1)
ax[0].plot(np.arange(x_raw.shape[1]), x_raw[0,:])
ax[1].scatter(np.arange(x_raw.shape[1]), x_fft[0,:], s=0.5)
fig.tight_layout()

# zmniejszenie rozdzielczości widma
mean_num = 3
x_fft = np.reshape(x_fft, (x_fft.shape[0], x_fft.shape[1]//mean_num, mean_num))
x_fft = x_fft.mean(axis=-1)
low_cut = 50*seconds
high_cut = 280*seconds
x_fft_cut = x_fft[:, low_cut:high_cut]
x_fft_cut = x_fft_cut / np.expand_dims(x_fft_cut.max(axis=1), axis=-1)
plt.figure(figsize=(10, 6))
plt.plot(np.arange(x_fft_cut.shape[1]), x_fft_cut[0,:])
plt.xlabel('Czestotliwosc [Hz]')
plt.ylabel('Amplituda')