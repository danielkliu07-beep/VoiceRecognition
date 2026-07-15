import numpy as np
import librosa
import matplotlib.pyplot as plt

from scipy.fftpack import dct

#Loading a trumpet audio file
audio_path = librosa.example('trumpet')
y, sr = librosa.load(audio_path) #y = audio time series, sr = sampling rate of y

plt.figure(figsize = (14, 5))
plt.plot(y)
plt.title('Waveform of the Audio Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

# Computing the full MFCC of an audio file:
# 1. Pre-emphasize the signal
# 2. Framing
# 3. Windowing
# 4. FFT (Fasr Fourier Transform)
# 5. Mel-filterbank
# 6. Logarithm - take log of output from Mel-filterbank
# 7. DCT (Discrete Cosine Transform)

#Pre-emphasis -> Amplifies higher frequencies to balance the spectrum
pre_emphasis = 0.97
y_preemphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1]) #As shown in the graphs, the higher amplitude peaks are lowered to smaller amplitudes

plt.figure(figsize=(14, 5))
plt.plot(y_preemphasized)
plt.title('Pre-emphasized Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

#Framing -> Breaking the signal into small, overlapping frames
frame_size = 0.025 #25 ms
frame_stride = 0.01 #10 ms #frame stride -> how much of a step a frame takes as it goes to the next frame
frame_length = frame_size * sr
frame_step = frame_stride * sr

signal_length = len(y_preemphasized)
frame_length = int(round(frame_length))
frame_step = int(round(frame_step))
num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))

#Pad signal to ensure all of the frames have an equal number of samples
pad_signal_length = num_frames * frame_step + frame_length #Length of frame + all the spaces in between
z = np.zeros((pad_signal_length - signal_length)) #Only the additional stuff is left
pad_signal = np.append(y_preemphasized, z) #Add the padding to the original array

#Slice the indices into frames
indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
frames = pad_signal[indices.astype(np.int32, copy = False)]

plt.figure(figsize = (14, 5))
plt.plot(frames[0])
plt.title('First Frame of the Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

#Windowing -> Soften the edges of each frame, applying a Hamming window
frames *= np.hamming(frame_length)

plt.figure(figsize=(14, 5))
plt.plot(frames[0])
plt.title('First Frame after Windowing')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.show()

#FFT -> Convert each frame from the time domain to the frequency domain
NFFT = 512 #Number of FFTs
mag_frames = np.absolute(np.fft.rfft(frames, NFFT)) #Magnitude of the Fast Fourier Transform
pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2)) #Power Spectrum

plt.figure(figsize = (14, 5))
plt.plot(mag_frames[0])
plt.title('Magnitude Spectrum of the First Frame')
plt.xlabel('Frequency Bin')
plt.ylabel('Amplitude')
plt.show()

#Mel-filterbank -> Apply overlapping triangular filters spaced according to the Mel-scale
nfilt = 40
low_freq_mel = 0
high_freq_mel = 2595 * np.log10(1 + (sr / 2) / 700) #Converts Hz to Mel
mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2) #Equally spaced in Mel scale
hz_points = 700 * (10 ** (mel_points / 2595) - 1) #Convert Mel to Hz
bin = np.floor((NFFT + 1) * hz_points / sr)

fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
for m in range(1, nfilt + 1):
    f_m_minus = int(bin[m - 1]) #left
    f_m = int(bin[m]) #center
    f_m_plus = int(bin[m + 1]) #right

    for k in range(f_m_minus, f_m):
        fbank[m - 1, k] = (k - bin[m - 1] / (bin[m] - bin[m - 1]))
    
    for k in range(f_m, f_m_plus):
        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
filter_banks = np.dot(pow_frames, fbank.T)
filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks) #Numerical stability
filter_banks = 20 * np.log10(filter_banks)

plt.figure(figsize = (14, 5))
plt.imshow(filter_banks.T, cmap = 'hot', aspect = 'auto')
plt.title('Filter Bank Energies')
plt.xlabel('Frame Index')
plt.ylabel('Filter Index')
plt.show()

#DCT -> Apply the Discrete Cosine Transform to the log Mel-sprectrum to obtain the Mel-frequency Cepstral Coefficients

num_ceps = 12
mfcc = dct(filter_banks, type = 2, axis = 1, norm = 'ortho')[:, :num_ceps]

plt.figure(figsize = (14, 5))
plt.imshow(mfcc.T, cmap='hot', aspect='auto')
plt.title('MFCC')
plt.xlabel('Frame Index')
plt.ylabel('Cepstral Coefficient Index')
plt.show()








