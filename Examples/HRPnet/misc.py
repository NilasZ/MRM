import os
import glob
import pickle
from numpy.fft import fftshift,ifft
from scipy.signal import stft, windows
import numpy as np

def load_pkl(file_path):
    with open(file_path, 'rb') as file:
        return pickle.load(file)
    
def rcs(echo):
    return 4*np.pi*np.abs(echo)**2

def awgn(signal, snr):
    # Calculate signal power and convert SNR to linear scale
    signal_power = np.mean(np.abs(signal)**2)
    snr_linear = 10**(snr / 10)
    
    # Calculate noise power and generate complex noise
    noise_power = signal_power / snr_linear
    noise_real = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
    noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
    noise = noise_real + 1j * noise_imag
    
    # Add noise to the signal
    signal_with_noise = signal + noise
    
    return signal_with_noise, noise_power

def awgnfp(signal, noise_power):
    noise_real = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
    noise_imag = np.random.normal(0, np.sqrt(noise_power / 2), signal.shape)
    noise = noise_real + 1j * noise_imag
    
    signal_with_noise = signal + noise
    signal_power = np.mean(np.abs(signal)**2)
    SNR = 10*np.log10(signal_power/ noise_power)

    return signal_with_noise, SNR


def normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    normalized_matrix = (matrix - min_val) / (max_val - min_val)
    return normalized_matrix   

def STFT(st,nfft):
    winlen = 64
    _, _, Zxx = stft(st, fs=1024, window = windows.hamming(winlen), nperseg=winlen, nfft=nfft ,noverlap=winlen-1, boundary='zeros', return_onesided=False)
    return np.fft.fftshift(Zxx,axes=0) 

def pad_hrrp(matrix, target_length):
    if target_length is None:
        return matrix
    else:
        rows, cols = matrix.shape 
        padded_matrix = np.zeros((target_length, cols),dtype=np.complex128)
        padded_matrix[:rows, :] = matrix
        return padded_matrix

def image_hrrp(hrrp, pad_size):
    hrrp = pad_hrrp(hrrp,pad_size)
    hrrp = fftshift(ifft(hrrp,axis = 0),axes=0)
    hrrp = np.log10(np.abs(hrrp))  
    hrrp = normalize(hrrp)
    return hrrp