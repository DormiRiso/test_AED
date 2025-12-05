from scipy.signal import butter, filtfilt, find_peaks
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def highpass_filter(data, sr, cutoff=250):
    '''Function to apply a high-pass filter to the audio data.
    Args:
        data (np.ndarray): The audio data to be filtered.
        sr (int): The sample rate of the audio data.
        cutoff (float): The cutoff frequency for the high-pass filter in Hz.
    Returns:
        np.ndarray: The filtered audio data.
    '''
    b, a = butter(8, cutoff / (sr / 2), btype='high')
    return filtfilt(b, a, data)


def lowpass_filter(data, sr, cutoff=1000):
    '''Function to apply a low-pass filter to the audio data.
    Args:
        data (np.ndarray): The audio data to be filtered.
        sr (int): The sample rate of the audio data.
        cutoff (float): The cutoff frequency for the low-pass filter in Hz.
    Returns:
        np.ndarray: The filtered audio data.
    '''
    b, a = butter(8, cutoff / (sr / 2), btype='low')
    return filtfilt(b, a, data)


def apply_filters(data, sr, highpass_cutoff=250, lowpass_cutoff=1100, save_path: str = None):
    '''Function to apply both high-pass and low-pass filters to the audio data.
    Args:
        data (np.ndarray): The audio data to be filtered.
        sr (int): The sample rate of the audio data.
        highpass_cutoff (float): The cutoff frequency for the high-pass filter in Hz.
        lowpass_cutoff (float): The cutoff frequency for the low-pass filter in Hz.
        file_path (str): If provided, saves the filtered audio to this path.
    Returns:
        np.ndarray: The filtered audio data.
    '''

    filtered_data = highpass_filter(data, sr, cutoff=highpass_cutoff)
    filtered_data = lowpass_filter(filtered_data, sr, cutoff=lowpass_cutoff)

    if save_path:
        sf.write(save_path, filtered_data, sr)

    return filtered_data
