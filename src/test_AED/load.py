from scipy.signal import butter, filtfilt, find_peaks
import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile


def load_audio(file_path, sr=22050, mono=True, save_path=None):
    """Load an audio file.

    Parameters:
    - file_path: str, path to the audio file
    - sr: int, target sampling rate
    - mono: bool, whether to convert the signal to mono
    - save_path: str or None, if provided, save the waveform plot and spectrogram to this path

    Returns:
    - y: np.ndarray, audio time series
    - sr: int, sampling rate of y
    """
    y, sr = librosa.load(file_path, sr=sr, mono=mono)

    if save_path:

        plt.figure(figsize=(12, 6))

        # Plot waveform
        plt.subplot(2, 1, 1)
        librosa.display.waveshow(y, sr=sr)
        plt.title('Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')

        # Plot spectrogram
        plt.subplot(2, 1, 2)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        S_dB = librosa.power_to_db(S, ref=np.max)
        librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel-frequency spectrogram')

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()

    return y, sr
