'''Module to compute and visualize the Root Mean Square (RMS) energy of an audio signal.'''
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def compute_rms(audio_data, frame_size=2048, hop_length=512, save_path: str = None):
    '''Function to compute the Root Mean Square (RMS) energy of an audio signal.
    Args:
        audio_data (np.ndarray): The audio data.
        frame_size (int): The size of each frame for RMS calculation.
        hop_length (int): The hop length between frames.
        save_path (str): If provided, saves the RMS plot to this path.
    Returns:
        np.ndarray: The RMS energy values.
    '''

    rms = librosa.feature.rms(y=audio_data, frame_length=frame_size, hop_length=hop_length)[0]
    rms_mean = np.mean(rms)

    if save_path:
        plt.figure(figsize=(10, 4))
        timestamp = librosa.frames_to_time(range(len(rms)), hop_length=hop_length)
        plt.plot(timestamp, rms, label='RMS Energy')
        plt.hlines(rms_mean, 0, timestamp[-1], color='r', linestyle='--', label='RMS Mean')
        plt.xlabel('Time (s)')
        plt.ylabel('RMS Energy')
        plt.title('RMS Energy Over Time')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.close()

    return rms
