'''Module for loading audio files for AED tests.'''
import librosa

def load_audio(file_path, sr=22050, mono=True):
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

    return y, sr
