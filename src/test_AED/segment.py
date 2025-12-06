'''Module for segmenting audio based on event timestamps.'''
import os
import numpy as np
import librosa
import soundfile as sf


def segment_audio(y_filtered, sr, events, save_folder: str = None, normalize_gain: bool = True, target_level: float = -3.0):
    '''Function to segment audio based on event timestamps.
    Args:
        y_filtered (np.ndarray): Filtered audio signal.
        sr (int): Sampling rate of the audio signal.
        events (list of tuples): List of (start_time, end_time) in seconds.
        save_folder (str, optional): Folder to save segmented audio clips. Defaults to None.
        normalize_gain (bool, optional): Whether to normalize the gain of audio clips. Defaults to True.
        target_level (float, optional): Target loudness level in dBFS for normalization. Defaults to -3.0.
    Returns:
        list: List of segmented audio clips.
    '''

    audio_clips = []

    valid_clips = 0
    skipped_clips = 0

    for i, (s, e) in enumerate(events):

        start_sample = librosa.time_to_samples(s, sr=sr)
        end_sample = librosa.time_to_samples(e, sr=sr)

        if start_sample >= len(y_filtered):
            print(f"SKIP: start_sample ({start_sample}) >= lunghezza y_filtered ({len(y_filtered)})")
            skipped_clips += 1
            continue

        end_sample = min(end_sample, len(y_filtered))

        if start_sample >= end_sample:
            print(f"SKIP: start_sample ({start_sample}) >= end_sample ({end_sample})")
            skipped_clips += 1
            continue

        clip = y_filtered[start_sample:end_sample]

        if len(clip) == 0:
            skipped_clips += 1
            continue

        # Normalizzazione del gain
        if normalize_gain and len(clip) > 0:
            clip = normalize_audio_gain(clip, target_level=target_level)

        audio_clips.append(clip)
        valid_clips += 1

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            sf.write(f"{save_folder}/event_{i}.wav", clip, sr)

    print(f"Segmentazione completata: {valid_clips} clip validi, {skipped_clips} clip saltati")
    return audio_clips


def normalize_audio_gain(audio: np.ndarray, target_level: float = -3.0, eps: float = 1e-10) -> np.ndarray:
    '''Normalize audio gain to a target level.
    
    Args:
        audio (np.ndarray): Input audio signal.
        target_level (float): Target loudness level in dBFS. Defaults to -3.0.
        eps (float): Small epsilon to avoid division by zero. Defaults to 1e-10.
    
    Returns:
        np.ndarray: Normalized audio signal.
    '''
    # Calcola il livello RMS corrente in dBFS
    rms = np.sqrt(np.mean(np.square(audio.astype(float))))
    current_level_db = 20 * np.log10(rms + eps)
    
    # Calcola il fattore di guadagno necessario
    gain_db = target_level - current_level_db
    gain_linear = 10 ** (gain_db / 20)
    
    # Applica il guadagno
    normalized_audio = audio * gain_linear
    
    # Limita il segnale a [-1, 1] per prevenire clipping
    # (rimuovi se vuoi preservare i valori originali)
    normalized_audio = np.clip(normalized_audio, -1.0, 1.0)
    
    return normalized_audio


def normalize_audio_peak(audio: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    '''Normalize audio to a target peak level.
    
    Args:
        audio (np.ndarray): Input audio signal.
        target_peak (float): Target peak amplitude (0 to 1). Defaults to 0.95.
    
    Returns:
        np.ndarray: Peak-normalized audio signal.
    '''
    # Trova il picco massimo
    peak = np.max(np.abs(audio))
    
    if peak > 0:
        # Calcola il fattore di normalizzazione
        gain = target_peak / peak
        
        # Applica il guadagno
        normalized_audio = audio * gain
    
        return normalized_audio
    
    return audio
