'''Module for segmenting audio based on event timestamps.'''
import os
import librosa
import soundfile as sf


def segment_audio(y_filtered, sr, events, save_folder: str = None):
    '''Function to segment audio based on event timestamps.
    Args:
        y_filtered (np.ndarray): Filtered audio signal.
        sr (int): Sampling rate of the audio signal.
        events (list of tuples): List of (start_time, end_time) in seconds.
        save_folder (str, optional): Folder to save segmented audio clips. Defaults to None.
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

        audio_clips.append(clip)
        valid_clips += 1

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            sf.write(f"{save_folder}/event_{i}.wav", clip, sr)
