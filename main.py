import os
from test_AED.load import load_audio
from test_AED.filter import apply_filters
from test_AED.rms import compute_rms
from test_AED.detect import detect_events, EventDetectionConfig

def process_audio(file_path: str):

    # Output structure: output/FILENAME/*
    output_folder = os.path.join("output", os.path.splitext(os.path.basename(file_path))[0])

    highpass_cutoff=250
    lowpass_cutoff=1100
    y, sr = load_audio(file_path)
    y_filtered = apply_filters(y, sr, highpass_cutoff=highpass_cutoff, lowpass_cutoff=lowpass_cutoff)
    rms = compute_rms(y_filtered, frame_size=2048, hop_length=512, save_path=None)
    config = EventDetectionConfig(
        threshold_coefficient=0.8,
        time_window=1.0,
        min_duration=1.5,
        save_path=os.path.join(output_folder, "events_plot.png"),
        transient_window=0.5,
        hop_length=512,
        n_fft=2048,
        sr=sr,
        min_max_freq=(lowpass_cutoff, highpass_cutoff)
    )
    events = detect_events(rms, y_filtered, config)


data_folder = "data/"


for filename in os.listdir(data_folder):
    if filename.endswith(".wav"):
        file_path = os.path.join(data_folder, filename)
        print(f"Processing file: {file_path}")
        process_audio(file_path)
