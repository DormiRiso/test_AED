'''Module to detect events in an audio signal based on RMS energy.'''
import os
from dataclasses import dataclass
from typing import List, Tuple
from scipy.signal import find_peaks
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


# Dataclass for configuration
@dataclass
class EventDetectionConfig:
    '''Configuration for event detection.
    Attributes:
        threshold_coefficient (float): Coefficient to multiply with mean RMS for threshold.
        time_window (float): Maximum gap between peaks to consider them part of the same event.
        min_duration (float): Minimum duration for an event to be valid.
        save_path (str | None): Path to save the event plot. If None, no plot is saved.
        transient_window (float): Time to extend before and after each event.
        hop_length (int): Hop length used in RMS computation.
        n_fft (int): FFT size used in Mel spectrogram computation.
        sr (int): Sample rate of the audio signal.
        min_max_freq (tuple): Minimum and maximum frequency range for analysis.
    '''

    threshold_coefficient: float = 0.8
    time_window: float = 1.0
    min_duration: float = 1.0
    save_path: str | None = None
    transient_window: float = 0.5
    hop_length: int = 512
    n_fft: int = 2048
    sr: int = 22050
    min_max_freq: tuple = (200, 1200)


def _compute_peaks(rms: np.ndarray, cfg: EventDetectionConfig):
    '''Function to compute peaks in the RMS signal.'''

    threshold = cfg.threshold_coefficient * float(np.mean(rms))
    peaks, _ = find_peaks(rms, height=threshold)
    times = librosa.frames_to_time(peaks, sr=cfg.sr, hop_length=cfg.hop_length)
    return times, threshold


def _merge_peaks_into_events(peak_times: np.ndarray, cfg: EventDetectionConfig):
    '''Function to merge peaks into events based on time window and minimum duration.'''

    if len(peak_times) == 0:
        return []

    events = []
    start = peak_times[0]
    end = peak_times[0]

    for t in peak_times[1:]:
        if t - end < cfg.time_window:
            end = t
        else:
            if (end - start) >= cfg.min_duration:
                events.append((start - cfg.transient_window, end + cfg.transient_window))
            start = t
            end = t

    # Add last event
    if (end - start) >= cfg.min_duration:
        events.append((start - cfg.transient_window, end + cfg.transient_window))

    return events


def _plot_events(y: np.ndarray, rms: np.ndarray,
                 events: List[Tuple[float, float]], threshold: float,
                 cfg: EventDetectionConfig):
    '''Plot spectrogram + RMS timeline with detected events (two subplots).'''

    if not cfg.save_path:
        return

    # Calcola STFT
    D = librosa.stft(y, n_fft=cfg.n_fft, hop_length=cfg.hop_length)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    # Timestamp per STFT (centri dei frames)
    times_stft = librosa.frames_to_time(
        np.arange(S_db.shape[1]), 
        sr=cfg.sr, 
        hop_length=cfg.hop_length
    )

    # Timestamp per RMS (USANDO LO STESSO CALCOLO)
    times_rms = librosa.frames_to_time(
        np.arange(len(rms)), 
        sr=cfg.sr, 
        hop_length=cfg.hop_length
    )

    # Calcola frequenze (in Hz)
    freqs = librosa.fft_frequencies(sr=cfg.sr, n_fft=cfg.n_fft)

    # Usa i limiti di frequenza dalla configurazione
    min_freq, max_freq = cfg.min_max_freq

    # Verifica che i limiti siano validi
    min_freq = max(min_freq, freqs[0])  # Non può essere minore della frequenza minima disponibile
    max_freq = min(max_freq, freqs[-1])  # Non può essere maggiore della frequenza massima disponibile

    if min_freq >= max_freq:
        # Se i limiti non sono validi, usa tutto lo spettro
        min_freq = freqs[0]
        max_freq = freqs[-1]

    # Trova gli indici delle frequenze nel range desiderato
    freq_indices = np.where((freqs >= min_freq) & (freqs <= max_freq))[0]

    if len(freq_indices) == 0:
        # Fallback a tutto lo spettro se il range non ha dati
        freq_indices = np.arange(len(freqs))
        min_freq = freqs[0]
        max_freq = freqs[-1]

    # Filtra lo spettrogramma per il range di frequenze
    S_db_filtered = S_db[freq_indices, :]
    freqs_filtered = freqs[freq_indices]

    # Tempo massimo
    if len(times_rms) > 0 and len(times_stft) > 0:
        max_time = min(times_stft[-1], times_rms[-1])
    else:
        max_time = librosa.get_duration(y=y, sr=cfg.sr)

    # Crea la figura con sfondo bianco
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        facecolor='white'
    )

    # Plot STFT con colori e senza colorbar
    extent = [times_stft[0], times_stft[-1], min_freq, max_freq]
    im = ax1.imshow(
        S_db_filtered,
        aspect='auto',
        origin='lower',
        extent=extent,
        cmap='viridis',
        interpolation='bilinear',
        vmin=-80,
        vmax=0
    )

    ax1.set_ylabel("Frequency (Hz)")
    ax1.set_title(f"Spectrogram ({int(min_freq)}-{int(max_freq)} Hz)")

    # Imposta limite verticale preciso dal range di configurazione
    ax1.set_ylim([min_freq, max_freq])

    # Plot RMS
    ax2.plot(
        times_rms,
        rms,
        label="RMS Energy",
        linewidth=2.0,
        color='darkblue'
    )

    # Threshold line
    ax2.hlines(
        threshold, 0, max_time,
        colors='red', linestyles='--',
        linewidth=2.0,
        label=f"Threshold: {threshold:.3f}"
    )

    # Event intervals - SOLO NEL PLOT RMS (rimosso dallo spettrogramma)
    for i, (s, e) in enumerate(events):
        s_clipped = max(0, min(s, max_time))
        e_clipped = max(0, min(e, max_time))
        if e_clipped > s_clipped:
            # Colore alternato per distinguere eventi consecutivi
            color = 'lime' if i % 2 == 0 else 'cyan'

            # Area evidenziata per l'evento SOLO nel plot RMS
            ax2.axvspan(s_clipped, e_clipped, color=color, alpha=0.3)

            # Linee verticali SOLO nel plot RMS (rimosse dallo spettrogramma)
            ax2.axvline(x=s_clipped, color=color, linestyle='-', alpha=0.7, linewidth=2.0)
            ax2.axvline(x=e_clipped, color=color, linestyle='-', alpha=0.7, linewidth=2.0)

            # Etichetta numerica per l'evento SOLO nel plot RMS
            mid_time = (s_clipped + e_clipped) / 2
            # Posiziona l'etichetta sopra la linea RMS
            y_pos = ax2.get_ylim()[1] * 0.95
            ax2.text(mid_time, y_pos, f"{i+1}", 
                    color='black', fontsize=10, fontweight='bold',
                    ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.7))

    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("RMS Energy")
    ax2.set_title(f"Detected Events: {len(events)}")
    ax2.legend(loc="upper right")

    # Imposta limiti espliciti sugli assi x
    ax1.set_xlim([0, max_time])
    ax2.set_xlim([0, max_time])

    # Aggiungi griglia (con colori adatti per sfondo bianco)
    ax1.grid(True, alpha=0.2, linestyle='--', color='gray')
    ax2.grid(True, alpha=0.3, color='gray')

    # Aggiungi linee di riferimento temporali (ogni secondo)
    for sec in range(0, int(max_time) + 1):
        if sec <= max_time:
            alpha = 0.2 if sec % 5 == 0 else 0.1  # Più visibili ogni 5 secondi
            linewidth = 1.0 if sec % 5 == 0 else 0.5
            ax1.axvline(x=sec, color='gray', alpha=alpha, linestyle='-', linewidth=linewidth)
            ax2.axvline(x=sec, color='gray', alpha=alpha, linestyle='-', linewidth=linewidth)

    # Aggiungi linee di riferimento per frequenze importanti nel range
    # Crea una lista di frequenze di riferimento basata sul range
    base_freqs = [100, 200, 500, 1000, 2000, 5000, 10000]
    important_freqs = [f for f in base_freqs if min_freq <= f <= max_freq]

    # Aggiungi i limiti del range se non sono già nella lista
    if min_freq not in important_freqs:
        important_freqs.insert(0, min_freq)
    if max_freq not in important_freqs:
        important_freqs.append(max_freq)

    # Ordina le frequenze
    important_freqs.sort()

    for freq in important_freqs:
        ax1.axhline(y=freq, color='gray', alpha=0.15, linestyle='-', linewidth=0.5)
        # Etichetta per le frequenze importanti a sinistra
        ax1.text(-max_time*0.008, freq, f"{int(freq)}", 
                color='black', fontsize=8, alpha=0.7,
                ha='right', va='center')

    # RIMOSSA: la scritta a lato destro dello spettrogramma
    # (la riga con ax1.text che mostrava "Energy\n(Low → High)")

    # Ottimizza il layout
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.08)  # Riduce lo spazio verticale tra i subplot

    # Salva con sfondo bianco
    os.makedirs(os.path.dirname(cfg.save_path), exist_ok=True)
    plt.savefig(cfg.save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()


# Main function to detect events
def detect_events(rms: np.ndarray, y: np.ndarray,
                  cfg: EventDetectionConfig = EventDetectionConfig()
                  ) -> List[Tuple[float, float]]:
    '''
    Detect events in an audio signal based on RMS energy.
    Args:
        rms (np.ndarray): The RMS energy values.
        y (np.ndarray): The original audio signal.
        cfg (EventDetectionConfig): Configuration for event detection.
    Returns:
        List[Tuple[float, float]]: List of detected events as (start_time, end_time) tuples.
    '''

    peak_times, threshold = _compute_peaks(rms, cfg)
    events = _merge_peaks_into_events(peak_times, cfg)

    # Ensure event times are within valid range
    events = [(max(0, s), min(e, librosa.get_duration(y=y, sr=cfg.sr))) for s, e in events]

    print("Events detected:", len(events))

    _plot_events(y, rms, events, threshold, cfg)

    return events
