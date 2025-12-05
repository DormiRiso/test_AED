def compute_binned_frequency_histogram(audio_data, sr, frame_size=2048, hop_length=512,
                                       n_bins=50, freq_range=(0, 5000), save_path: str = None):
    '''Calcola un istogramma binnato della distribuzione di frequenze.

    Args:
        audio_data (np.ndarray): Segnale audio
        sr (int): Sample rate
        frame_size (int): Dimensione della finestra per STFT
        hop_length (int): Hop length per STFT
        n_bins (int): Numero di bin per l'istogramma
        freq_range (tuple): Range di frequenze da considerare (min, max) in Hz
        save_path (str, optional): Path per salvare il plot

    Returns:
        tuple: (hist_values, bin_edges, bin_centers)
    '''
    # Calcola STFT
    stft = np.abs(librosa.stft(audio_data, n_fft=frame_size, hop_length=hop_length))

    # Frequenze corrispondenti a ciascuna riga dello STFT
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_size)

    # Somma lungo il tempo per ogni frequenza
    sum_along_time = np.sum(stft, axis=1)

    # Filtra solo le frequenze nel range specificato
    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    filtered_freqs = frequencies[mask]
    filtered_sums = sum_along_time[mask]

    # Crea i bin per l'istogramma
    bin_edges = np.linspace(freq_range[0], freq_range[1], n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calcola l'istogramma manualmente per avere più controllo
    hist_values = np.zeros(n_bins)
    for freq, sum_val in zip(filtered_freqs, filtered_sums):
        # Trova il bin corretto
        bin_idx = np.searchsorted(bin_edges, freq, side='right') - 1
        if 0 <= bin_idx < n_bins:
            hist_values[bin_idx] += sum_val

    if save_path:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 12))

        # 1. Spettrogramma
        ax1.set_title('Spectrogram')
        img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                                      y_axis='log', x_axis='time',
                                      sr=sr, hop_length=hop_length,
                                      ax=ax1)
        plt.colorbar(img, ax=ax1, format='%+2.0f dB')

        # 2. Istogramma binnato (bar plot)
        ax2.set_title(f'Binned Frequency Histogram ({n_bins} bins, {freq_range[0]}-{freq_range[1]} Hz)')
        width = (freq_range[1] - freq_range[0]) / n_bins * 0.8  # Larghezza delle barre
        ax2.bar(bin_centers, hist_values, width=width,
               edgecolor='black', linewidth=0.5, alpha=0.7)
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Amplitude Sum')
        ax2.grid(True, alpha=0.3, axis='y')

        # 3. Istogramma binnato con scala logaritmica sulle frequenze
        ax3.set_title('Binned Histogram (Log Frequency Scale)')
        ax3.bar(bin_centers, hist_values, width=width,
               edgecolor='black', linewidth=0.5, alpha=0.7)
        ax3.set_xscale('log')
        ax3.set_xlabel('Frequency (Hz) - Log Scale')
        ax3.set_ylabel('Amplitude Sum')
        ax3.grid(True, alpha=0.3, axis='y')

        # Aggiungi statistiche
        stats_text = f"""
        Statistics:
        Total energy: {np.sum(hist_values):.2f}
        Max bin value: {np.max(hist_values):.2f}
        Mean bin value: {np.mean(hist_values):.2f}
        Number of bins: {n_bins}
        Bin width: {bin_edges[1] - bin_edges[0]:.1f} Hz
        """
        fig.text(0.02, 0.02, stats_text, fontsize=9,
                verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    return hist_values, bin_edges, bin_centers

# Versione con bin logaritmici (più utile per audio)
def compute_log_binned_histogram(audio_data, sr, frame_size=2048, hop_length=512,
                                 n_bins=50, freq_range=(20, 8000), save_path: str = None):
    '''Istogramma con bin logaritmici (più naturale per percezione uditiva).

    Args:
        audio_data (np.ndarray): Segnale audio
        sr (int): Sample rate
        frame_size (int): Dimensione della finestra per STFT
        hop_length (int): Hop length per STFT
        n_bins (int): Numero di bin
        freq_range (tuple): Range di frequenze (min, max) in Hz
        save_path (str, optional): Path per salvare il plot

    Returns:
        tuple: (hist_values, bin_edges, bin_centers)
    '''
    # Calcola STFT
    stft = np.abs(librosa.stft(audio_data, n_fft=frame_size, hop_length=hop_length))

    # Frequenze corrispondenti
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=frame_size)

    # Somma lungo il tempo
    sum_along_time = np.sum(stft, axis=1)

    # Filtra range di frequenze
    mask = (frequencies >= freq_range[0]) & (frequencies <= freq_range[1])
    filtered_freqs = frequencies[mask]
    filtered_sums = sum_along_time[mask]

    # Crea bin logaritmici
    log_min = np.log10(freq_range[0])
    log_max = np.log10(freq_range[1])
    log_bin_edges = np.logspace(log_min, log_max, n_bins + 1)
    bin_centers = np.sqrt(log_bin_edges[:-1] * log_bin_edges[1:])  # Media geometrica

    # Calcola istogramma
    hist_values = np.zeros(n_bins)
    for freq, sum_val in zip(filtered_freqs, filtered_sums):
        bin_idx = np.searchsorted(log_bin_edges, freq, side='right') - 1
        if 0 <= bin_idx < n_bins:
            hist_values[bin_idx] += sum_val

    if save_path:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Spettrogramma
        ax1 = axes[0, 0]
        ax1.set_title('Spectrogram')
        img = librosa.display.specshow(librosa.amplitude_to_db(stft, ref=np.max),
                                      y_axis='log', x_axis='time',
                                      sr=sr, hop_length=hop_length,
                                      ax=ax1)
        plt.colorbar(img, ax=ax1, format='%+2.0f dB')

        # Istogramma lineare
        ax2 = axes[0, 1]
        ax2.set_title('Linear Binned Histogram')
        width = (log_bin_edges[1:] - log_bin_edges[:-1]) * 0.8
        ax2.bar(bin_centers, hist_values, width=width,
               edgecolor='black', linewidth=0.5, alpha=0.7, color='skyblue')
        ax2.set_xscale('log')
        ax2.set_xlabel('Frequency (Hz)')
        ax2.set_ylabel('Energy Sum')
        ax2.grid(True, alpha=0.3, axis='y')

        # Istogramma a gradini (più preciso)
        ax3 = axes[1, 0]
        ax3.set_title('Step Histogram (More Accurate)')
        ax3.step(log_bin_edges, np.append(hist_values, hist_values[-1]),
                where='post', linewidth=2, color='darkred')
        ax3.fill_between(log_bin_edges, np.append(hist_values, hist_values[-1]),
                         step='post', alpha=0.3, color='darkred')
        ax3.set_xscale('log')
        ax3.set_xlabel('Frequency (Hz)')
        ax3.set_ylabel('Energy Sum')
        ax3.grid(True, alpha=0.3)

        # Distribuzione cumulativa
        ax4 = axes[1, 1]
        ax4.set_title('Cumulative Distribution')
        cumulative = np.cumsum(hist_values)
        cumulative_norm = cumulative / cumulative[-1] if cumulative[-1] > 0 else cumulative
        ax4.plot(bin_centers, cumulative_norm, 'g-', linewidth=2, marker='o', markersize=4)
        ax4.set_xscale('log')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Cumulative Fraction')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(0.5, color='r', linestyle='--', alpha=0.5, label='50%')
        ax4.axhline(0.9, color='orange', linestyle='--', alpha=0.5, label='90%')
        ax4.legend()

        # Aggiungi linee per le frequenze importanti
        if np.sum(hist_values) > 0:
            # Trova bin con massima energia
            max_bin_idx = np.argmax(hist_values)
            max_freq = bin_centers[max_bin_idx]

            # Calcola mediana
            median_idx = np.where(cumulative_norm >= 0.5)[0][0]
            median_freq = bin_centers[median_idx]

            ax2.axvline(max_freq, color='red', linestyle='--', alpha=0.7,
                       label=f'Max: {max_freq:.0f} Hz')
            ax2.axvline(median_freq, color='green', linestyle='--', alpha=0.7,
                       label=f'Median: {median_freq:.0f} Hz')
            ax2.legend()

            ax3.axvline(max_freq, color='red', linestyle='--', alpha=0.7)
            ax3.axvline(median_freq, color='green', linestyle='--', alpha=0.7)

        plt.suptitle(f'Frequency Distribution Analysis - {len(audio_data)/sr:.2f}s clip',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.show()
        plt.close()

    return hist_values, log_bin_edges, bin_centers

# Funzione per analizzare tutti i clip
def analyze_all_clips_binned(audio_clips, sr, n_bins=40, freq_range=(50, 4000)):
    '''Analizza tutti i clip con istogrammi binnati.'''

    if len(audio_clips) == 0:
        print("Nessun clip da analizzare!")
        return []

    all_histograms = []

    for i, clip in enumerate(audio_clips):
        if len(clip) > 0:
            print(f"\n{'='*60}")
            print(f"ANALISI EVENTO {i} - {len(clip)/sr:.2f} secondi")
            print(f"{'='*60}")

            # Usa la versione con bin logaritmici (più utile per audio)
            hist_values, bin_edges, bin_centers = compute_log_binned_histogram(
                clip, sr,
                frame_size=2048, hop_length=512,
                n_bins=n_bins, freq_range=freq_range,
                save_path=f"event_{i}_binned_histogram.png"
            )

            # Salva risultati
            all_histograms.append({
                'event_id': i,
                'hist_values': hist_values,
                'bin_edges': bin_edges,
                'bin_centers': bin_centers,
                'clip_duration': len(clip)/sr,
                'total_energy': np.sum(hist_values)
            })

            # Stampa statistiche
            print(f"Durata clip: {len(clip)/sr:.2f}s")
            print(f"Energia totale: {np.sum(hist_values):.2f}")
            print(f"Valore massimo bin: {np.max(hist_values):.2f}")
            print(f"Valore medio bin: {np.mean(hist_values):.2f}")

            # Trova frequenza dominante (bin con massima energia)
            max_bin_idx = np.argmax(hist_values)
            dominant_freq = bin_centers[max_bin_idx]
            print(f"Frequenza dominante: {dominant_freq:.1f} Hz")

            # Calcola larghezza di banda (dove si concentra l'80% dell'energia)
            cumulative = np.cumsum(hist_values)
            if cumulative[-1] > 0:
                cumulative_norm = cumulative / cumulative[-1]
                low_idx = np.where(cumulative_norm >= 0.1)[0][0]
                high_idx = np.where(cumulative_norm >= 0.9)[0][0]
                bandwidth_low = bin_centers[low_idx]
                bandwidth_high = bin_centers[high_idx]
                print(f"Banda 10-90%: {bandwidth_low:.1f} - {bandwidth_high:.1f} Hz")

            # Riproduci audio
            display(Audio(clip, rate=sr))
        else:
            print(f"Evento {i} - Clip vuoto, saltato")

    return all_histograms

# Esegui l'analisi
all_histograms = analyze_all_clips_binned(
    audio_clips, sr,
    n_bins=40,  # Numero di bin
    freq_range=(50, 4000)  # Range di frequenze di interesse per ululati
)

# Se vuoi confrontare tutti gli istogrammi insieme
def plot_comparison_histograms(all_histograms, save_path="comparison_histograms.png"):
    '''Confronta tutti gli istogrammi nello stesso plot.'''
    if not all_histograms:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Normalizza e plotta tutti gli istogrammi
    for hist_data in all_histograms:
        hist_values = hist_data['hist_values']
        bin_centers = hist_data['bin_centers']

        # Normalizza per confronto
        if np.sum(hist_values) > 0:
            hist_norm = hist_values / np.max(hist_values)
        else:
            hist_norm = hist_values

        ax1.plot(bin_centers, hist_norm, alpha=0.6, linewidth=1.5,
                label=f"Event {hist_data['event_id']}")

        # Plot cumulativo
        cumulative = np.cumsum(hist_values)
        if cumulative[-1] > 0:
            cumulative_norm = cumulative / cumulative[-1]
            ax2.plot(bin_centers, cumulative_norm, alpha=0.6, linewidth=1.5)

    ax1.set_title('Normalized Frequency Histograms Comparison')
    ax1.set_xscale('log')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Normalized Energy')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title('Cumulative Distributions Comparison')
    ax2.set_xscale('log')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Cumulative Fraction')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(0.5, color='r', linestyle='--', alpha=0.5)
    ax2.axhline(0.9, color='orange', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    plt.close()

# Esegui il confronto se ci sono dati
if all_histograms:
    plot_comparison_histograms(all_histograms)


# Analizza un singolo clip
hist_values, bin_edges, bin_centers = compute_binned_frequency_histogram(
    clip, sr, n_bins=50, freq_range=(50, 4000), save_path="histogram.png"
)

# Oppure analizza tutti i clip
all_results = analyze_all_clips_binned(audio_clips, sr)