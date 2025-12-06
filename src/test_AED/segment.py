

def segment_audio(y, y_filtered, sr, events, save_folder: str = None):
    """Divido i segmenti trovati in tanti pezzettini di audio"""

    # Prima, verifica le lunghezze
    print(f"Lunghezza y (originale): {len(y)} campioni")
    print(f"Lunghezza y_filtered: {len(y_filtered)} campioni")
    print(f"Sample rate: {sr} Hz")
    print(f"Durata y: {len(y)/sr:.2f} secondi")
    print(f"Durata y_filtered: {len(y_filtered)/sr:.2f} secondi")

    # Inizializza la lista per i clip audio
    audio_clips = []

    # Contatori per debug
    valid_clips = 0
    skipped_clips = 0

    for i, (s, e) in enumerate(events):
        # Converti tempi in campioni
        start_sample = librosa.time_to_samples(s, sr=sr)
        end_sample = librosa.time_to_samples(e, sr=sr)

        # DEBUG: Stampa informazioni sugli indici
        print(f"\n--- Evento {i} ---")
        print(f"Tempo: {s:.2f}s → {e:.2f}s (durata: {e-s:.2f}s)")
        print(f"Campioni calcolati: {start_sample} → {end_sample}")

        # Controllo fondamentale: verifica che gli indici siano validi per y_filtered
        if start_sample >= len(y_filtered):
            print(f"SKIP: start_sample ({start_sample}) >= lunghezza y_filtered ({len(y_filtered)})")
            skipped_clips += 1
            continue

        # Limita end_sample alla lunghezza di y_filtered
        end_sample = min(end_sample, len(y_filtered))

        # Controlla che start_sample < end_sample
        if start_sample >= end_sample:
            print(f"SKIP: start_sample ({start_sample}) >= end_sample ({end_sample})")
            skipped_clips += 1
            continue

        # Estrai il clip
        clip = y_filtered[start_sample:end_sample]

        # Controlla che il clip non sia vuoto
        if len(clip) == 0:
            print(f"SKIP: clip vuoto (lunghezza 0)")
            skipped_clips += 1
            continue

        # Clip valido
        audio_clips.append(clip)
        valid_clips += 1

        print(f"Clip estratto: {len(clip)} campioni ({len(clip)/sr:.2f} secondi)")
        print(f"Evento {i} — {s:.2f}s → {e:.2f}s (durata {e-s:.2f}s)")

        # Riproduci il clip
        display(Audio(clip, rate=sr))

        # Salva se necessario
        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            sf.write(f"{save_folder}/event_{i}.wav", clip, sr)

    print(f"\n\nRIEPILOGO:")
    print(f"Eventi totali: {len(events)}")
    print(f"Clip validi estratti: {valid_clips}")
    print(f"Clip saltati: {skipped_clips}")
