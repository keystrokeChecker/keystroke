import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import numpy as np
import pandas as pd
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Dict, List, Any

from datasets.skaid import SkaidDatasetLoader
from audio.quality import check_audio_quality

OFFSETS_MS = [-100.0, -75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0, 100.0]


def run_skaid_alignment_experiment(
    rec_zip_path: str,
    log_zip_path: str,
    output_debug_dir: str,
    sample_count: int = 500
) -> Dict[str, Any]:
    """
    Evaluates SKAID timestamp alignment across -100ms to +100ms offset range.
    Measures transient energy and spectral centroid for 500+ random SKAID events.
    Saves debug spectrograms and waveform statistics to output_debug_dir.
    """
    os.makedirs(output_debug_dir, exist_ok=True)
    loader = SkaidDatasetLoader(rec_zip_path, log_zip_path)

    print(f"=== SKAID TIMESTAMP ALIGNMENT TEST (n={sample_count}) ===")
    
    # Load events centered at 0 ms offset
    events = loader.load_events(offset_ms=0.0, window_duration_s=0.300)
    if not events:
        raise ValueError("No valid SKAID events loaded for alignment test.")

    # Random sample of 500 events
    np.random.seed(42)
    sample_indices = np.random.choice(len(events), size=min(sample_count, len(events)), replace=False)
    sampled_events = [events[i] for i in sample_indices]

    best_offsets = []
    event_debug_records = []

    for idx, ev in enumerate(sampled_events):
        audio = ev["audio_clip"]
        sr = ev["sample_rate"]

        offset_energies = {}
        for off_ms in OFFSETS_MS:
            # Shift window by off_ms
            shift_samples = int((off_ms / 1000.0) * sr)
            center = len(audio) // 2 + shift_samples
            half_w = int(0.050 * sr)  # 100ms transient sub-window
            
            s_idx = max(0, center - half_w)
            e_idx = min(len(audio), center + half_w)
            sub_clip = audio[s_idx:e_idx]

            # RMS energy of sub-clip
            rms = float(np.sqrt(np.mean(sub_clip ** 2))) if len(sub_clip) > 0 else 0.0
            offset_energies[off_ms] = rms

        best_off = max(offset_energies, key=offset_energies.get)
        best_offsets.append(best_off)

        # Basic audio statistics for winning clip
        peak = float(np.max(np.abs(audio)))
        rms_val = float(np.sqrt(np.mean(audio ** 2)))
        cent = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))) if len(audio) > 0 else 0.0

        record = {
            "event_idx": idx,
            "participant": ev["participant_id"],
            "recording": ev["recording_id"],
            "timestamp_ms": ev["timestamp_ms"],
            "expected_key": ev["canonical_label"],
            "best_offset_ms": best_off,
            "peak_amp": peak,
            "rms": rms_val,
            "spectral_centroid": cent
        }
        event_debug_records.append(record)

        # Generate sample spectrograms for first 20 events
        if idx < 20:
            fig, ax = plt.subplots(figsize=(6, 3))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax)
            ax.set_title(f"SKAID {ev['canonical_label']} | Best Off: {best_off}ms")
            fig.tight_layout()
            fig.savefig(os.path.join(output_debug_dir, f"spec_event_{idx}_{ev['canonical_label']}.png"))
    best_off_series = pd.Series(best_offsets)
    dist = best_off_series.value_counts(normalize=True).to_dict()
    zero_offset_acc = float((best_off_series == 0.0).mean())

    debug_df = pd.DataFrame(event_debug_records)
    debug_df.to_csv(os.path.join(output_debug_dir, "alignment_event_statistics.csv"), index=False)

    print("\n--- SKAID Alignment Results ---")
    print(f"Alignment accuracy at 0ms offset: {zero_offset_acc * 100:.2f}%")
    print("Best offset distribution (ms):")
    for off, val in sorted(dist.items()):
        print(f"  {off:6.1f} ms: {val * 100:5.2f}%")

    return {
        "alignment_accuracy": zero_offset_acc,
        "best_offset_distribution": dist,
        "event_records_count": len(event_debug_records)
    }


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    data_dir = os.path.join(os.path.dirname(BASE_DIR), "keystroke_backend", "data")
    rec_zip = os.path.join(data_dir, "Participant Recordings_zip.zip")
    log_zip = os.path.join(data_dir, "Keystroke Logs_zip.zip")
    debug_dir = os.path.join(BASE_DIR, "data", "debug", "skaid_alignment")

    run_skaid_alignment_experiment(rec_zip_path=rec_zip, log_zip_path=log_zip, output_debug_dir=debug_dir, sample_count=500)
