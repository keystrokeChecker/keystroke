"""
Script to test sensitivity delta values against labeled phone recordings in data/uploads/
File naming convention expected: <label>_count<N>.wav (e.g., letter_a_count22.wav, silence_count0.wav)
"""

import os
import re
from pathlib import Path
import numpy as np

from onset_detector import detect_onsets

UPLOADS_DIR = Path(__file__).resolve().parent / "data" / "uploads"

def parse_true_count(filename: str) -> int | None:
    match = re.search(r"count(\d+)", filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None

def test_deltas(delta_list=None):
    if delta_list is None:
        delta_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10, 0.12, 0.15, 0.20]
    
    if not UPLOADS_DIR.exists():
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

    files = [f for f in UPLOADS_DIR.glob("*.wav") if parse_true_count(f.name) is not None]
    
    if not files:
        print(f"⚠️ No labeled files found in {UPLOADS_DIR}")
        print("Please place phone recordings in data/uploads/ named like: letter_a_count22.wav, space_count15.wav, silence_count0.wav")
        return None

    print(f"\n========================================================")
    print(f"  SWEEPING DELTA VALUES ON {len(files)} PHONE RECORDINGS")
    print(f"========================================================")
    for f in files:
        print(f"  - {f.name} (True Count: {parse_true_count(f.name)})")
    print("-" * 56)

    results = []

    for delta in delta_list:
        errors = []
        file_details = []
        for wav_path in files:
            true_count = parse_true_count(wav_path.name)
            onsets, _, _ = detect_onsets(str(wav_path), delta=delta)
            pred_count = len(onsets)
            err = abs(pred_count - true_count)
            errors.append(err)
            file_details.append((wav_path.name, true_count, pred_count, err))

        mae = float(np.mean(errors))
        results.append((delta, mae, errors, file_details))

    # Sort results by Mean Absolute Error (MAE)
    results.sort(key=lambda x: x[1])

    print("\nRESULTS SUMMARY (Ranked by Minimum Error):")
    print(f"{'Delta':<10} | {'MAE':<10} | {'Per-file (True -> Pred)':<40}")
    print("-" * 65)
    for delta, mae, errors, file_details in results:
        details_str = ", ".join([f"{name.split('.')[0]}: {tc}->{pc}" for name, tc, pc, _ in file_details])
        print(f"{delta:<10.3f} | {mae:<10.3f} | {details_str}")

    best_delta, best_mae, _, best_details = results[0]
    print(f"\n========================================================")
    print(f"  BEST DELTA: {best_delta:.3f} (MAE: {best_mae:.3f})")
    print(f"========================================================\n")
    return best_delta

if __name__ == "__main__":
    test_deltas()