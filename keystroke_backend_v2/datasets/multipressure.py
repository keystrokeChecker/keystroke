import os
import zipfile
import csv
import io
import soundfile as sf
import numpy as np
from typing import List, Dict, Any, Optional

from datasets.canonical import sanitize_raw_label, label_to_index, is_forbidden_recording


class MultiPressureDatasetLoader:
    """
    Loader for Multi-Pressure Dataset (dataset.zip and Keystrokes_Dataset.zip).
    Filters strictly to 27 canonical classes (A-Z + SPACE).
    Extracts pressure metadata (High, Medium, Low) for grouping and domain analysis.
    """

    def __init__(self, dataset_zip_path: str, keystrokes_zip_path: Optional[str] = None):
        self.dataset_zip_path = dataset_zip_path
        self.keystrokes_zip_path = keystrokes_zip_path

        if not os.path.exists(dataset_zip_path):
            raise FileNotFoundError(f"Multi-Pressure zip not found at {dataset_zip_path}")

    def load_events(self, max_samples_per_class: Optional[int] = None) -> List[Dict[str, Any]]:
        events = []
        class_counts: Dict[str, int] = {}

        # 1. Parse main dataset.zip
        with zipfile.ZipFile(self.dataset_zip_path, 'r') as zf:
            namelist = [n for n in zf.namelist() if n.lower().endswith('.wav')]
            
            for item in sorted(namelist):
                if is_forbidden_recording(item):
                    continue

                filename = os.path.basename(item)  # e.g., Key-A-H.wav
                folder = item.split('/')[1] if '/' in item else ''

                # Determine pressure level from filename or folder
                pressure = "Unknown"
                if "-H." in filename or "High" in folder:
                    pressure = "High"
                elif "-M." in filename or "Medium" in folder:
                    pressure = "Medium"
                elif "-L." in filename or "Low" in folder:
                    pressure = "Low"

                # Extract key component e.g. "A" from "Key-A-H.wav" or "Spacebar"
                raw_key = filename.replace('.wav', '')
                if raw_key.startswith("Key-"):
                    raw_key = raw_key[4:]
                if len(raw_key) >= 3 and raw_key[-2] == '-':
                    raw_key = raw_key[:-2]

                canonical_label = sanitize_raw_label(raw_key)
                if canonical_label is None:
                    continue

                if max_samples_per_class is not None:
                    if class_counts.get(canonical_label, 0) >= max_samples_per_class:
                        continue

                wav_bytes = zf.read(item)
                try:
                    data, sr = sf.read(io.BytesIO(wav_bytes))
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    duration_s = float(len(data) / sr)
                except Exception as e:
                    print(f"[MultiPressureLoader] Error reading {item}: {e}")
                    continue

                class_idx = label_to_index(canonical_label)
                class_counts[canonical_label] = class_counts.get(canonical_label, 0) + 1
                rec_id = f"multipressure_{pressure}_{filename}"

                events.append({
                    "dataset": "MultiPressure",
                    "original_label": raw_key,
                    "canonical_label": canonical_label,
                    "class_index": class_idx,
                    "pressure": pressure,
                    "recording_id": rec_id,
                    "participant_id": f"mp_participant_{pressure.lower()}",
                    "audio_bytes": wav_bytes,
                    "sample_rate": sr,
                    "duration_s": duration_s,
                    "samples_count": len(data),
                    "zip_item_path": item
                })

        # 2. Parse secondary Keystrokes_Dataset.zip if provided
        if self.keystrokes_zip_path and os.path.exists(self.keystrokes_zip_path):
            with zipfile.ZipFile(self.keystrokes_zip_path, 'r') as zf:
                namelist = [n for n in zf.namelist() if n.lower().endswith('.wav')]
                
                for item in sorted(namelist):
                    if is_forbidden_recording(item):
                        continue

                    filename = os.path.basename(item)
                    folder = item.split('/')[0] if '/' in item else ''

                    pressure = "Medium" if "Medium" in folder or "-M." in filename else "Unknown"
                    raw_key = filename.replace('.wav', '')
                    if raw_key.startswith("Key-"):
                        raw_key = raw_key[4:]
                    if len(raw_key) >= 3 and raw_key[-2] == '-':
                        raw_key = raw_key[:-2]

                    canonical_label = sanitize_raw_label(raw_key)
                    if canonical_label is None:
                        continue

                    if max_samples_per_class is not None:
                        if class_counts.get(canonical_label, 0) >= max_samples_per_class:
                            continue

                    wav_bytes = zf.read(item)
                    try:
                        data, sr = sf.read(io.BytesIO(wav_bytes))
                        if data.ndim > 1:
                            data = np.mean(data, axis=1)
                        duration_s = float(len(data) / sr)
                    except Exception as e:
                        print(f"[MultiPressureLoader] Error reading {item}: {e}")
                        continue

                    class_idx = label_to_index(canonical_label)
                    class_counts[canonical_label] = class_counts.get(canonical_label, 0) + 1
                    rec_id = f"multipressure_ks_{pressure}_{filename}"

                    events.append({
                        "dataset": "MultiPressure",
                        "original_label": raw_key,
                        "canonical_label": canonical_label,
                        "class_index": class_idx,
                        "pressure": pressure,
                        "recording_id": rec_id,
                        "participant_id": f"mp_participant_{pressure.lower()}",
                        "audio_bytes": wav_bytes,
                        "sample_rate": sr,
                        "duration_s": duration_s,
                        "samples_count": len(data),
                        "zip_item_path": item
                    })

        return events
