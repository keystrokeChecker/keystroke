import os
import zipfile
import io
import soundfile as sf
import numpy as np
from typing import List, Dict, Any, Optional

from datasets.canonical import sanitize_raw_label, label_to_index, is_forbidden_recording


class KaggleDatasetLoader:
    """
    Loader for Kaggle Keyboard Sound Dataset (archive.zip).
    Filters strictly to 27 canonical classes (A-Z + SPACE).
    """

    def __init__(self, zip_path: str):
        self.zip_path = zip_path
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Kaggle zip dataset not found at {zip_path}")

    def load_events(self, max_samples_per_class: Optional[int] = None) -> List[Dict[str, Any]]:
        events = []
        class_counts: Dict[str, int] = {}

        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            namelist = [n for n in zf.namelist() if n.lower().endswith('.wav')]
            
            for item in sorted(namelist):
                if is_forbidden_recording(item):
                    continue

                parts = item.split('/')
                if len(parts) < 2:
                    continue

                raw_folder = parts[1]
                canonical_label = sanitize_raw_label(raw_folder)
                if canonical_label is None:
                    continue

                if max_samples_per_class is not None:
                    current_cnt = class_counts.get(canonical_label, 0)
                    if current_cnt >= max_samples_per_class:
                        continue

                filename = parts[-1]
                rec_id = f"kaggle_{raw_folder}_{filename}"

                # Read audio metadata quickly
                wav_bytes = zf.read(item)
                try:
                    data, sr = sf.read(io.BytesIO(wav_bytes))
                    if data.ndim > 1:
                        data = np.mean(data, axis=1)
                    duration_s = float(len(data) / sr)
                except Exception as e:
                    print(f"[KaggleLoader] Error reading {item}: {e}")
                    continue

                class_idx = label_to_index(canonical_label)
                class_counts[canonical_label] = class_counts.get(canonical_label, 0) + 1

                events.append({
                    "dataset": "Kaggle",
                    "original_label": raw_folder,
                    "canonical_label": canonical_label,
                    "class_index": class_idx,
                    "recording_id": rec_id,
                    "participant_id": "kaggle_participant",
                    "audio_bytes": wav_bytes,
                    "sample_rate": sr,
                    "duration_s": duration_s,
                    "samples_count": len(data),
                    "zip_item_path": item
                })

        return events
