import os
import zipfile
import csv
import io
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from datasets.canonical import sanitize_raw_label, label_to_index, is_forbidden_recording
from audio.io import load_audio_from_bytes
from audio.onset import align_onset_local_neighborhood


class SkaidDatasetLoader:
    """
    Loader for the SKAID (Smartphone Keystroke Audio & Impression Dataset).
    Extracts keystroke events from M4A participant recordings aligned with CSV key log timestamps.
    Supports multi-method acoustic onset alignment and event quality scoring.
    """

    def __init__(self, rec_zip_path: str, log_zip_path: str):
        if not os.path.exists(rec_zip_path):
            raise FileNotFoundError(f"SKAID participant recordings zip not found at {rec_zip_path}")
        if not os.path.exists(log_zip_path):
            raise FileNotFoundError(f"SKAID keystroke logs zip not found at {log_zip_path}")

        self.rec_zip_path = rec_zip_path
        self.log_zip_path = log_zip_path

    def get_session_manifest(self) -> List[Dict[str, Any]]:
        manifest = []
        with zipfile.ZipFile(self.log_zip_path, 'r') as zf_log, zipfile.ZipFile(self.rec_zip_path, 'r') as zf_rec:
            log_names = [n for n in zf_log.namelist() if n.endswith('.csv') and 'full_text' not in n]
            rec_names = [n for n in zf_rec.namelist() if n.endswith('.m4a')]

            for log_path in sorted(log_names):
                if is_forbidden_recording(log_path):
                    continue

                session_id = os.path.basename(log_path).replace('.csv', '')
                participant_id = session_id.split('_')[0]

                # Match Phase 1 and Phase 2 audio files
                phase1_rec = next((r for r in rec_names if session_id in r and ('_email_1' in r or 'phase1' in r.lower())), None)
                phase2_rec = next((r for r in rec_names if session_id in r and ('_free_form' in r or 'phase2' in r.lower())), None)

                # Fallback matching if phase suffix is different
                if not phase1_rec:
                    phase1_rec = next((r for r in rec_names if session_id in r and ('_1' in r or 'email' in r)), None)
                if not phase2_rec:
                    phase2_rec = next((r for r in rec_names if session_id in r and ('_2' in r or 'free' in r)), None)

                manifest.append({
                    "session_id": session_id,
                    "participant_id": participant_id,
                    "log_path": log_path,
                    "phase1_rec": phase1_rec,
                    "phase2_rec": phase2_rec
                })

        return manifest

    def load_events(
        self,
        max_samples_per_class: Optional[int] = None,
        target_sr: int = 22050,
        window_duration_s: float = 0.200,
        offset_ms: float = 0.0,
        use_acoustic_alignment: bool = True,
        alignment_method: str = "hybrid",
        alignment_search_ms: float = 150.0,
        quality_filter: bool = False
    ) -> List[Dict[str, Any]]:
        events = []
        class_counts: Dict[str, int] = {}
        manifest = self.get_session_manifest()

        with zipfile.ZipFile(self.rec_zip_path, 'r') as zf_rec, zipfile.ZipFile(self.log_zip_path, 'r') as zf_log:
            for item in manifest:
                session_id = item["session_id"]
                participant_id = item["participant_id"]

                log_bytes = zf_log.read(item["log_path"])
                try:
                    reader = csv.DictReader(io.StringIO(log_bytes.decode('utf-8')))
                    log_rows = list(reader)
                except Exception:
                    continue

                if not log_rows:
                    continue

                phase_map = [
                    ('Phase 1', item["phase1_rec"]),
                    ('Phase 2', item["phase2_rec"])
                ]

                for phase_name, rec_path in phase_map:
                    if not rec_path:
                        continue

                    phase_rows = [r for r in log_rows if r.get('Phase') == phase_name]
                    if not phase_rows:
                        continue

                    valid_phase_rows = []
                    for row in phase_rows:
                        if row.get('Event') != 'press':
                            continue
                        raw_key = row.get('Key', '')
                        canonical_label = sanitize_raw_label(raw_key)
                        if canonical_label is not None and not is_forbidden_recording(rec_path):
                            valid_phase_rows.append((row, canonical_label))

                    if not valid_phase_rows:
                        continue

                    m4a_bytes = zf_rec.read(rec_path)
                    try:
                        audio, sr = load_audio_from_bytes(m4a_bytes, target_sr=target_sr)
                    except Exception as e:
                        continue

                    first_phase_row_ts = int(float(phase_rows[0]['Timestamp (ms)']))

                    for row, canonical_label in valid_phase_rows:
                        if max_samples_per_class is not None:
                            if class_counts.get(canonical_label, 0) >= max_samples_per_class:
                                continue

                        ts_ms = int(float(row['Timestamp (ms)']))
                        rel_time_s = (ts_ms - first_phase_row_ts) / 1000.0 + (offset_ms / 1000.0)

                        if rel_time_s < 0 or rel_time_s > (len(audio) / sr):
                            continue

                        logged_center_idx = int(rel_time_s * sr)

                        if use_acoustic_alignment:
                            best_center_idx, q_info = align_onset_local_neighborhood(
                                audio=audio,
                                sr=sr,
                                logged_center_idx=logged_center_idx,
                                search_radius_ms=alignment_search_ms,
                                window_duration_s=window_duration_s,
                                method=alignment_method
                            )
                        else:
                            best_center_idx = logged_center_idx
                            q_info = {
                                "onset_confidence": 1.0,
                                "method_used": "fixed_timestamp",
                                "offset_ms": 0.0,
                                "peak_amp": float(np.max(np.abs(audio[max(0, logged_center_idx - 1000):min(len(audio), logged_center_idx + 1000)]))) if len(audio) > 0 else 0.0,
                                "rms": float(np.sqrt(np.mean(audio[max(0, logged_center_idx - 1000):min(len(audio), logged_center_idx + 1000)]**2))) if len(audio) > 0 else 0.0,
                                "snr_db": 10.0,
                                "overlap_indicator": False,
                                "quality_tier": "valid"
                            }

                        if quality_filter and q_info.get("quality_tier") == "unusable":
                            continue

                        target_samples = int(window_duration_s * sr)
                        half_w = target_samples // 2
                        start_idx = best_center_idx - half_w
                        end_idx = start_idx + target_samples

                        s_valid = max(0, start_idx)
                        e_valid = min(len(audio), end_idx)

                        raw_clip = audio[s_valid:e_valid]
                        pad_left = s_valid - start_idx if start_idx < 0 else 0
                        pad_right = target_samples - len(raw_clip) - pad_left
                        pad_right = max(0, pad_right)

                        if pad_left > 0 or pad_right > 0:
                            clip = np.pad(raw_clip, (pad_left, pad_right), mode='constant')
                        else:
                            clip = raw_clip[:target_samples]

                        class_idx = label_to_index(canonical_label)
                        class_counts[canonical_label] = class_counts.get(canonical_label, 0) + 1
                        rec_id = f"skaid_{session_id}_{phase_name.replace(' ', '_')}"

                        events.append({
                            "dataset": "SKAID",
                            "original_label": row.get('Key', ''),
                            "canonical_label": canonical_label,
                            "class_index": class_idx,
                            "session_id": session_id,
                            "participant_id": participant_id,
                            "recording_id": rec_id,
                            "phase": phase_name,
                            "timestamp_ms": ts_ms,
                            "rel_time_s": rel_time_s,
                            "aligned_center_s": best_center_idx / sr,
                            "alignment_offset_ms": q_info["offset_ms"],
                            "onset_confidence": q_info["onset_confidence"],
                            "quality_tier": q_info["quality_tier"],
                            "snr_db": q_info["snr_db"],
                            "overlap_indicator": q_info["overlap_indicator"],
                            "audio_clip": clip,
                            "sample_rate": sr,
                            "duration_s": window_duration_s,
                            "zip_item_path": rec_path
                        })

        return events
