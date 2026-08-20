import os
import json
import joblib
import numpy as np
from typing import Dict, List, Any, Optional

from app.config import (
    MODEL_PATH, AUDIO_CONFIG_PATH, FEATURE_CONFIG_PATH,
    PREPROCESSING_CONFIG_PATH, THRESHOLDS_PATH, METADATA_PATH
)
from app.inference.preprocessing import preprocess_audio_bytes
from app.inference.segmentation import segment_audio_for_inference
from app.inference.prediction import format_top_k_predictions
from features.extractor import extract_features


class InferenceEngine:
    """
    Production Inference Engine reproducing exact training pipeline.
    """

    def __init__(self, prod_dir: Optional[str] = None):
        if prod_dir:
            model_p = os.path.join(prod_dir, "model.joblib")
            audio_cfg_p = os.path.join(prod_dir, "audio_config.json")
            feat_cfg_p = os.path.join(prod_dir, "feature_config.json")
            prep_cfg_p = os.path.join(prod_dir, "preprocessing_config.json")
            thresh_p = os.path.join(prod_dir, "thresholds.json")
            meta_p = os.path.join(prod_dir, "model_metadata.json")
        else:
            model_p, audio_cfg_p, feat_cfg_p, prep_cfg_p, thresh_p, meta_p = (
                MODEL_PATH, AUDIO_CONFIG_PATH, FEATURE_CONFIG_PATH,
                PREPROCESSING_CONFIG_PATH, THRESHOLDS_PATH, METADATA_PATH
            )

        self.is_loaded = False
        if os.path.exists(model_p):
            self.model = joblib.load(model_p)
            with open(audio_cfg_p, 'r') as f: self.audio_config = json.load(f)
            with open(feat_cfg_p, 'r') as f: self.feature_config = json.load(f)
            with open(prep_cfg_p, 'r') as f: self.prep_config = json.load(f)
            with open(thresh_p, 'r') as f: self.thresholds = json.load(f)
            with open(meta_p, 'r') as f: self.metadata = json.load(f)
            self.is_loaded = True

    def predict_single_clip(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Inference for a single isolated keystroke audio clip."""
        if not self.is_loaded:
            raise RuntimeError("Production model not loaded. Run training script first.")

        sr = self.audio_config.get("sample_rate", 22050)
        norm_strat = self.prep_config.get("normalization", "P2")
        fset = self.feature_config.get("feature_set", "C")

        audio, sr, quality_info = preprocess_audio_bytes(
            audio_bytes, target_sr=sr, normalization_strategy=norm_strat
        )

        feats = extract_features(audio, sr=sr, feature_set=fset)
        feats = np.expand_dims(feats, axis=0)

        probs = self.model.predict_proba(feats)[0]
        res = format_top_k_predictions(
            probs,
            k=3,
            confident_threshold=self.thresholds.get("confident_threshold", 0.60),
            low_confidence_threshold=self.thresholds.get("low_confidence_threshold", 0.35)
        )
        res["quality_info"] = quality_info
        return res

    def analyze_continuous_audio(self, audio_bytes: bytes) -> Dict[str, Any]:
        """Inference for continuous typing audio stream (silence check -> raw onset detection -> per-event P2 norm -> classification)."""
        if not self.is_loaded:
            raise RuntimeError("Production model not loaded. Run training script first.")

        sr = self.audio_config.get("sample_rate", 22050)
        norm_strat = self.prep_config.get("normalization", "P2")
        fset = self.feature_config.get("feature_set", "D")
        win_dur = self.audio_config.get("window_duration_s", 0.200)

        from audio.io import load_audio_from_bytes
        from audio.quality import check_audio_quality
        from audio.normalization import apply_normalization

        audio, loaded_sr = load_audio_from_bytes(audio_bytes, target_sr=sr)
        quality_info = check_audio_quality(audio, sr)

        # If entire recording is quiet / ambient room noise without keystrokes
        if quality_info.get("is_silent", False) or quality_info.get("peak_amplitude", 0.0) < 0.025:
            return {
                "counts": [0] * 27,
                "formatted": "No keystrokes detected",
                "detected_events_count": 0,
                "reconstructed_sequence": "",
                "events": [],
                "quality_info": quality_info
            }

        segments = segment_audio_for_inference(audio, sr=sr, window_duration_s=win_dur)
        if not segments:
            return {
                "counts": [0] * 27,
                "formatted": "No keystrokes detected",
                "detected_events_count": 0,
                "reconstructed_sequence": "",
                "events": [],
                "quality_info": quality_info
            }

        event_predictions = []
        for seg in segments:
            clip = seg["audio_clip"]
            if len(clip) == 0:
                continue
            # Apply P2 normalization to the individual 200ms keystroke clip (matching training)
            norm_clip = apply_normalization(clip, strategy=norm_strat)
            feats = extract_features(norm_clip, sr=sr, feature_set=fset)
            feats = np.expand_dims(feats, axis=0)

            probs = self.model.predict_proba(feats)[0]
            formatted = format_top_k_predictions(
                probs,
                k=3,
                confident_threshold=self.thresholds.get("confident_threshold", 0.60),
                low_confidence_threshold=self.thresholds.get("low_confidence_threshold", 0.35)
            )
            event_predictions.append({
                "event_index": seg["event_index"],
                "timestamp_s": seg["timestamp_s"],
                **formatted
            })

        valid_events = [e for e in event_predictions if e.get("status") != "UNKNOWN"]
        sequence_str = "".join([e["prediction"] if e["prediction"] != "SPACE" else " " for e in valid_events])

        counts = [0] * 27
        from datasets.canonical import label_to_index
        for e in valid_events:
            pred = e.get("prediction", "")
            try:
                idx = label_to_index(pred)
                counts[idx] += 1
            except Exception:
                pass

        if valid_events:
            formatted_summary = ", ".join([f"{e['prediction']} ({e['confidence']:.2f})" for e in valid_events[:5]])
        else:
            formatted_summary = "No keystrokes detected"

        return {
            "counts": counts,
            "formatted": formatted_summary if formatted_summary else (sequence_str if sequence_str else "No keystrokes detected"),
            "detected_events_count": len(valid_events),
            "reconstructed_sequence": sequence_str,
            "events": event_predictions,
            "quality_info": quality_info
        }
