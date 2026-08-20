import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional

from datasets.canonical import CANONICAL_CLASSES, label_to_index
from audio.normalization import apply_normalization
from audio.io import resample_audio
from features.extractor import extract_features
from training.augment import apply_audio_augmentation


def balance_events(
    events: List[Dict[str, Any]],
    balance_strategy: str = "A",
    max_skaid_per_class: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Subsamples and balances events based on strategy:
    - BALANCE A: Natural distribution
    - BALANCE B: Equal dataset weighting
    - BALANCE C: Class-balanced dataset
    - BALANCE D: Dataset + class balanced
    """
    strat = balance_strategy.upper()
    df = pd.DataFrame(events)

    if df.empty:
        return events

    # Subsample SKAID if requested
    if max_skaid_per_class is not None:
        skaid_mask = df["dataset"] == "SKAID"
        skaid_df = df[skaid_mask]
        non_skaid_df = df[~skaid_mask]

        skaid_subsampled = skaid_df.groupby("canonical_label").apply(
            lambda g: g.sample(min(len(g), max_skaid_per_class), random_state=42)
        ).reset_index(drop=True)

        df = pd.concat([non_skaid_df, skaid_subsampled], ignore_index=True)

    if strat == "A":
        balanced_df = df

    elif strat == "B":
        # Equal dataset weighting
        min_dataset_size = df["dataset"].value_counts().min()
        balanced_df = df.groupby("dataset").apply(
            lambda g: g.sample(min_dataset_size, random_state=42)
        ).reset_index(drop=True)

    elif strat == "C":
        # Class-balanced
        min_class_size = df["canonical_label"].value_counts().min()
        balanced_df = df.groupby("canonical_label").apply(
            lambda g: g.sample(min_class_size, random_state=42)
        ).reset_index(drop=True)

    elif strat == "D":
        # Dataset + class balanced
        min_count = df.groupby(["dataset", "canonical_label"]).size().min()
        balanced_df = df.groupby(["dataset", "canonical_label"]).apply(
            lambda g: g.sample(min_count, random_state=42)
        ).reset_index(drop=True)

    else:
        balanced_df = df

    # Convert back to list of dicts
    indices = balanced_df.index.tolist()
    return [events[i] for i in indices if i < len(events)]


def prepare_dataset_matrix(
    events: List[Dict[str, Any]],
    target_sr: int = 22050,
    feature_set: str = "D",
    normalization: str = "P2",
    augment: bool = False,
    n_mels: int = 80
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts features for all events and returns:
    X: Feature matrix
    y: Labels array (0..26)
    groups: Recording/Session group IDs
    datasets: Dataset name string IDs
    """
    X_list = []
    y_list = []
    group_list = []
    dataset_list = []

    for ev in events:
        # 1. Get raw audio clip
        if "audio_clip" in ev:
            audio = ev["audio_clip"]
            sr = ev["sample_rate"]
        elif "audio_bytes" in ev:
            from audio.io import load_audio_from_bytes
            audio, sr = load_audio_from_bytes(ev["audio_bytes"], target_sr=target_sr)

        # 2. Resample if needed
        if sr != target_sr:
            audio = resample_audio(audio, sr, target_sr)

        # 3. Augment if training flag active
        if augment:
            audio = apply_audio_augmentation(audio)

        # 4. Normalize
        audio = apply_normalization(audio, strategy=normalization)

        # 5. Extract features
        feat = extract_features(audio, sr=target_sr, feature_set=feature_set, n_mels=n_mels)

        X_list.append(feat)
        y_list.append(ev["class_index"])
        group_list.append(ev.get("recording_id", "unknown_rec"))
        dataset_list.append(ev.get("dataset", "unknown_ds"))

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups = np.array(group_list)
    datasets_arr = np.array(dataset_list)

    return X, y, groups, datasets_arr
