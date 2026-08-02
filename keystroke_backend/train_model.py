"""
Train a RandomForest model to predict per-word keystroke counts from YAMNet
per-onset embeddings, pooled per ground-truth word.

NOTE: trained on 6/18 sessions (rest had empty keylogs, no-detect, or
incomplete data). Treat accuracy as provisional until dataset is expanded.
keystroke_classifier.joblib was assumed canonical based on filename
convention, not confirmed by the team — verify before relying on this.
Step 8 requires a quality-passing manifest. Only typing recordings assigned to
the train split are fitted; validation IDs are recorded as tuning provenance
and test recordings are never opened by this trainer.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

from onset_detector import detect_onsets
from segmenter import segment_into_words
from src.yamnet_config import GAP_THRESHOLD_ML
from src.training_protocol import (
    artifact_provenance,
    load_training_plan,
)
from yamnet_filter import extract_features, extract_yamnet_candidates, pool_word_features

try:  # package and script invocation styles are both supported
    from src.evaluation import parse_keylog
except ModuleNotFoundError:  # pragma: no cover - depends on import context
    from keystroke_backend.src.evaluation import parse_keylog

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"
OUTPUT_MODEL = MODELS_DIR / "count_predictor_candidate.joblib"

def _default_meta_path(log_path: str | Path) -> Path:
    """Return the capture metadata path conventionally paired with a keylog."""

    path = Path(log_path)
    stem = path.stem[:-4] if path.stem.endswith("_log") else path.stem
    return path.with_name(f"{stem}_meta.json")


def _read_sync_offset_ms(
    log_path: str | Path,
    *,
    sync_offset_ms: float | None = None,
    meta_path: str | Path | None = None,
) -> float:
    """Resolve a keylogger-to-audio offset from an explicit value or metadata.

    Legacy callers passed only a log path.  For those calls, use the sibling
    ``*_meta.json`` when present; an absent metadata file means no correction.
    An explicit offset always wins, which keeps old scripts easy to reproduce.
    """

    if sync_offset_ms is not None:
        try:
            value = float(sync_offset_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError("sync_offset_ms must be a finite number") from exc
    else:
        candidate = Path(meta_path) if meta_path is not None else _default_meta_path(log_path)
        if not candidate.is_file():
            return 0.0
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read capture metadata: {candidate}") from exc
        raw_value = metadata.get("sync_offset_ms", 0.0)
        try:
            value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Metadata field sync_offset_ms must be numeric: {candidate}"
            ) from exc

    if not math.isfinite(value):
        raise ValueError("sync_offset_ms must be a finite number")
    return value


def parse_ground_truth(
    log_path: str,
    sync_offset_ms: float | None = None,
    *,
    meta_path: str | Path | None = None,
) -> tuple[list[int], list[float], list[float]]:
    """
    Parse a keylog CSV and return per-word counts and the time range of each word.

    Returns
    -------
    word_counts : list[int]
        True keystroke count per word.
    word_starts : list[float]
        Onset timestamp of the first keypress in each word (seconds).
    word_ends   : list[float]
        Onset timestamp of the last  keypress in each word (seconds).
    """
    offset = _read_sync_offset_ms(
        log_path,
        sync_offset_ms=sync_offset_ms,
        meta_path=meta_path,
    )
    truth = parse_keylog(log_path, sync_offset_ms=offset)
    return (
        list(truth.counts),
        [word.start_time_seconds for word in truth.words],
        [word.end_time_seconds for word in truth.words],
    )


def build_training_data(
    session_names: list[str] | None = None,
    pool_by: str = "segmenter",
    *,
    recordings=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build feature matrix *X* and target vector *y* from rule-based onset
    candidates paired with ground-truth word counts.

    For each word in each session:
    1. Detect all rule-based onsets for the full recording.
    2. Extract per-onset YAMNet features (1028-D) in one batch.
    3. Group onsets into words, then pool each group via
       `pool_word_features` → one 1032-D feature per word.
    4. Target = ground-truth keystroke count for that word.

    pool_by
    -------
    "segmenter"   : group onsets with `segment_into_words`, exactly as the
                    serving path does, then label each group by maximum time
                    overlap with the keylog word windows. Default, because
                    training on ground-truth groupings and serving on
                    segmenter groupings is a train/serve skew.
    "groundtruth" : group onsets by the keylog word windows. Optimistic —
                    the serving path has no access to these boundaries.

    Returns
    -------
    X      : (n_words, 1032)
    y      : (n_words,)
    groups : (n_words,)  — session-name label for each word.
    """
    if pool_by not in {"segmenter", "groundtruth"}:
        raise ValueError("pool_by must be 'segmenter' or 'groundtruth'")

    if recordings is not None and session_names is not None:
        raise ValueError("pass session_names or recordings, not both")
    if recordings is None:
        if session_names is None:
            raise ValueError("session_names or recordings is required")
        sources = [
            (
                name,
                DATA_DIR / f"{name}.wav",
                DATA_DIR / f"{name}_log.csv",
                DATA_DIR / f"{name}_meta.json",
                name,
            )
            for name in session_names
        ]
    else:
        sources = [
            (
                entry.id,
                entry.wav_path,
                entry.log_path,
                entry.meta_path,
                entry.setup_id,
            )
            for entry in recordings
        ]

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    groups_list: list[str] = []

    for session_name, wav_path, log_path, meta_path, group_label in sources:

        if not wav_path.exists() or not log_path.exists():
            print(f"  Skipping {session_name}: missing .wav or .csv")
            continue

        print(f"  Processing {session_name} ...")

        # ── Parse ground truth ────────────────────────────────────────────
        word_counts, word_starts, word_ends = parse_ground_truth(
            str(log_path), meta_path=meta_path
        )
        print(f"    {len(word_counts)} words, true counts = {word_counts}")

        # ── Rule-based onset detection ────────────────────────────────────
        onsets, _, _ = detect_onsets(str(wav_path), delta=0.07)
        print(f"    {len(onsets)} rule-based onsets")

        if len(onsets) == 0:
            continue

        # ── YAMNet features for all onsets at once ────────────────────────
        candidates = extract_yamnet_candidates(str(wav_path), onsets)
        if not candidates:
            continue
        all_features = extract_features(candidates, str(wav_path))  # (n_onsets, 1028)

        onset_times = np.array([c.onset_time for c in candidates])

        # ── Pool per word ─────────────────────────────────────────────────
        PADDING_SEC = 0.05  # small tolerance around the ground-truth window
        n_words_added = 0

        if pool_by == "groundtruth":
            for wi in range(len(word_counts)):
                start_t = word_starts[wi] - PADDING_SEC
                end_t = word_ends[wi] + PADDING_SEC

                mask = (onset_times >= start_t) & (onset_times <= end_t)
                if not mask.any():
                    continue

                X_list.append(pool_word_features(all_features[mask], onset_times[mask]))
                y_list.append(word_counts[wi])
                groups_list.append(group_label)
                n_words_added += 1
        else:
            # Group onsets the way the serving path does, then label each
            # group by which ground-truth word it overlaps most.
            _, word_groups = segment_into_words(onsets, gap_threshold=GAP_THRESHOLD_ML)

            for group in word_groups:
                group = np.asarray(group, dtype=float)
                mask = np.isin(onset_times, group)
                if not mask.any():
                    continue

                g_start, g_end = float(group[0]), float(group[-1])
                overlaps = [
                    max(0.0, min(g_end, word_ends[wi] + PADDING_SEC)
                             - max(g_start, word_starts[wi] - PADDING_SEC))
                    for wi in range(len(word_counts))
                ]
                if not overlaps or max(overlaps) <= 0.0:
                    continue

                X_list.append(pool_word_features(all_features[mask], onset_times[mask]))
                y_list.append(word_counts[int(np.argmax(overlaps))])
                groups_list.append(group_label)
                n_words_added += 1

        print(f"    Added {n_words_added} word-level samples (pool_by={pool_by})")

    if not X_list:
        raise RuntimeError("No training data produced — check session files.")

    return np.vstack(X_list), np.array(y_list), np.array(groups_list)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a RandomForest regressor that predicts per-word keystroke "
            "counts from mean-pooled YAMNet onset embeddings."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Quality-passing private dataset manifest",
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUT_MODEL),
        help="Output candidate model path",
    )
    parser.add_argument(
        "--pool-by",
        choices=["segmenter", "groundtruth"],
        default="segmenter",
        help=(
            "How to group onsets into words. 'segmenter' matches the serving "
            "path (default); 'groundtruth' uses keylog boundaries the server "
            "never has, so it reports optimistic scores."
        ),
    )
    args = parser.parse_args()

    plan = load_training_plan(args.manifest)

    # ── Build dataset ─────────────────────────────────────────────────────
    print(f"Building training data from {len(plan.training_recordings)} recording(s)...")
    X, y, groups = build_training_data(
        pool_by=args.pool_by,
        recordings=plan.training_recordings,
    )
    print(f"\n  Total word samples : {len(X)}")
    print(f"  Feature dimension : {X.shape[1]}")
    print(f"  Count distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # ── Leave-One-Session-Out CV ──────────────────────────────────────────
    print("\nLeave-One-Session-Out Cross-Validation:")
    unique_sessions = np.unique(groups)
    fold_mae: list[float] = []
    fold_acc: list[float] = []

    for held_out in unique_sessions:
        test_mask = groups == held_out
        train_mask = ~test_mask

        if train_mask.sum() < 5:
            print(f"  Hold out {held_out}: skipped (only {train_mask.sum()} training samples)")
            continue

        X_tr, X_te = X[train_mask], X[test_mask]
        y_tr, y_te = y[train_mask], y[test_mask]

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        y_pred = model.predict(X_te)
        y_pred_int = np.round(y_pred).astype(int)
        mae = float(mean_absolute_error(y_te, y_pred))
        acc = float(accuracy_score(y_te, y_pred_int))

        fold_mae.append(mae)
        fold_acc.append(acc)
        print(
            f"  Hold out {held_out:15s}: "
            f"MAE={mae:.3f},  exact-match acc={acc:.3f}  "
            f"({len(X_te)} words)"
        )

    if fold_mae:
        print(
            f"\n  Mean held-out MAE  : {np.mean(fold_mae):.3f}  "
            f"(±{np.std(fold_mae):.3f})"
        )
        print(
            f"  Mean held-out acc  : {np.mean(fold_acc):.3f}  "
            f"(±{np.std(fold_acc):.3f})"
        )

    # ── Train final model on all data ─────────────────────────────────────
    print(f"\nTraining final model on all {len(X)} samples ...")
    final_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        random_state=42,
    )
    final_model.fit(X, y)

    # ── Feature importance overview ───────────────────────────────────────
    importances = final_model.feature_importances_
    top5 = np.argsort(importances)[::-1][:5]
    scalar_names = {1028: "n_onsets", 1029: "duration", 1030: "mean_gap", 1031: "std_gap"}
    print("\nTop 5 most important feature dimensions:")
    for rank, idx in enumerate(top5, 1):
        if idx in scalar_names:
            region = f"count scalar: {scalar_names[idx]}"
        elif idx < 1024:
            region = "YAMNet embedding"
        else:
            region = "local DSP feature"
        print(f"  {rank}. dim {idx:4d} ({region}):  {importances[idx]:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────
    fit_config = {
        "estimator": "RandomForestRegressor",
        "n_estimators": 200,
        "max_depth": 12,
        "min_samples_leaf": 2,
        "random_state": 42,
        "pool_by": args.pool_by,
    }
    payload = {
        "model": final_model,
        "feature_type": "YAMNet_mean_pooled_plus_count_scalars",
        "feature_dimension": int(X.shape[1]),
        "pool_by": args.pool_by,
        "n_word_samples": int(len(X)),
        "held_out_mae_mean": float(np.mean(fold_mae)) if fold_mae else None,
        "held_out_acc_mean": float(np.mean(fold_acc)) if fold_acc else None,
        **artifact_provenance(
            plan,
            artifact_role="count_predictor",
            fit_config=fit_config,
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, output_path)
    print(f"\nSaved model to: {output_path}")


if __name__ == "__main__":
    main()
