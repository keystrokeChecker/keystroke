import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from typing import Dict, Any


def run_domain_and_participant_analysis(
    X: np.ndarray,
    y_keys: np.ndarray,
    datasets: np.ndarray,
    participants: np.ndarray,
    output_dir: str
) -> Dict[str, Any]:
    """
    Trains diagnostic classifiers (Features -> Dataset, Features -> Participant)
    and generates 2D PCA plots colored by Key, Dataset, and Participant.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=== RUNNING DOMAIN & PARTICIPANT DIAGNOSTIC ANALYSIS ===")

    # 1. Domain Classifier (Features -> Dataset ID)
    clf_domain = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_domain.fit(X, datasets)
    domain_preds = clf_domain.predict(X)
    domain_acc = float(accuracy_score(datasets, domain_preds))
    print(f"Domain Classification Accuracy (Kaggle vs SKAID vs MultiPressure): {domain_acc * 100:.2f}%")

    # 2. Participant Classifier (Features -> Participant ID)
    clf_part = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf_part.fit(X, participants)
    part_preds = clf_part.predict(X)
    part_acc = float(accuracy_score(participants, part_preds))
    print(f"Participant Classification Accuracy: {part_acc * 100:.2f}%")

    # 3. PCA Embedding Analysis
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X)

    # Plot 1: Colored by Dataset
    fig, ax = plt.subplots(figsize=(7, 5))
    for ds in np.unique(datasets):
        mask = (datasets == ds)
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=str(ds), alpha=0.6, s=15)
    ax.set_title("PCA Embedding Colored by Dataset Identity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_by_dataset.png"))
    plt.close(fig)

    # Plot 2: Colored by Key Class
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y_keys, cmap="tab20", alpha=0.6, s=15)
    ax.set_title("PCA Embedding Colored by Target Key Class (0-26)")
    fig.colorbar(scatter, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "pca_by_key.png"))
    plt.close(fig)

    # Summary report
    analysis_res = {
        "domain_classifier_accuracy": domain_acc,
        "participant_classifier_accuracy": part_acc,
        "pca_variance_explained": [float(v) for v in pca.explained_variance_ratio_]
    }

    with open(os.path.join(output_dir, "DOMAIN_ANALYSIS_SUMMARY.json"), 'w') as f:
        import json
        json.dump(analysis_res, f, indent=2)

    return analysis_res


if __name__ == "__main__":
    import sys
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if BASE_DIR not in sys.path:
        sys.path.insert(0, BASE_DIR)

    from datasets.kaggle import KaggleDatasetLoader
    from datasets.multipressure import MultiPressureDatasetLoader
    from datasets.skaid import SkaidDatasetLoader
    from training.prepare import prepare_dataset_matrix

    data_dir = os.path.join(BASE_DIR, "data", "raw")
    output_debug = os.path.join(BASE_DIR, "data", "debug", "domain")

    def find_file(filename: str) -> str:
        p1 = os.path.join(data_dir, filename)
        if os.path.exists(p1): return p1
        p2 = os.path.join(os.path.dirname(data_dir), "raw", filename)
        if os.path.exists(p2): return p2
        return p1

    k_loader = KaggleDatasetLoader(find_file('archive.zip'))
    mp_loader = MultiPressureDatasetLoader(find_file('dataset.zip'), find_file('Keystrokes_Dataset.zip'))
    s_loader = SkaidDatasetLoader(find_file('Participant Recordings_zip.zip'), find_file('Keystroke Logs_zip.zip'))

    events = k_loader.load_events(max_samples_per_class=100) + mp_loader.load_events() + s_loader.load_events(max_samples_per_class=100)
    X, y_keys, groups, datasets = prepare_dataset_matrix(events, feature_set="D", normalization="P2")
    participants = np.array([e.get("participant_id", "unknown") for e in events])

    run_domain_and_participant_analysis(X, y_keys, datasets, participants, output_debug)
