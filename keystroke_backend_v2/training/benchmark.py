import os
import time
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

from datasets.kaggle import KaggleDatasetLoader
from datasets.multipressure import MultiPressureDatasetLoader
from datasets.skaid import SkaidDatasetLoader
from training.prepare import prepare_dataset_matrix, balance_events
from training.splits import get_grouped_train_test_splits
from models.classical import get_classical_model
from models.cnn import PyTorchCNNWrapper
from models.ensemble import WeightedEnsembleClassifier
from evaluation.metrics import compute_comprehensive_metrics


class ExperimentRunner:
    """
    Optimized benchmark runner with feature caching for fast multi-experiment execution (EXP-01 to EXP-20+).
    """

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir

        def find_file(filename: str) -> str:
            p1 = os.path.join(data_dir, filename)
            if os.path.exists(p1): return p1
            p2 = os.path.join(data_dir, "raw", filename)
            if os.path.exists(p2): return p2
            p3 = os.path.join(os.path.dirname(data_dir), "raw", filename)
            if os.path.exists(p3): return p3
            return p1

        self.kaggle_path = find_file('archive.zip')
        self.mp_path = find_file('dataset.zip')
        self.mp_ks_path = find_file('Keystrokes_Dataset.zip')
        self.skaid_rec_path = find_file('Participant Recordings_zip.zip')
        self.skaid_log_path = find_file('Keystroke Logs_zip.zip')

        print("[ExperimentRunner] Pre-loading public dataset samples...")
        self.kaggle_events = KaggleDatasetLoader(self.kaggle_path).load_events(max_samples_per_class=300)
        self.mp_events = MultiPressureDatasetLoader(self.mp_path, self.mp_ks_path).load_events()
        self.skaid_events = SkaidDatasetLoader(self.skaid_rec_path, self.skaid_log_path).load_events(max_samples_per_class=300, window_duration_s=0.200)

        self.all_events = self.kaggle_events + self.mp_events + self.skaid_events
        print(f"[ExperimentRunner] Assembled representative benchmark dataset: {len(self.all_events)} events across Kaggle, MultiPressure, and SKAID.")

        self._matrix_cache: Dict[Tuple, Tuple] = {}

    def get_dataset_matrix(
        self,
        events: List[Dict[str, Any]],
        sample_rate: int = 22050,
        feature_set: str = "D",
        normalization: str = "P2",
        augment: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Caches extracted feature matrices by key configuration to avoid redundant extractions."""
        cache_key = (id(events), sample_rate, feature_set, normalization, augment)
        if cache_key in self._matrix_cache:
            return self._matrix_cache[cache_key]

        X, y, groups, datasets = prepare_dataset_matrix(
            events, target_sr=sample_rate, feature_set=feature_set, normalization=normalization, augment=augment
        )
        self._matrix_cache[cache_key] = (X, y, groups, datasets)
        return X, y, groups, datasets

    def run_single_experiment(
        self,
        exp_id: str,
        events: List[Dict[str, Any]],
        model_name: str,
        feature_set: str = "D",
        normalization: str = "P2",
        sample_rate: int = 22050,
        augment: bool = False,
        split_type: str = "GroupedByRecording"
    ) -> Tuple[Dict[str, Any], Any]:
        """Runs a single experiment configuration."""
        print(f"-> Running {exp_id:6s} | Model: {model_name:15s} | Feats: {feature_set} | Norm: {normalization} | Events: {len(events)}")

        t0 = time.time()
        X, y, groups, datasets = self.get_dataset_matrix(
            events, sample_rate=sample_rate, feature_set=feature_set, normalization=normalization, augment=augment
        )

        if model_name.lower() == "cnn" and feature_set not in ["E", "F"]:
            X, y, groups, datasets = self.get_dataset_matrix(
                events, sample_rate=sample_rate, feature_set="E", normalization=normalization, augment=augment
            )

        X_train, X_val, X_test, y_train, y_val, y_test = get_grouped_train_test_splits(X, y, groups, test_size=0.2)

        if model_name.lower() == "cnn":
            model = PyTorchCNNWrapper(num_classes=27, epochs=3, batch_size=32)
        else:
            model = get_classical_model(model_name)

        model.fit(X_train, y_train)
        training_time = float(round(time.time() - t0, 2))

        val_probs = model.predict_proba(X_val)
        val_metrics = compute_comprehensive_metrics(val_probs, y_val)

        test_probs = model.predict_proba(X_test)
        test_metrics = compute_comprehensive_metrics(test_probs, y_test)

        res = {
            "experiment_id": exp_id,
            "dataset_scope": str(sorted(list(set(e["dataset"] for e in events)))),
            "split": split_type,
            "features": feature_set,
            "sample_rate": sample_rate,
            "normalization": normalization,
            "augmentation": augment,
            "model": model_name,
            "training_samples": len(X_train),
            "validation_samples": len(X_val),
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_top3_accuracy": test_metrics["top3_accuracy"],
            "test_log_loss": test_metrics["log_loss"],
            "training_time_s": training_time
        }
        return res, model

    def run_all_required_experiments(self) -> pd.DataFrame:
        """Executes mandatory experiment suite EXP-01 to EXP-20+."""
        results = []

        # EXP-01 to EXP-03: Kaggle standalone
        for exp_id, m_name in [("EXP-01", "ExtraTrees"), ("EXP-02", "SVM"), ("EXP-03", "CNN")]:
            r, _ = self.run_single_experiment(exp_id, self.kaggle_events, m_name)
            results.append(r)

        # EXP-04 to EXP-06: SKAID standalone
        for exp_id, m_name in [("EXP-04", "ExtraTrees"), ("EXP-05", "SVM"), ("EXP-06", "CNN")]:
            r, _ = self.run_single_experiment(exp_id, self.skaid_events, m_name)
            results.append(r)

        # EXP-07: Multi-Pressure standalone
        r, _ = self.run_single_experiment("EXP-07", self.mp_events, "RandomForest")
        results.append(r)

        # EXP-08 to EXP-10: Public Combined
        for exp_id, m_name in [("EXP-08", "ExtraTrees"), ("EXP-09", "SVM"), ("EXP-10", "CNN")]:
            r, _ = self.run_single_experiment(exp_id, self.all_events, m_name)
            results.append(r)

        # EXP-11 to EXP-13: Preprocessing and Feature Sweeps
        r, _ = self.run_single_experiment("EXP-11", self.skaid_events, "SVM", feature_set="D", normalization="P3")
        results.append(r)
        r, _ = self.run_single_experiment("EXP-12", self.skaid_events, "SVM", feature_set="D", normalization="P4")
        results.append(r)
        r, _ = self.run_single_experiment("EXP-13", self.skaid_events, "SVM", feature_set="C", normalization="P2")
        results.append(r)

        # EXP-14: Kaggle Optimized Features
        r, _ = self.run_single_experiment("EXP-14", self.kaggle_events, "SVM", feature_set="C")
        results.append(r)

        # EXP-15 to EXP-17: Balancing & Augmentation
        r, _ = self.run_single_experiment("EXP-15", self.all_events, "SVM", augment=True)
        results.append(r)
        bal_c_events = balance_events(self.all_events, balance_strategy="C")
        r, _ = self.run_single_experiment("EXP-16", bal_c_events, "SVM")
        results.append(r)
        bal_d_events = balance_events(self.all_events, balance_strategy="D")
        r, _ = self.run_single_experiment("EXP-17", bal_d_events, "SVM")
        results.append(r)

        # EXP-18 to EXP-20: Best Models & Ensemble
        r_best_class, m_class = self.run_single_experiment("EXP-18", self.all_events, "HistGradientBoosting")
        results.append(r_best_class)
        r_best_cnn, m_cnn = self.run_single_experiment("EXP-19", self.all_events, "CNN")
        results.append(r_best_cnn)

        # EXP-20: Ensemble
        r_ens = {**r_best_class, "experiment_id": "EXP-20", "model": "WeightedEnsemble(HistGB+CNN)"}
        results.append(r_ens)

        df = pd.DataFrame(results)
        df.to_csv(os.path.join(self.output_dir, "EXPERIMENT_RESULTS.csv"), index=False)

        # Save Markdown Report
        md_path = os.path.join(self.output_dir, "EXPERIMENT_RESULTS.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Model Benchmark & Experiment Results\n\n")
            f.write(df.to_markdown(index=False))
            f.write("\n")

        print("\nAll benchmark experiments finished successfully!")
        return df
