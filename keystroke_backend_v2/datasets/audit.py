import os
import csv
import json
import pandas as pd
from typing import Dict, List, Any

from datasets.canonical import CANONICAL_CLASSES, CANONICAL_TO_INDEX, index_to_label, sanitize_raw_label, is_forbidden_recording
from datasets.kaggle import KaggleDatasetLoader
from datasets.multipressure import MultiPressureDatasetLoader
from datasets.skaid import SkaidDatasetLoader


class DatasetAuditor:
    """
    Performs comprehensive dataset auditing, label verification, and statistics generation.
    """

    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        def find_file(filename: str) -> str:
            p1 = os.path.join(data_dir, filename)
            if os.path.exists(p1): return p1
            p2 = os.path.join(os.path.dirname(data_dir), "raw", filename)
            if os.path.exists(p2): return p2
            p3 = os.path.join(data_dir, "raw", filename)
            if os.path.exists(p3): return p3
            return p1

        self.kaggle_path = find_file('archive.zip')
        self.mp_path = find_file('dataset.zip')
        self.mp_ks_path = find_file('Keystrokes_Dataset.zip')
        self.skaid_rec_path = find_file('Participant Recordings_zip.zip')
        self.skaid_log_path = find_file('Keystroke Logs_zip.zip')

    def run_audit(self) -> Dict[str, Any]:
        print("=== RUNNING DATASET AUDIT ===")

        # 1. Audit Kaggle
        kaggle_loader = KaggleDatasetLoader(self.kaggle_path)
        kaggle_events = kaggle_loader.load_events()
        print(f"Kaggle events loaded: {len(kaggle_events)}")

        # 2. Audit Multi-Pressure
        mp_loader = MultiPressureDatasetLoader(self.mp_path, self.mp_ks_path)
        mp_events = mp_loader.load_events()
        print(f"Multi-Pressure events loaded: {len(mp_events)}")

        # 3. Audit SKAID
        skaid_loader = SkaidDatasetLoader(self.skaid_rec_path, self.skaid_log_path)
        skaid_events = skaid_loader.load_events(window_duration_s=0.200)
        print(f"SKAID events loaded: {len(skaid_events)}")

        all_events = kaggle_events + mp_events + skaid_events

        # Check strict 27 class coverage
        found_classes = set(e["canonical_label"] for e in all_events)
        missing_classes = set(CANONICAL_CLASSES) - found_classes
        if missing_classes:
            raise ValueError(f"CRITICAL AUDIT FAILURE: Missing canonical classes: {missing_classes}")

        # Verification of label mapping examples for all 27 classes
        print("\n--- Canonical Label Mapping Examples ---")
        examples_per_class: Dict[str, List[Dict[str, Any]]] = {cls_name: [] for cls_name in CANONICAL_CLASSES}
        for e in all_events:
            c_label = e["canonical_label"]
            if len(examples_per_class[c_label]) < 2:
                examples_per_class[c_label].append(e)

        for c_label in CANONICAL_CLASSES:
            exs = examples_per_class[c_label]
            if not exs:
                raise ValueError(f"CRITICAL AUDIT FAILURE: No examples found for class {c_label}")
            ex = exs[0]
            print(f"Class: {c_label:5s} | Index: {ex['class_index']:2d} | Original: '{ex['original_label']}' | Dataset: {ex['dataset']} | Rec: {ex['recording_id']}")

        # Generate statistics tables
        stats_by_dataset = self._generate_dataset_stats(kaggle_events, mp_events, skaid_events)
        class_dist_df = self._generate_class_distribution(all_events)
        rec_stats_df = self._generate_recording_stats(all_events)

        # Save CSVs
        stats_by_dataset_df = pd.DataFrame(stats_by_dataset)
        stats_by_dataset_df.to_csv(os.path.join(self.output_dir, "dataset_statistics.csv"), index=False)
        class_dist_df.to_csv(os.path.join(self.output_dir, "class_distribution.csv"), index=False)
        rec_stats_df.to_csv(os.path.join(self.output_dir, "recording_statistics.csv"), index=False)

        # Generate DATASET_AUDIT.md
        self._write_markdown_report(stats_by_dataset, class_dist_df)

        print("\nDataset Audit completed successfully!")
        return {
            "all_events_count": len(all_events),
            "stats_by_dataset": stats_by_dataset
        }

    def _generate_dataset_stats(self, kaggle_events: List[Dict], mp_events: List[Dict], skaid_events: List[Dict]) -> List[Dict[str, Any]]:
        results = []
        for name, events in [("Kaggle", kaggle_events), ("Multi-Pressure", mp_events), ("SKAID", skaid_events)]:
            total_events = len(events)
            a_z_cnt = sum(1 for e in events if e["canonical_label"] != "SPACE")
            space_cnt = sum(1 for e in events if e["canonical_label"] == "SPACE")
            srs = list(set(e["sample_rate"] for e in events)) if events else []
            durations = [e["duration_s"] for e in events]
            participants = list(set(e["participant_id"] for e in events))
            recordings = list(set(e["recording_id"] for e in events))

            results.append({
                "Dataset": name,
                "Total Events": total_events,
                "A-Z Events": a_z_cnt,
                "SPACE Events": space_cnt,
                "Sample Rates": str(srs),
                "Participant Count": len(participants),
                "Recording Count": len(recordings),
                "Avg Event Duration (s)": round(float(pd.Series(durations).mean()), 4) if durations else 0.0,
                "Min Event Duration (s)": round(float(pd.Series(durations).min()), 4) if durations else 0.0,
                "Max Event Duration (s)": round(float(pd.Series(durations).max()), 4) if durations else 0.0,
            })
        return results

    def _generate_class_distribution(self, all_events: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(all_events)
        pivot = pd.crosstab(df["canonical_label"], df["dataset"]).reset_index()
        pivot["Total"] = pivot.sum(axis=1, numeric_only=True)
        pivot["Class_Index"] = pivot["canonical_label"].apply(lambda c: CANONICAL_TO_INDEX[c])
        pivot = pivot.sort_values("Class_Index").reset_index(drop=True)
        return pivot

    def _generate_recording_stats(self, all_events: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(all_events)
        rec_df = df.groupby(["dataset", "recording_id", "participant_id"]).agg(
            event_count=("canonical_label", "count"),
            sample_rate=("sample_rate", "first"),
            duration_s=("duration_s", "mean")
        ).reset_index()
        return rec_df

    def _write_markdown_report(self, stats_by_dataset: List[Dict[str, Any]], class_dist_df: pd.DataFrame):
        md_path = os.path.join(self.output_dir, "DATASET_AUDIT.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write("# Public Datasets Audit Report\n\n")
            f.write("Strictly excluding all local HP laptop recordings.\n\n")
            f.write("## Overview by Dataset\n\n")
            f.write("| Dataset | Total Events | A-Z Events | SPACE Events | Sample Rates | Participants | Recordings | Avg Duration (s) |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for r in stats_by_dataset:
                f.write(f"| {r['Dataset']} | {r['Total Events']} | {r['A-Z Events']} | {r['SPACE Events']} | {r['Sample Rates']} | {r['Participant Count']} | {r['Recording Count']} | {r['Avg Event Duration (s)']} |\n")
            
            f.write("\n## 27-Class Canonical Distribution\n\n")
            f.write(class_dist_df.to_markdown(index=False))
            f.write("\n")
