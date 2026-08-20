import pytest
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from datasets.kaggle import KaggleDatasetLoader
from datasets.multipressure import MultiPressureDatasetLoader
from datasets.canonical import CANONICAL_CLASSES


def test_kaggle_loader_structure():
    data_dir = os.path.join(os.path.dirname(BASE_DIR), "keystroke_backend", "data")
    kaggle_path = os.path.join(data_dir, "archive.zip")
    if os.path.exists(kaggle_path):
        loader = KaggleDatasetLoader(kaggle_path)
        events = loader.load_events(max_samples_per_class=5)
        assert len(events) > 0
        for ev in events:
            assert ev["canonical_label"] in CANONICAL_CLASSES
            assert 0 <= ev["class_index"] <= 26
            assert ev["dataset"] == "Kaggle"


def test_multipressure_loader_structure():
    data_dir = os.path.join(os.path.dirname(BASE_DIR), "keystroke_backend", "data")
    mp_path = os.path.join(data_dir, "dataset.zip")
    if os.path.exists(mp_path):
        loader = MultiPressureDatasetLoader(mp_path)
        events = loader.load_events(max_samples_per_class=5)
        assert len(events) > 0
        for ev in events:
            assert ev["canonical_label"] in CANONICAL_CLASSES
            assert 0 <= ev["class_index"] <= 26
            assert ev["dataset"] == "MultiPressure"
            assert ev["pressure"] in ["High", "Medium", "Low", "Unknown"]
