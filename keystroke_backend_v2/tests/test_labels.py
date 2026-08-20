import pytest
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from datasets.canonical import (
    CANONICAL_CLASSES, CANONICAL_TO_INDEX, INDEX_TO_CANONICAL,
    sanitize_raw_label, label_to_index, index_to_label, verify_classes_json, is_forbidden_recording
)


def test_27_canonical_classes():
    assert len(CANONICAL_CLASSES) == 27
    assert CANONICAL_CLASSES[0] == "A"
    assert CANONICAL_CLASSES[25] == "Z"
    assert CANONICAL_CLASSES[26] == "SPACE"


def test_label_sanitization():
    assert sanitize_raw_label("a") == "A"
    assert sanitize_raw_label("A") == "A"
    assert sanitize_raw_label("Key-A-H") == "A"
    assert sanitize_raw_label("space") == "SPACE"
    assert sanitize_raw_label("Spacebar") == "SPACE"
    assert sanitize_raw_label("Key.space") == "SPACE"
    assert sanitize_raw_label("0") is None
    assert sanitize_raw_label("Enter") is None
    assert sanitize_raw_label("shift") is None


def test_forbidden_recording_exclusion():
    assert is_forbidden_recording("session1") is True
    assert is_forbidden_recording("C:/data/new2/recording.wav") is True
    assert is_forbidden_recording("session_calibrated_test.wav") is True
    assert is_forbidden_recording("Participant Recordings/qt_1746047498719.m4a") is False


def test_classes_json():
    classes_json_path = os.path.join(BASE_DIR, "classes.json")
    assert verify_classes_json(classes_json_path) is True
