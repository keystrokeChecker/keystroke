import json
import os
from typing import Dict, Optional, List, Tuple

FORBIDDEN_PATTERNS: List[str] = [
    "session1",
    "session4",
    "new1",
    "new2",
    "session_calibrated_test",
    "gain_check",
    "gain_test",
    "session3"
]

CANONICAL_CLASSES: List[str] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
    "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "SPACE"
]

CANONICAL_TO_INDEX: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(CANONICAL_CLASSES)}
INDEX_TO_CANONICAL: Dict[int, str] = {i: cls_name for i, cls_name in enumerate(CANONICAL_CLASSES)}


def is_forbidden_recording(filepath_or_name: str) -> bool:
    """
    Checks whether a file or session name contains any forbidden local HP recording identifier.
    If forbidden, prints notification and returns True.
    """
    lower_path = str(filepath_or_name).lower()
    for pattern in FORBIDDEN_PATTERNS:
        if pattern.lower() in lower_path:
            print(f"[EXCLUSION GUARD] LOCAL HP RECORDING EXCLUDED FROM TRAINING: {filepath_or_name} (matched '{pattern}')")
            return True
    return False


def sanitize_raw_label(raw_label: str) -> Optional[str]:
    """
    Maps any dataset-specific raw string label to one of the 27 canonical classes,
    or returns None if it is not an A-Z or SPACE key.
    """
    if not raw_label:
        return None

    cleaned = str(raw_label).strip()

    # Handle common space variants
    if cleaned.lower() in ["space", "spacebar", "key.space", " ", "space_key"]:
        return "SPACE"

    # Strip prefixes like "Key-" or "Key." or "key_"
    if cleaned.lower().startswith("key."):
        cleaned = cleaned[4:]
    elif cleaned.lower().startswith("key-"):
        cleaned = cleaned[4:]
    elif cleaned.lower().startswith("key_"):
        cleaned = cleaned[4:]

    # Remove pressure suffix if present (e.g. 'Key-A-H' or 'A-H' or 'A-M')
    if len(cleaned) == 3 and cleaned[1] == '-' and cleaned[2].upper() in ['H', 'M', 'L']:
        cleaned = cleaned[0]

    cleaned_upper = cleaned.upper()

    if cleaned_upper in CANONICAL_TO_INDEX:
        return cleaned_upper

    return None


def label_to_index(canonical_label: str) -> int:
    """Returns integer index 0-26 for a valid canonical label."""
    if canonical_label not in CANONICAL_TO_INDEX:
        raise ValueError(f"Unknown canonical label '{canonical_label}'. Must be one of {CANONICAL_CLASSES}")
    return CANONICAL_TO_INDEX[canonical_label]


def index_to_label(index: int) -> str:
    """Returns canonical label for integer index 0-26."""
    if index not in INDEX_TO_CANONICAL:
        raise ValueError(f"Invalid class index '{index}'. Must be 0-26.")
    return INDEX_TO_CANONICAL[index]


def verify_classes_json(classes_json_path: str) -> bool:
    """Verifies that classes.json exists and strictly matches CANONICAL_CLASSES."""
    if not os.path.exists(classes_json_path):
        raise FileNotFoundError(f"classes.json not found at {classes_json_path}")

    with open(classes_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if len(data) != 27:
        raise ValueError(f"classes.json must contain exactly 27 classes, got {len(data)}")

    for idx_str, label in data.items():
        idx = int(idx_str)
        if INDEX_TO_CANONICAL.get(idx) != label:
            raise ValueError(f"Mismatch in classes.json at index {idx}: expected {INDEX_TO_CANONICAL.get(idx)}, got {label}")

    return True
