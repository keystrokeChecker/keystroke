import os
import sys

# Ensure root package directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from datasets.audit import DatasetAuditor
from datasets.canonical import verify_classes_json

def main():
    print("=== STARTING KEYSTROKE DATASET AUDIT ===")
    
    classes_json_path = os.path.join(BASE_DIR, "classes.json")
    verify_classes_json(classes_json_path)
    print(f"Verified classes.json at {classes_json_path}")

    # Public datasets directory in project root
    data_dir = os.path.join(os.path.dirname(BASE_DIR), "keystroke_backend", "data")
    output_dir = BASE_DIR

    auditor = DatasetAuditor(data_dir=data_dir, output_dir=output_dir)
    res = auditor.run_audit()

    print("\nAudit Summary:")
    print(f"Total events analyzed across public datasets: {res['all_events_count']}")
    print(f"Reports written to {output_dir}")

if __name__ == "__main__":
    main()
