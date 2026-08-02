"""Validate a private dataset manifest without running model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.dataset_quality import validate_dataset
from src.evaluation_manifest import ManifestValidationError, load_evaluation_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="evaluation manifest JSON")
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    args = parser.parse_args()

    try:
        manifest = load_evaluation_manifest(args.manifest)
        report = validate_dataset(manifest)
    except ManifestValidationError as exc:
        parser.error(str(exc))

    rendered = json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
