"""Lock one production method using validation evidence only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.model_selection import ModelSelectionError, load_and_select


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_result", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        selection = load_and_select(args.validation_result)
    except ModelSelectionError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Selected {selection['selected_method']}; lock written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
