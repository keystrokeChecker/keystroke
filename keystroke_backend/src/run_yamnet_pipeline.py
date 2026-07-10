import argparse
import subprocess
import sys
from pathlib import Path

try:
    from .yamnet_config import HOP_SECONDS, WINDOW_SECONDS
except ImportError:
    from yamnet_config import HOP_SECONDS, WINDOW_SECONDS


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--names", nargs="*", help="Optional session names, e.g. session1 session2")
    parser.add_argument("--window-seconds", type=float, default=WINDOW_SECONDS)
    parser.add_argument("--hop-seconds", type=float, default=HOP_SECONDS)
    args = parser.parse_args()

    build_cmd = [
        sys.executable,
        "-m",
        "src.dataset_builder",
        "--window-seconds",
        str(args.window_seconds),
        "--hop-seconds",
        str(args.hop_seconds),
    ]
    if args.names:
        build_cmd.extend(["--names", *args.names])

    subprocess.run(build_cmd, cwd=ROOT, check=True)
    subprocess.run([sys.executable, "-m", "src.train"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
