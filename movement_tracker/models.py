"""Explicit CLI model download; application itself works offline."""
import argparse
from pathlib import Path
import urllib.request
import zipfile

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_{variant}/float16/1/pose_landmarker_{variant}.task"
DEFAULT_MODEL = Path(__file__).resolve().parent.parent / "models" / "pose_landmarker_full.task"

def main():
    parser = argparse.ArgumentParser(description="Download an official MediaPipe pose model")
    parser.add_argument("--variant", choices=("lite", "full", "heavy"), default="full")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    target = args.output or DEFAULT_MODEL.with_name(f"pose_landmarker_{args.variant}.task")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".download")
    try:
        with urllib.request.urlopen(MODEL_URL.format(variant=args.variant), timeout=60) as response:
            with partial.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
        if not zipfile.is_zipfile(partial):
            raise RuntimeError("Downloaded model is not a valid MediaPipe task archive")
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    print(target)

if __name__ == "__main__":
    main()
