"""Build fixed-split high resolution training caches from audited original PNGs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.training_data import IMAGE_SIZES, prepare_training_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parents[1] / "data",
                        help="Directory containing audit.json, manifest.jsonl and the original ZIP")
    parser.add_argument("--size", type=int, choices=IMAGE_SIZES, default=224)
    parser.add_argument("--output", type=Path, help="Complete cache destination directory (optional)")
    args = parser.parse_args()
    metadata = prepare_training_cache(args.data, args.size, args.output)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
