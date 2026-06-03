from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transcripio.diarization_setup import (
    DEFAULT_DIARIZATION_OUTPUT_DIR,
    DEFAULT_DIARIZATION_REPO_ID,
    download_diarization_pipeline,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a gated pyannote diarization pipeline into a local models directory."
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_DIARIZATION_REPO_ID,
        help="Hugging Face repository id for the diarization pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_DIARIZATION_OUTPUT_DIR),
        help="Local directory where the pipeline snapshot should be stored.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to HF_TOKEN from the environment.",
    )
    args = parser.parse_args()

    if not args.token:
        raise SystemExit(
            "Missing Hugging Face token. Set HF_TOKEN or pass --token after accepting the "
            "pyannote model terms on Hugging Face."
        )

    result = download_diarization_pipeline(
        repo_id=args.repo_id,
        output_dir=Path(args.output_dir),
        token=args.token,
    )
    print(f"Pipeline snapshot saved to: {result.snapshot_path}")
    print(f"Use this path in the app: {result.config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
