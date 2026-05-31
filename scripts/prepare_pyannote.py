from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download a gated pyannote diarization pipeline into a local models directory."
    )
    parser.add_argument(
        "--repo-id",
        default="pyannote/speaker-diarization-3.1",
        help="Hugging Face repository id for the diarization pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default="models/pyannote-speaker-diarization",
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

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is missing. Run .\\setup.ps1 first.") from exc

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshot_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(output_dir),
        token=args.token,
    )

    config_path = Path(snapshot_path) / "config.yaml"
    print(f"Pipeline snapshot saved to: {snapshot_path}")
    if config_path.exists():
        print(f"Use this path in the app: {config_path}")
    else:
        print("Download completed, but config.yaml was not found. Check the repository contents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
