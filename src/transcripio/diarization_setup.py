from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DIARIZATION_REPO_ID = "pyannote/speaker-diarization-3.1"
DEFAULT_DIARIZATION_OUTPUT_DIR = Path("models/pyannote-speaker-diarization")


@dataclass(frozen=True, slots=True)
class DiarizationDownloadResult:
    snapshot_path: Path
    config_path: Path


def download_diarization_pipeline(
    repo_id: str,
    output_dir: Path,
    token: str,
) -> DiarizationDownloadResult:
    cleaned_repo_id = repo_id.strip() or DEFAULT_DIARIZATION_REPO_ID
    cleaned_token = token.strip()
    if not cleaned_token:
        raise RuntimeError(
            "Missing Hugging Face token. Enter a token after accepting the pyannote model terms."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is missing. Run setup.bat first.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(
        snapshot_download(
            repo_id=cleaned_repo_id,
            local_dir=str(output_dir),
            token=cleaned_token,
        )
    )
    config_path = snapshot_path / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(
            f"Downloaded {cleaned_repo_id}, but config.yaml was not found under {snapshot_path}."
        )

    return DiarizationDownloadResult(snapshot_path=snapshot_path, config_path=config_path)
