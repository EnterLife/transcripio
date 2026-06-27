from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from transcripio.model_catalog import SHORT_MODEL_REPOS


DEFAULT_WHISPER_DOWNLOADS = (
    "Systran/faster-whisper-tiny",
    "Systran/faster-whisper-base",
    "Systran/faster-whisper-small",
    "Systran/faster-whisper-medium",
    "Systran/faster-whisper-large-v3",
    "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
)


@dataclass(frozen=True, slots=True)
class WhisperDownloadResult:
    repo_id: str
    snapshot_path: Path
    model_path: Path


def resolve_whisper_repo_id(model_name_or_repo_id: str) -> str:
    cleaned = model_name_or_repo_id.strip()
    return SHORT_MODEL_REPOS.get(cleaned, cleaned)


def default_whisper_output_dir(repo_id: str, models_dir: Path = Path("models")) -> Path:
    return models_dir / repo_id.replace("/", "--")


def download_whisper_model(
    model_name_or_repo_id: str,
    output_dir: Path | None = None,
    token: str | None = None,
) -> WhisperDownloadResult:
    repo_id = resolve_whisper_repo_id(model_name_or_repo_id)
    if not repo_id:
        raise RuntimeError("Choose a Whisper model before downloading.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is missing. Run setup.bat first.") from exc

    target_dir = output_dir or default_whisper_output_dir(repo_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target_dir),
                token=token.strip() if token else None,
            )
        )
    except Exception as exc:
        raise RuntimeError(f"Could not download Whisper model {repo_id}: {exc}") from exc

    if not (snapshot_path / "model.bin").exists() or not (snapshot_path / "config.json").exists():
        raise RuntimeError(
            f"Downloaded {repo_id}, but model.bin and config.json were not found under {snapshot_path}."
        )

    return WhisperDownloadResult(
        repo_id=repo_id,
        snapshot_path=snapshot_path,
        model_path=snapshot_path,
    )
