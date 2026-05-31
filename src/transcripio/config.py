from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    whisper_model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = "ru"
    diarization_model_path: str | None = None
    ffmpeg_path: str = "ffmpeg"
    output_dir: Path = Path("data/output")
