from __future__ import annotations

import subprocess
from pathlib import Path


AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def is_supported_media(path: Path) -> bool:
    return path.suffix.lower() in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS


def ensure_audio_wav(input_path: Path, work_dir: Path, ffmpeg_path: str = "ffmpeg") -> Path:
    if not input_path.exists():
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    if not is_supported_media(input_path):
        raise ValueError(f"Неподдерживаемый формат файла: {input_path.suffix}")

    output_path = work_dir / f"{input_path.stem}.wav"
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]

    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg не найден. Установите ffmpeg и добавьте его в PATH.") from exc
    except subprocess.CalledProcessError as exc:
        details = exc.stderr.strip() or exc.stdout.strip()
        raise RuntimeError(f"ffmpeg не смог извлечь аудио: {details}") from exc

    if completed.returncode != 0 or not output_path.exists():
        raise RuntimeError("Не удалось подготовить WAV-аудио.")

    return output_path
