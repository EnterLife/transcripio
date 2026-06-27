from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from transcripio.media import ensure_audio_wav, is_supported_media


def test_is_supported_media_accepts_audio_and_video_extensions() -> None:
    assert is_supported_media(Path("call.MP3")) is True
    assert is_supported_media(Path("meeting.mp4")) is True
    assert is_supported_media(Path("notes.txt")) is False


def test_ensure_audio_wav_creates_mono_16khz_wav(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "meeting.mp4"
    input_path.write_bytes(b"video")
    work_dir = tmp_path / "prepared"
    captured_command: list[str] = []

    def fake_run(command, **kwargs):
        captured_command.extend(command)
        Path(command[-1]).write_bytes(b"wav")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    output_path = ensure_audio_wav(input_path, work_dir, ffmpeg_path="ffmpeg-test")

    assert output_path == work_dir / "meeting.wav"
    assert output_path.read_bytes() == b"wav"
    assert captured_command == [
        "ffmpeg-test",
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


def test_ensure_audio_wav_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="File was not found"):
        ensure_audio_wav(tmp_path / "missing.mp3", tmp_path)


def test_ensure_audio_wav_rejects_unsupported_format(tmp_path: Path) -> None:
    input_path = tmp_path / "notes.txt"
    input_path.write_text("not media", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format"):
        ensure_audio_wav(input_path, tmp_path)


def test_ensure_audio_wav_reports_ffmpeg_stderr(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "broken.mp3"
    input_path.write_bytes(b"media")

    def fake_run(command, **_kwargs):
        raise subprocess.CalledProcessError(
            returncode=1,
            cmd=command,
            stderr="Invalid data found when processing input",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="Invalid data found"):
        ensure_audio_wav(input_path, tmp_path)
