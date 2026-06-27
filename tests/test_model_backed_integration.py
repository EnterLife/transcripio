from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from transcripio.config import AppConfig
from transcripio.pipeline import TranscriptionPipeline


def test_model_backed_pipeline_with_local_assets(tmp_path: Path) -> None:
    if os.environ.get("TRANSCRIPIO_RUN_MODEL_TESTS") != "1":
        pytest.skip("set TRANSCRIPIO_RUN_MODEL_TESTS=1 to run local model-backed checks")

    source_audio = _model_test_audio_path()
    whisper_model = os.environ.get("TRANSCRIPIO_WHISPER_MODEL", "small")
    diarization_model = os.environ.get(
        "TRANSCRIPIO_DIARIZATION_MODEL",
        "models/pyannote-speaker-diarization/config.yaml",
    )

    clip_path = tmp_path / "clip.wav"
    completed = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(source_audio),
            "-t",
            "8",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(clip_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    config = AppConfig(
        whisper_model=whisper_model,
        device="cpu",
        compute_type="int8",
        local_files_only=True,
        diarization_model_path=diarization_model if Path(diarization_model).exists() else None,
        output_dir=tmp_path / "output",
        history_dir=tmp_path / "history",
        vad_filter=False,
        beam_size=1,
        best_of=1,
    )

    result = TranscriptionPipeline(config).run(clip_path)

    assert result.duration is not None
    assert result.audio_path.exists()
    assert result.segments
    if config.diarization_model_path:
        assert any(segment.speaker for segment in result.segments)


def _model_test_audio_path() -> Path:
    configured_path = os.environ.get("TRANSCRIPIO_MODEL_TEST_AUDIO")
    if configured_path:
        path = Path(configured_path)
        if path.exists():
            return path
        pytest.skip(f"TRANSCRIPIO_MODEL_TEST_AUDIO does not exist: {path}")

    candidates = sorted(Path("data/output").glob("*.wav"), key=lambda path: path.stat().st_size)
    if not candidates:
        pytest.skip("no local WAV files found under data/output")
    return candidates[0]
