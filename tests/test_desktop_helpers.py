from __future__ import annotations

from pathlib import Path

import pytest

from transcripio.config import AppConfig
from transcripio.health import HealthCheck
from transcripio.models import TranscriptSegment, TranscriptionResult
from transcripio_desktop.helpers import (
    blocking_health_message,
    build_desktop_config,
    export_result,
    format_duration,
    format_file_size,
    health_log_lines,
    result_metrics,
    result_title,
    safe_file_stem,
)


def test_build_desktop_config_preserves_backend_defaults() -> None:
    base = AppConfig(
        whisper_model="small",
        device="cpu",
        compute_type="int8",
        output_dir=Path("data/output"),
        history_dir=Path("data/history"),
        allow_cpu_fallback=True,
        auto_install_cuda_runtime=True,
    )

    config = build_desktop_config(
        base,
        whisper_model="models/whisper-small",
        device="cuda",
        compute_type="float16",
        language="en",
        diarization_model_path="models/pyannote/config.yaml",
        ffmpeg_path="bin/ffmpeg.exe",
        output_dir=Path("out"),
        history_dir=Path("history"),
        local_files_only=True,
        vad_filter=False,
        word_timestamps=True,
        beam_size=3,
        best_of=2,
        cpu_threads=4,
        num_workers=2,
        initial_prompt="Names: Transcripio",
        hotwords="Transcripio",
    )

    assert config.whisper_model == "models/whisper-small"
    assert config.device == "cuda"
    assert config.compute_type == "float16"
    assert config.language == "en"
    assert config.diarization_model_path == "models/pyannote/config.yaml"
    assert config.ffmpeg_path == "bin/ffmpeg.exe"
    assert config.output_dir == Path("out")
    assert config.history_dir == Path("history")
    assert config.local_files_only is True
    assert config.vad_filter is False
    assert config.word_timestamps is True
    assert config.initial_prompt == "Names: Transcripio"
    assert config.hotwords == "Transcripio"
    assert config.allow_cpu_fallback is True


def test_desktop_format_helpers_are_readable() -> None:
    assert format_duration(None) == "Unknown"
    assert format_duration(65.4) == "1:05"
    assert format_duration(3661) == "1:01:01"
    assert format_file_size(500) == "500 B"
    assert format_file_size(2048) == "2.0 KB"
    assert safe_file_stem('bad:name?.mp3') == "bad_name_.mp3"


def test_blocking_health_message_reports_errors_only() -> None:
    checks = [
        HealthCheck("ffmpeg", "ok", "ffmpeg is available."),
        HealthCheck("Whisper model", "warning", "May download on first use."),
        HealthCheck("Network proxy", "error", "Unsupported proxy scheme."),
    ]

    assert blocking_health_message(checks) == (
        "Fix these issues before processing:\n"
        "- Network proxy: Unsupported proxy scheme."
    )
    assert health_log_lines(checks) == [
        "  ffmpeg: ok - ffmpeg is available.",
        "  Whisper model: warning - May download on first use.",
        "  Network proxy: error - Unsupported proxy scheme.",
    ]


def test_blocking_health_message_allows_warnings() -> None:
    checks = [HealthCheck("Whisper model", "warning", "May download on first use.")]

    assert blocking_health_message(checks) is None


def test_result_metrics_count_distinct_speakers(tmp_path: Path) -> None:
    result = _result(tmp_path)

    assert result_title(result) == "meeting.mp3 | 2026-06-23 12:00:00 | job-1234"
    assert result_metrics(result) == {
        "Language": "en",
        "Duration": "0:03",
        "Segments": "2",
        "Speakers": "2",
        "Created": "2026-06-23 12:00:00",
    }


def test_export_result_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported export format"):
        export_result(_result(tmp_path), "PDF")


def test_export_result_creates_text_artifact(tmp_path: Path) -> None:
    artifact = export_result(_result(tmp_path), "TXT")

    assert artifact.file_name == "meeting.txt"
    assert artifact.mime == "text/plain"
    assert b"SPEAKER_00: Hello" in artifact.data


def _result(tmp_path: Path) -> TranscriptionResult:
    return TranscriptionResult(
        job_id="job-123456",
        source_name="meeting.mp3",
        source_path=tmp_path / "meeting.mp3",
        audio_path=tmp_path / "meeting.wav",
        language="en",
        duration=3.2,
        created_at="2026-06-23T12:00:00+00:00",
        segments=[
            TranscriptSegment(start=0.0, end=1.2, speaker="SPEAKER_00", text="Hello"),
            TranscriptSegment(start=1.2, end=3.2, speaker="SPEAKER_01", text="Hi"),
        ],
    )
