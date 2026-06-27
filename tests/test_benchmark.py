from __future__ import annotations

import subprocess
from pathlib import Path

from transcripio.benchmark import find_benchmark_audio, run_transcription_benchmark
from transcripio.config import AppConfig
from transcripio.models import TranscriptSegment


def test_find_benchmark_audio_returns_smallest_wav(tmp_path: Path) -> None:
    large = tmp_path / "large.wav"
    small = tmp_path / "small.wav"
    large.write_bytes(b"x" * 10)
    small.write_bytes(b"x")

    assert find_benchmark_audio(tmp_path) == small


def test_run_transcription_benchmark_reports_realtime_factor(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"wav")

    def fake_run(command, **_kwargs):
        Path(command[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    class FakeTranscriber:
        def __init__(self, _config) -> None:
            pass

        def transcribe(self, _clip_path):
            return [TranscriptSegment(start=0.0, end=2.0, text="hello")], "en", 2.0

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("transcripio.benchmark.WhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr("transcripio.benchmark.time.perf_counter", iter([10.0, 11.0]).__next__)

    result = run_transcription_benchmark(AppConfig(), source, tmp_path / "work", seconds=2)

    assert result.segment_count == 1
    assert result.language == "en"
    assert result.audio_duration == 2.0
    assert result.elapsed_seconds == 1.0
    assert result.realtime_factor == 2.0
