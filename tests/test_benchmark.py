from __future__ import annotations

import subprocess
from pathlib import Path

from transcripio.benchmark import (
    compare_transcription_configs,
    find_benchmark_audio,
    run_transcription_benchmark,
)
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


def test_compare_transcription_configs_reuses_one_clip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"wav")
    ffmpeg_calls = 0
    transcribe_calls: list[str] = []

    def fake_run(command, **_kwargs):
        nonlocal ffmpeg_calls
        ffmpeg_calls += 1
        Path(command[-1]).write_bytes(b"clip")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    class FakeTranscriber:
        def __init__(self, config) -> None:
            self._config = config

        def transcribe(self, _clip_path):
            transcribe_calls.append(self._config.whisper_model)
            return [TranscriptSegment(start=0.0, end=2.0, text="hello")], "en", 2.0

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("transcripio.benchmark.WhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr(
        "transcripio.benchmark.time.perf_counter",
        iter([10.0, 11.0, 20.0, 22.0]).__next__,
    )

    results = compare_transcription_configs(
        [
            ("small", AppConfig(whisper_model="small")),
            ("medium", AppConfig(whisper_model="medium")),
        ],
        source,
        tmp_path / "work",
        seconds=2,
    )

    assert ffmpeg_calls == 1
    assert transcribe_calls == ["small", "medium"]
    assert [result.label for result in results] == ["small", "medium"]
    assert [result.realtime_factor for result in results] == [2.0, 1.0]
