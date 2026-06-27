from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from transcripio.config import AppConfig
from transcripio.transcriber import WhisperTranscriber


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    source_path: Path
    clip_path: Path
    elapsed_seconds: float
    audio_duration: float | None
    realtime_factor: float | None
    language: str | None
    segment_count: int


def find_benchmark_audio(output_dir: Path = Path("data/output")) -> Path | None:
    if not output_dir.exists():
        return None
    candidates = sorted(output_dir.glob("*.wav"), key=lambda path: path.stat().st_size)
    return candidates[0] if candidates else None


def run_transcription_benchmark(
    config: AppConfig,
    source_path: Path,
    work_dir: Path,
    seconds: int = 15,
) -> BenchmarkResult:
    work_dir.mkdir(parents=True, exist_ok=True)
    clip_path = work_dir / f"{source_path.stem}.benchmark.wav"
    _create_benchmark_clip(source_path, clip_path, seconds, config.ffmpeg_path)

    transcriber = WhisperTranscriber(config)
    started_at = time.perf_counter()
    segments, language, duration = transcriber.transcribe(clip_path)
    elapsed = time.perf_counter() - started_at
    realtime_factor = (duration / elapsed) if duration and elapsed > 0 else None
    return BenchmarkResult(
        source_path=source_path,
        clip_path=clip_path,
        elapsed_seconds=elapsed,
        audio_duration=duration,
        realtime_factor=realtime_factor,
        language=language,
        segment_count=len(segments),
    )


def _create_benchmark_clip(
    source_path: Path,
    clip_path: Path,
    seconds: int,
    ffmpeg_path: str,
) -> None:
    completed = subprocess.run(
        [
            ffmpeg_path,
            "-y",
            "-i",
            str(source_path),
            "-t",
            str(seconds),
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
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeError(f"ffmpeg failed to create benchmark clip: {details}")
