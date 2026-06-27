from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TranscriptWord:
    start: float
    end: float
    text: str
    probability: float | None = None


@dataclass(slots=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    speaker: str | None = None
    words: list[TranscriptWord] = field(default_factory=list)


@dataclass(slots=True)
class DiarizationSegment:
    start: float
    end: float
    speaker: str


@dataclass(slots=True)
class TranscriptionResult:
    job_id: str
    source_name: str
    source_path: Path
    audio_path: Path
    language: str | None
    duration: float | None
    created_at: str
    segments: list[TranscriptSegment] = field(default_factory=list)
