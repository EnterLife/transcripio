from __future__ import annotations

import json
from dataclasses import asdict

from transcripio.models import TranscriptSegment, TranscriptionResult


def to_txt(segments: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for segment in segments:
        prefix = f"{segment.speaker}: " if segment.speaker else ""
        lines.append(f"{_format_time(segment.start)} {_format_time(segment.end)}  {prefix}{segment.text}")
    return "\n".join(lines)


def to_srt(segments: list[TranscriptSegment]) -> str:
    blocks: list[str] = []
    for index, segment in enumerate(segments, start=1):
        speaker = f"{segment.speaker}: " if segment.speaker else ""
        blocks.append(
            "\n".join(
                [
                    str(index),
                    f"{_format_srt_time(segment.start)} --> {_format_srt_time(segment.end)}",
                    f"{speaker}{segment.text}",
                ]
            )
        )
    return "\n\n".join(blocks)


def to_json(result: TranscriptionResult) -> str:
    payload = asdict(result)
    payload["source_path"] = str(result.source_path)
    payload["audio_path"] = str(result.audio_path)
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _format_time(seconds: float) -> str:
    whole = int(seconds)
    minutes, sec = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"[{hours:02d}:{minutes:02d}:{sec:02d}]"


def _format_srt_time(seconds: float) -> str:
    whole = int(seconds)
    millis = int(round((seconds - whole) * 1000))
    minutes, sec = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d},{millis:03d}"
