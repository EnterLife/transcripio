from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from transcripio.models import TranscriptSegment, TranscriptionResult


def result_to_dict(result: TranscriptionResult) -> dict:
    payload = asdict(result)
    payload["source_path"] = str(result.source_path)
    payload["audio_path"] = str(result.audio_path)
    return payload


def result_from_dict(payload: dict) -> TranscriptionResult:
    return TranscriptionResult(
        job_id=str(payload["job_id"]),
        source_name=str(payload["source_name"]),
        source_path=Path(payload["source_path"]),
        audio_path=Path(payload["audio_path"]),
        language=payload.get("language"),
        duration=payload.get("duration"),
        created_at=str(payload["created_at"]),
        segments=[
            TranscriptSegment(
                start=float(segment["start"]),
                end=float(segment["end"]),
                text=str(segment["text"]),
                speaker=segment.get("speaker"),
            )
            for segment in payload.get("segments", [])
        ],
    )


def save_result(result: TranscriptionResult, history_dir: Path) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{result.created_at[:10]}-{result.job_id}.json"
    path.write_text(json.dumps(result_to_dict(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_result(path: Path) -> TranscriptionResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return result_from_dict(payload)


def list_history(history_dir: Path) -> list[Path]:
    if not history_dir.exists():
        return []
    return sorted(history_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
