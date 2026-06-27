from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from transcripio.models import TranscriptSegment, TranscriptionResult


class StorageError(RuntimeError):
    pass


def result_to_dict(result: TranscriptionResult) -> dict:
    payload = asdict(result)
    payload["source_path"] = str(result.source_path)
    payload["audio_path"] = str(result.audio_path)
    return payload


def result_from_dict(payload: dict) -> TranscriptionResult:
    try:
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
    except (KeyError, TypeError, ValueError) as exc:
        raise StorageError("Transcript history file has an unsupported format.") from exc


def save_result(result: TranscriptionResult, history_dir: Path) -> Path:
    history_dir.mkdir(parents=True, exist_ok=True)
    path = history_dir / f"{result.created_at[:10]}-{result.job_id}.json"
    tmp_path = history_dir / f".{path.name}.{uuid4().hex}.tmp"
    try:
        tmp_path.write_text(
            json.dumps(result_to_dict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def load_result(path: Path) -> TranscriptionResult:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StorageError(f"Transcript history file is invalid JSON: {path.name}") from exc
    except OSError as exc:
        raise StorageError(f"Could not read transcript history file: {path.name}") from exc

    if not isinstance(payload, dict):
        raise StorageError("Transcript history file must contain a JSON object.")
    return result_from_dict(payload)


def list_history(history_dir: Path) -> list[Path]:
    if not history_dir.exists():
        return []
    return sorted(history_dir.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
