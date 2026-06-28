from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from transcripio.models import TranscriptSegment, TranscriptWord, TranscriptionResult


class StorageError(RuntimeError):
    pass


def result_to_dict(result: TranscriptionResult) -> dict:
    payload = asdict(result)
    payload["source_path"] = str(result.source_path)
    payload["audio_path"] = str(result.audio_path)
    return payload


def result_from_dict(payload: dict) -> TranscriptionResult:
    try:
        segments_payload = payload.get("segments", [])
        if not isinstance(segments_payload, list):
            raise TypeError("segments must be a list")

        return TranscriptionResult(
            job_id=str(payload["job_id"]),
            source_name=str(payload["source_name"]),
            source_path=Path(payload["source_path"]),
            audio_path=Path(payload["audio_path"]),
            language=_optional_text(payload.get("language")),
            duration=_optional_float(payload.get("duration")),
            created_at=str(payload["created_at"]),
            segments=[
                _transcript_segment_from_history(segment)
                for segment in segments_payload
            ],
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise StorageError("Transcript history file has an unsupported format.") from exc


def _transcript_segment_from_history(value: object) -> TranscriptSegment:
    if not isinstance(value, dict):
        raise TypeError("history segment must be an object")

    words_payload = value.get("words", [])
    if not isinstance(words_payload, list):
        raise TypeError("history segment words must be a list")

    return TranscriptSegment(
        start=float(value["start"]),
        end=float(value["end"]),
        text=str(value["text"]),
        speaker=_optional_text(value.get("speaker")),
        words=[_transcript_word_from_history(word) for word in words_payload],
    )


def _transcript_word_from_history(value: object) -> TranscriptWord:
    if not isinstance(value, dict):
        raise TypeError("history word must be an object")

    return TranscriptWord(
        start=float(value["start"]),
        end=float(value["end"]),
        text=str(value["text"]),
        probability=_optional_float(value.get("probability")),
    )


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def save_result(result: TranscriptionResult, history_dir: Path) -> Path:
    try:
        history_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise StorageError(f"Could not create transcript history directory: {history_dir}") from exc

    path = history_dir / f"{result.created_at[:10]}-{result.job_id}.json"
    tmp_path = history_dir / f".{path.name}.{uuid4().hex}.tmp"
    try:
        tmp_path.write_text(
            json.dumps(result_to_dict(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(path)
    except OSError as exc:
        raise StorageError(f"Could not save transcript history file: {path.name}") from exc
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

    history_files: list[tuple[float, Path]] = []
    for path in history_dir.glob("*.json"):
        try:
            modified_at = path.stat().st_mtime
        except OSError:
            continue
        history_files.append((modified_at, path))

    history_files.sort(key=lambda item: item[0], reverse=True)
    return [path for _modified_at, path in history_files]
