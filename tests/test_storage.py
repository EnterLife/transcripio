import os
from pathlib import Path

import pytest

from transcripio.models import TranscriptSegment, TranscriptWord, TranscriptionResult
from transcripio.storage import StorageError, list_history, load_result, save_result


def test_save_and_load_result_round_trip(tmp_path: Path) -> None:
    result = TranscriptionResult(
        job_id="abc123",
        source_name="call.mp4",
        source_path=Path("call.mp4"),
        audio_path=Path("call.prepared.wav"),
        language="en",
        duration=2.0,
        created_at="2026-05-31T00:00:00+00:00",
        segments=[
            TranscriptSegment(
                start=0.0,
                end=2.0,
                text="hello",
                speaker=None,
                words=[TranscriptWord(start=0.0, end=0.5, text="hello", probability=0.95)],
            )
        ],
    )

    path = save_result(result, tmp_path)
    loaded = load_result(path)

    assert loaded.job_id == "abc123"
    assert loaded.source_name == "call.mp4"
    assert loaded.segments[0].text == "hello"
    assert loaded.segments[0].words[0].text == "hello"
    assert loaded.segments[0].words[0].probability == 0.95


def test_load_result_reports_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(StorageError, match="invalid JSON"):
        load_result(path)


def test_load_result_reports_unsupported_payload(tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("[]", encoding="utf-8")

    with pytest.raises(StorageError, match="JSON object"):
        load_result(path)


def test_list_history_skips_files_that_disappear_during_listing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    older = tmp_path / "older.json"
    newer = tmp_path / "newer.json"
    disappearing = tmp_path / "disappearing.json"
    older.write_text("{}", encoding="utf-8")
    newer.write_text("{}", encoding="utf-8")
    disappearing.write_text("{}", encoding="utf-8")
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))
    os.utime(disappearing, (3, 3))

    original_stat = Path.stat

    def stat_or_disappear(path: Path, *args, **kwargs):
        if path.name == disappearing.name:
            raise FileNotFoundError(path)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat_or_disappear)

    assert list_history(tmp_path) == [newer, older]
