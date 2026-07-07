from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

from app import (
    QUEUE_SELECTED_RESULT_KEY,
    _completed_result_label,
    _extracted_audio_download,
    _format_duration_label,
    _format_file_size,
    _remember_completed_result_selection,
    _save_uploaded_media,
    _save_result_to_history,
    _speaker_count_label,
    _uploaded_file_rows,
)
from transcripio.models import TranscriptSegment, TranscriptionResult
from transcripio.storage import StorageError


class StreamOnlyUpload(BytesIO):
    def getbuffer(self):  # pragma: no cover - should never be called by the helper.
        raise AssertionError("large uploads must be copied as a stream")


def test_save_uploaded_media_copies_stream_without_buffering(tmp_path) -> None:
    uploaded_file = StreamOnlyUpload(b"first chunk second chunk")
    destination = tmp_path / "large-video.mp4"

    _save_uploaded_media(uploaded_file, destination)

    assert destination.read_bytes() == b"first chunk second chunk"
    assert uploaded_file.tell() == 0


def test_extracted_audio_download_is_available_for_video_source(tmp_path: Path) -> None:
    audio_path = tmp_path / "meeting.prepared.wav"
    audio_path.write_bytes(b"prepared wav")

    result = TranscriptionResult(
        job_id="job-1",
        source_name="meeting.mp4",
        source_path=tmp_path / "meeting.mp4",
        audio_path=audio_path,
        language="en",
        duration=1.0,
        created_at="2026-06-23T12:00:00",
    )

    download = _extracted_audio_download(result)

    assert download is not None
    assert download.file_name == "meeting.wav"
    assert download.data == b"prepared wav"


def test_extracted_audio_download_is_skipped_for_audio_source(tmp_path: Path) -> None:
    audio_path = tmp_path / "meeting.prepared.wav"
    audio_path.write_bytes(b"prepared wav")

    result = TranscriptionResult(
        job_id="job-1",
        source_name="meeting.mp3",
        source_path=tmp_path / "meeting.mp3",
        audio_path=audio_path,
        language="en",
        duration=1.0,
        created_at="2026-06-23T12:00:00",
    )

    assert _extracted_audio_download(result) is None


def test_format_file_size_uses_readable_units() -> None:
    assert _format_file_size(12) == "12 B"
    assert _format_file_size(1536) == "1.5 KB"
    assert _format_file_size(2 * 1024 * 1024) == "2.0 MB"


def test_format_file_size_rejects_negative_values() -> None:
    with pytest.raises(ValueError, match="zero or greater"):
        _format_file_size(-1)


def test_format_duration_label_uses_clock_format() -> None:
    assert _format_duration_label(None) == "Unknown"
    assert _format_duration_label(65.2) == "1:05"
    assert _format_duration_label(3661.0) == "1:01:01"


def test_speaker_count_label_counts_distinct_speakers(tmp_path: Path) -> None:
    result = TranscriptionResult(
        job_id="job-1",
        source_name="meeting.mp3",
        source_path=tmp_path / "meeting.mp3",
        audio_path=tmp_path / "meeting.prepared.wav",
        language="en",
        duration=1.0,
        created_at="2026-06-23T12:00:00",
        segments=[
            TranscriptSegment(start=0.0, end=1.0, speaker="SPEAKER_00", text="Hi"),
            TranscriptSegment(start=1.0, end=2.0, speaker="SPEAKER_01", text="Hello"),
            TranscriptSegment(start=2.0, end=3.0, speaker="SPEAKER_00", text="Again"),
        ],
    )

    assert _speaker_count_label(result) == "2"


def test_uploaded_file_rows_format_sizes() -> None:
    class UploadedFile:
        name = "meeting.wav"
        size = 2048

    assert _uploaded_file_rows([UploadedFile()]) == [{"File": "meeting.wav", "Size": "2.0 KB"}]


def test_completed_result_selection_switches_to_new_result(tmp_path: Path) -> None:
    session_state = {
        QUEUE_SELECTED_RESULT_KEY: "old.mp3 | 2026-06-22T10:00 | old-job"
    }
    result = TranscriptionResult(
        job_id="new-job-123456",
        source_name="new.mp3",
        source_path=tmp_path / "new.mp3",
        audio_path=tmp_path / "new.prepared.wav",
        language="en",
        duration=1.0,
        created_at="2026-06-23T12:00:00+00:00",
    )

    _remember_completed_result_selection(session_state, result)

    assert session_state[QUEUE_SELECTED_RESULT_KEY] == _completed_result_label(result)


def test_save_result_to_history_warns_without_losing_session_result(
    tmp_path: Path,
    monkeypatch,
) -> None:
    warnings: list[str] = []
    result = TranscriptionResult(
        job_id="job-1",
        source_name="meeting.mp3",
        source_path=tmp_path / "meeting.mp3",
        audio_path=tmp_path / "meeting.prepared.wav",
        language="en",
        duration=1.0,
        created_at="2026-06-23T12:00:00",
    )

    def fail_save_result(_result, _history_dir):
        raise StorageError("disk is read-only")

    monkeypatch.setattr("app.save_result", fail_save_result)
    monkeypatch.setattr("app.st.warning", warnings.append)

    assert _save_result_to_history(result, tmp_path) is False
    assert warnings == [
        "Transcript is available in this session, but history was not saved: disk is read-only"
    ]
