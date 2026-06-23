from __future__ import annotations

from io import BytesIO
from pathlib import Path

from app import _extracted_audio_download, _save_uploaded_media
from transcripio.models import TranscriptionResult


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
