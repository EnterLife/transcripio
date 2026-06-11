from __future__ import annotations

from io import BytesIO

from app import _save_uploaded_media


class StreamOnlyUpload(BytesIO):
    def getbuffer(self):  # pragma: no cover - should never be called by the helper.
        raise AssertionError("large uploads must be copied as a stream")


def test_save_uploaded_media_copies_stream_without_buffering(tmp_path) -> None:
    uploaded_file = StreamOnlyUpload(b"first chunk second chunk")
    destination = tmp_path / "large-video.mp4"

    _save_uploaded_media(uploaded_file, destination)

    assert destination.read_bytes() == b"first chunk second chunk"
    assert uploaded_file.tell() == 0
