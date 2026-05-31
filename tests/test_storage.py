from pathlib import Path

from transcripio.models import TranscriptSegment, TranscriptionResult
from transcripio.storage import load_result, save_result


def test_save_and_load_result_round_trip(tmp_path: Path) -> None:
    result = TranscriptionResult(
        job_id="abc123",
        source_name="call.mp4",
        source_path=Path("call.mp4"),
        audio_path=Path("call.prepared.wav"),
        language="en",
        duration=2.0,
        created_at="2026-05-31T00:00:00+00:00",
        segments=[TranscriptSegment(start=0.0, end=2.0, text="hello", speaker=None)],
    )

    path = save_result(result, tmp_path)
    loaded = load_result(path)

    assert loaded.job_id == "abc123"
    assert loaded.source_name == "call.mp4"
    assert loaded.segments[0].text == "hello"
