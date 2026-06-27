from pathlib import Path

from transcripio.formatters import to_docx, to_srt, to_vtt, to_words_csv
from transcripio.models import TranscriptSegment, TranscriptWord, TranscriptionResult


def test_srt_and_vtt_roll_milliseconds_into_next_second() -> None:
    segments = [TranscriptSegment(start=0.9996, end=1.2, text="hello")]

    assert "00:00:01,000 --> 00:00:01,200" in to_srt(segments)
    assert "00:00:01.000 --> 00:00:01.200" in to_vtt(segments)


def test_docx_export_returns_bytes() -> None:
    result = TranscriptionResult(
        job_id="job",
        source_name="meeting.wav",
        source_path=Path("meeting.wav"),
        audio_path=Path("meeting.prepared.wav"),
        language="en",
        duration=1.2,
        created_at="2026-05-31T00:00:00+00:00",
        segments=[TranscriptSegment(start=0.0, end=1.2, text="hello", speaker="SPEAKER_00")],
    )

    docx = to_docx(result)

    assert docx.startswith(b"PK")


def test_words_csv_exports_word_timestamps() -> None:
    segments = [
        TranscriptSegment(
            start=0.0,
            end=1.0,
            text="hello",
            speaker="A",
            words=[TranscriptWord(start=0.1, end=0.4, text="hello", probability=0.91)],
        )
    ]

    csv_text = to_words_csv(segments)

    assert "segment_start,segment_end,speaker,word_start,word_end,word,probability" in csv_text
    assert "0.000,1.000,A,0.100,0.400,hello,0.9100" in csv_text
