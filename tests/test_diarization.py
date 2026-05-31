from transcripio.diarization import assign_speakers
from transcripio.models import DiarizationSegment, TranscriptSegment


def test_assign_speakers_uses_largest_overlap() -> None:
    transcript = [TranscriptSegment(start=0.0, end=10.0, text="hello")]
    diarization = [
        DiarizationSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        DiarizationSegment(start=2.0, end=10.0, speaker="SPEAKER_01"),
    ]

    result = assign_speakers(transcript, diarization)

    assert result[0].speaker == "SPEAKER_01"
