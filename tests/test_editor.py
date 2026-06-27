from __future__ import annotations

from transcripio.editor import (
    find_segments,
    merge_segments,
    rename_speaker,
    replace_in_segments,
    split_segment_at,
)
from transcripio.models import TranscriptSegment, TranscriptWord


def test_find_segments_matches_text_and_speaker() -> None:
    segments = [
        TranscriptSegment(start=0, end=1, text="hello world", speaker="A"),
        TranscriptSegment(start=1, end=2, text="next", speaker="Speaker Bob"),
    ]

    assert find_segments(segments, "WORLD") == [0]
    assert find_segments(segments, "bob") == [1]


def test_replace_in_segments_clears_stale_words() -> None:
    segments = [
        TranscriptSegment(
            start=0,
            end=1,
            text="hello world",
            words=[TranscriptWord(start=0, end=0.5, text="hello")],
        )
    ]

    updated, count = replace_in_segments(segments, "world", "team")

    assert count == 1
    assert updated[0].text == "hello team"
    assert updated[0].words == []


def test_rename_speaker_updates_all_matching_segments() -> None:
    segments = [
        TranscriptSegment(start=0, end=1, text="a", speaker="SPEAKER_00"),
        TranscriptSegment(start=1, end=2, text="b", speaker="SPEAKER_01"),
        TranscriptSegment(start=2, end=3, text="c", speaker="SPEAKER_00"),
    ]

    updated, count = rename_speaker(segments, "SPEAKER_00", "Pavel")

    assert count == 2
    assert [segment.speaker for segment in updated] == ["Pavel", "SPEAKER_01", "Pavel"]


def test_split_segment_at_uses_word_timestamps() -> None:
    segments = [
        TranscriptSegment(
            start=0,
            end=4,
            text="hello brave world",
            speaker="A",
            words=[
                TranscriptWord(start=0.0, end=1.0, text="hello"),
                TranscriptWord(start=1.1, end=2.0, text="brave"),
                TranscriptWord(start=2.1, end=3.0, text="world"),
            ],
        )
    ]

    updated = split_segment_at(segments, 0, 2.05)

    assert len(updated) == 2
    assert updated[0].start == 0
    assert updated[0].end == 2.05
    assert updated[0].text == "hello brave"
    assert updated[1].text == "world"


def test_merge_segments_combines_neighbors() -> None:
    segments = [
        TranscriptSegment(start=0, end=1, text="hello", speaker="A"),
        TranscriptSegment(start=1, end=2, text="world", speaker="A"),
        TranscriptSegment(start=2, end=3, text="next", speaker="B"),
    ]

    updated = merge_segments(segments, 0)

    assert len(updated) == 2
    assert updated[0].start == 0
    assert updated[0].end == 2
    assert updated[0].speaker == "A"
    assert updated[0].text == "hello world"
