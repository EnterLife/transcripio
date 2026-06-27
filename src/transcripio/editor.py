from __future__ import annotations

from dataclasses import replace

from transcripio.models import TranscriptSegment, TranscriptWord


def find_segments(segments: list[TranscriptSegment], query: str) -> list[int]:
    clean_query = query.strip().casefold()
    if not clean_query:
        return []
    return [
        index
        for index, segment in enumerate(segments)
        if clean_query in segment.text.casefold()
        or (segment.speaker and clean_query in segment.speaker.casefold())
    ]


def replace_in_segments(
    segments: list[TranscriptSegment],
    search_text: str,
    replacement_text: str,
    *,
    case_sensitive: bool = False,
) -> tuple[list[TranscriptSegment], int]:
    if not search_text:
        return segments, 0

    updated: list[TranscriptSegment] = []
    replacements = 0
    for segment in segments:
        new_text, count = _replace_text(
            segment.text,
            search_text,
            replacement_text,
            case_sensitive=case_sensitive,
        )
        replacements += count
        updated.append(
            replace(
                segment,
                text=new_text,
                words=segment.words if new_text == segment.text else [],
            )
        )
    return updated, replacements


def rename_speaker(
    segments: list[TranscriptSegment],
    old_speaker: str,
    new_speaker: str | None,
) -> tuple[list[TranscriptSegment], int]:
    clean_old = old_speaker.strip()
    clean_new = new_speaker.strip() if new_speaker else None
    if not clean_old:
        return segments, 0

    updated: list[TranscriptSegment] = []
    count = 0
    for segment in segments:
        if segment.speaker == clean_old:
            updated.append(replace(segment, speaker=clean_new or None))
            count += 1
        else:
            updated.append(segment)
    return updated, count


def split_segment_at(
    segments: list[TranscriptSegment],
    index: int,
    split_time: float,
) -> list[TranscriptSegment]:
    if index < 0 or index >= len(segments):
        raise ValueError("Segment index is out of range.")

    segment = segments[index]
    if split_time <= segment.start or split_time >= segment.end:
        raise ValueError("Split time must be inside the selected segment.")

    first_words = [word for word in segment.words if word.end <= split_time]
    second_words = [word for word in segment.words if word.start >= split_time]
    first_text, second_text = _split_text_for_time(segment, split_time, first_words, second_words)

    first = TranscriptSegment(
        start=segment.start,
        end=split_time,
        text=first_text,
        speaker=segment.speaker,
        words=first_words,
    )
    second = TranscriptSegment(
        start=split_time,
        end=segment.end,
        text=second_text,
        speaker=segment.speaker,
        words=second_words,
    )
    return [*segments[:index], first, second, *segments[index + 1 :]]


def merge_segments(
    segments: list[TranscriptSegment],
    first_index: int,
) -> list[TranscriptSegment]:
    if first_index < 0 or first_index + 1 >= len(segments):
        raise ValueError("Choose a segment that has a following segment to merge.")

    first = segments[first_index]
    second = segments[first_index + 1]
    speaker = first.speaker if first.speaker == second.speaker else first.speaker or second.speaker
    merged = TranscriptSegment(
        start=first.start,
        end=second.end,
        text=" ".join(part for part in [first.text.strip(), second.text.strip()] if part),
        speaker=speaker,
        words=[*first.words, *second.words],
    )
    return [*segments[:first_index], merged, *segments[first_index + 2 :]]


def _replace_text(
    text: str,
    search_text: str,
    replacement_text: str,
    *,
    case_sensitive: bool,
) -> tuple[str, int]:
    if case_sensitive:
        return text.replace(search_text, replacement_text), text.count(search_text)

    lowered = text.casefold()
    needle = search_text.casefold()
    pieces: list[str] = []
    start = 0
    count = 0
    while True:
        index = lowered.find(needle, start)
        if index == -1:
            pieces.append(text[start:])
            break
        pieces.append(text[start:index])
        pieces.append(replacement_text)
        start = index + len(search_text)
        count += 1
    return "".join(pieces), count


def _split_text_for_time(
    segment: TranscriptSegment,
    split_time: float,
    first_words: list[TranscriptWord],
    second_words: list[TranscriptWord],
) -> tuple[str, str]:
    if first_words and second_words:
        return _words_to_text(first_words), _words_to_text(second_words)

    ratio = (split_time - segment.start) / (segment.end - segment.start)
    words = segment.text.split()
    if len(words) < 2:
        return segment.text.strip(), ""
    split_index = min(max(1, round(len(words) * ratio)), len(words) - 1)
    return " ".join(words[:split_index]), " ".join(words[split_index:])


def _words_to_text(words: list[TranscriptWord]) -> str:
    return " ".join(word.text for word in words if word.text).strip()
