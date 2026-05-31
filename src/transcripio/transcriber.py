from __future__ import annotations

from pathlib import Path
from collections.abc import Callable

from transcripio.config import AppConfig
from transcripio.models import TranscriptSegment

SegmentProgressCallback = Callable[[TranscriptSegment, float | None], None]


class WhisperTranscriber:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self._model = WhisperModel(
            self._config.whisper_model,
            device=self._config.device,
            compute_type=self._config.compute_type,
        )
        return self._model

    def transcribe(
        self,
        audio_path: Path,
        on_segment: SegmentProgressCallback | None = None,
    ) -> tuple[list[TranscriptSegment], str | None, float | None]:
        model = self._load_model()
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=self._config.language,
            vad_filter=True,
            beam_size=5,
        )

        duration = getattr(info, "duration", None)
        segments: list[TranscriptSegment] = []
        for item in segments_iter:
            segment = TranscriptSegment(start=item.start, end=item.end, text=item.text.strip())
            segments.append(segment)
            if on_segment:
                ratio = min(segment.end / duration, 1.0) if duration else None
                on_segment(segment, ratio)

        language = getattr(info, "language", self._config.language)
        return segments, language, duration
