from __future__ import annotations

import tempfile
from collections.abc import Callable
from shutil import copy2
from pathlib import Path

from transcripio.config import AppConfig
from transcripio.diarization import LocalPyannoteDiarizer, assign_speakers
from transcripio.media import ensure_audio_wav
from transcripio.models import TranscriptionResult
from transcripio.transcriber import WhisperTranscriber

ProgressCallback = Callable[[str, float], None]


class TranscriptionPipeline:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._transcriber = WhisperTranscriber(config)
        self._diarizer = LocalPyannoteDiarizer(config)

    def run(self, input_path: Path, on_step: ProgressCallback | None = None) -> TranscriptionResult:
        notify = on_step or (lambda _message, _value: None)

        with tempfile.TemporaryDirectory(prefix="transcripio-") as tmp_dir:
            work_dir = Path(tmp_dir)

            notify("Извлекаю аудио", 0.15)
            audio_path = ensure_audio_wav(input_path, work_dir, self._config.ffmpeg_path)

            notify("Загружаю локальную Whisper модель и распознаю речь", 0.35)
            transcript_segments, language, duration = self._transcriber.transcribe(audio_path)

            if self._diarizer.is_enabled():
                notify("Распознаю участников", 0.75)
                diarization_segments = self._diarizer.diarize(audio_path)
                transcript_segments = assign_speakers(transcript_segments, diarization_segments)

            notify("Формирую результат", 0.95)
            self._config.output_dir.mkdir(parents=True, exist_ok=True)
            stable_audio_path = self._config.output_dir / f"{input_path.stem}.prepared.wav"
            copy2(audio_path, stable_audio_path)

            result = TranscriptionResult(
                source_path=input_path,
                audio_path=stable_audio_path,
                language=language,
                duration=duration,
                segments=transcript_segments,
            )
            notify("Готово", 1.0)
            return result
