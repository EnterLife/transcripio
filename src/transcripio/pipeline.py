from __future__ import annotations

import tempfile
from collections.abc import Callable
from shutil import copy2
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

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

            notify("Extracting audio", 0.15)
            audio_path = ensure_audio_wav(input_path, work_dir, self._config.ffmpeg_path)

            notify("Loading the local Whisper model", 0.30)

            def on_segment(segment, ratio: float | None) -> None:
                if ratio is None:
                    notify(f"Transcribed segment ending at {segment.end:.1f}s", 0.55)
                    return
                notify(f"Transcribing speech: {ratio:.0%}", 0.35 + (ratio * 0.35))

            transcript_segments, language, duration = self._transcriber.transcribe(
                audio_path,
                on_segment=on_segment,
            )
            if self._transcriber.runtime_notice:
                notify(self._transcriber.runtime_notice, 0.72)

            if self._diarizer.is_enabled():
                notify("Assigning speakers with the local diarization model", 0.75)
                diarization_segments = self._diarizer.diarize(audio_path)
                transcript_segments = assign_speakers(transcript_segments, diarization_segments)

            notify("Writing result artifacts", 0.95)
            self._config.output_dir.mkdir(parents=True, exist_ok=True)
            job_id = uuid4().hex
            stable_audio_path = self._config.output_dir / f"{input_path.stem}-{job_id[:8]}.prepared.wav"
            copy2(audio_path, stable_audio_path)

            result = TranscriptionResult(
                job_id=job_id,
                source_name=input_path.name,
                source_path=input_path,
                audio_path=stable_audio_path,
                language=language,
                duration=duration,
                created_at=datetime.now(timezone.utc).isoformat(),
                segments=transcript_segments,
            )
            notify("Done", 1.0)
            return result
