from __future__ import annotations

from pathlib import Path

from transcripio.config import AppConfig
from transcripio.models import DiarizationSegment, TranscriptSegment


class LocalPyannoteDiarizer:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._pipeline = None

    def is_enabled(self) -> bool:
        return bool(self._config.diarization_model_path)

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline

        if not self._config.diarization_model_path:
            raise RuntimeError("A local diarization model path was not provided.")

        model_path = Path(self._config.diarization_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Local diarization model was not found: {model_path}")

        try:
            from pyannote.audio import Pipeline
        except ImportError as exc:
            raise RuntimeError(
                "Diarization dependencies are missing. Run: pip install -r requirements.txt"
            ) from exc

        self._pipeline = Pipeline.from_pretrained(str(model_path))
        return self._pipeline

    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        pipeline = self._load_pipeline()

        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError("torchaudio from requirements.txt is required for diarization") from exc

        waveform, sample_rate = torchaudio.load(str(audio_path))
        annotation = pipeline({"waveform": waveform, "sample_rate": sample_rate})

        diarized: list[DiarizationSegment] = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            diarized.append(
                DiarizationSegment(
                    start=float(turn.start),
                    end=float(turn.end),
                    speaker=str(speaker),
                )
            )
        return diarized


def assign_speakers(
    transcript_segments: list[TranscriptSegment],
    diarization_segments: list[DiarizationSegment],
) -> list[TranscriptSegment]:
    for transcript in transcript_segments:
        overlaps: dict[str, float] = {}
        for diarized in diarization_segments:
            overlap = _overlap_seconds(transcript.start, transcript.end, diarized.start, diarized.end)
            if overlap > 0:
                overlaps[diarized.speaker] = overlaps.get(diarized.speaker, 0.0) + overlap

        if overlaps:
            transcript.speaker = max(overlaps, key=overlaps.get)

    return transcript_segments


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))
