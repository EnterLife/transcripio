from __future__ import annotations

import os
import warnings
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

        os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*torchcodec is not installed correctly.*")
                from pyannote.audio import Pipeline
                self._pipeline = Pipeline.from_pretrained(str(model_path))
        except ImportError as exc:
            raise RuntimeError(
                "Diarization dependencies are missing. Run: pip install -r requirements.txt"
            ) from exc
        return self._pipeline

    def diarize(self, audio_path: Path) -> list[DiarizationSegment]:
        pipeline = self._load_pipeline()

        waveform, sample_rate = _load_waveform_for_diarization(audio_path)
        file = {
            "waveform": waveform,
            "sample_rate": sample_rate,
            "uri": audio_path.stem,
        }

        try:
            annotation = pipeline(file)
        except Exception as exc:
            if _is_torchcodec_error(exc):
                raise RuntimeError(
                    "Diarization failed because pyannote tried to use TorchCodec audio decoding. "
                    "Transcripio now passes preloaded audio to avoid TorchCodec; restart the app and "
                    "try again. If it still fails, use the Community-1 diarization model."
                ) from exc
            raise

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


def _load_waveform_for_diarization(audio_path: Path):
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio from requirements.txt is required for diarization") from exc

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.ndim != 2:
        raise RuntimeError("Prepared audio waveform has an unexpected shape.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.to(dtype=torch.float32).contiguous(), int(sample_rate)


def _is_torchcodec_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "torchcodec" in message or "libtorchcodec" in message
