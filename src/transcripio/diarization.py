from __future__ import annotations

import os
import sys
import wave
import warnings
from array import array
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
                warnings.filterwarnings(
                    "ignore",
                    message=r"\s*torchcodec is not installed correctly.*",
                )
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
                    "pyannote could not use TorchCodec on this Windows environment."
                ) from exc
            raise

        speaker_annotation = _speaker_annotation_from_pipeline_output(annotation)

        diarized: list[DiarizationSegment] = []
        for turn, _, speaker in speaker_annotation.itertracks(yield_label=True):
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


def _speaker_annotation_from_pipeline_output(output):
    if hasattr(output, "itertracks"):
        return output

    speaker_diarization = getattr(output, "speaker_diarization", None)
    if hasattr(speaker_diarization, "itertracks"):
        return speaker_diarization

    raise RuntimeError("pyannote returned an unsupported diarization output format.")


def _overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def _load_waveform_for_diarization(audio_path: Path):
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise RuntimeError("torchaudio from requirements.txt is required for diarization") from exc

    try:
        waveform, sample_rate = torchaudio.load(str(audio_path))
    except RuntimeError as exc:
        if not _is_torchcodec_error(exc):
            raise
        waveform, sample_rate = _load_wav_with_standard_library(audio_path, torch)
    if waveform.ndim != 2:
        raise RuntimeError("Prepared audio waveform has an unexpected shape.")
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.to(dtype=torch.float32).contiguous(), int(sample_rate)


def _is_torchcodec_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "torchcodec" in message or "libtorchcodec" in message


def _load_wav_with_standard_library(audio_path: Path, torch):
    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
    except wave.Error as exc:
        raise RuntimeError(
            "torchaudio could not load audio and the fallback WAV reader could not read it."
        ) from exc

    if channels <= 0:
        raise RuntimeError("Prepared audio WAV has no channels.")

    samples = _decode_pcm_samples(frames, sample_width)
    waveform = torch.tensor(samples, dtype=torch.float32)
    if waveform.numel() % channels != 0:
        raise RuntimeError("Prepared audio WAV has an unexpected frame layout.")
    waveform = waveform.reshape(-1, channels).transpose(0, 1)
    return waveform, sample_rate


def _decode_pcm_samples(frames: bytes, sample_width: int) -> list[float]:
    if sample_width == 1:
        return [(sample - 128) / 128.0 for sample in frames]

    if sample_width == 2:
        samples = array("h")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        return [sample / 32768.0 for sample in samples]

    if sample_width == 3:
        return [
            int.from_bytes(frames[index : index + 3], byteorder="little", signed=True)
            / 8388608.0
            for index in range(0, len(frames), 3)
        ]

    if sample_width == 4:
        samples = array("i")
        samples.frombytes(frames)
        if sys.byteorder != "little":
            samples.byteswap()
        return [sample / 2147483648.0 for sample in samples]

    raise RuntimeError(f"Unsupported WAV sample width for diarization: {sample_width} bytes.")
