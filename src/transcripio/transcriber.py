from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

from transcripio.config import AppConfig
from transcripio.cuda_runtime import configure_cuda_dll_paths, install_cuda_runtime_packages
from transcripio.models import TranscriptSegment

SegmentProgressCallback = Callable[[TranscriptSegment, float | None], None]


class WhisperTranscriber:
    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._model = None
        self.runtime_notice: str | None = None

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import BatchedInferencePipeline, WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            model = self._create_model(
                WhisperModel,
                device=self._config.device,
                compute_type=self._config.compute_type,
            )
            self._model = self._wrap_model(model, BatchedInferencePipeline)
        except Exception as exc:
            if not self._should_fallback_to_cpu(exc):
                raise
            self.runtime_notice = (
                "CUDA libraries are not available, so transcription is running on CPU with int8."
            )
            model = self._create_model(WhisperModel, device="cpu", compute_type="int8")
            self._model = self._wrap_model(model, BatchedInferencePipeline)
        return self._model

    def _create_model(self, model_class, device: str, compute_type: str):
        if device == "cuda":
            cuda_status = configure_cuda_dll_paths()
            if not cuda_status.is_ready and self._config.auto_install_cuda_runtime:
                completed = install_cuda_runtime_packages()
                if completed.returncode != 0:
                    details = completed.stderr.strip() or completed.stdout.strip()
                    raise RuntimeError(f"Could not install CUDA runtime packages: {details}")
                cuda_status = configure_cuda_dll_paths()
            if not cuda_status.is_ready:
                missing = ", ".join(cuda_status.missing_dlls)
                raise RuntimeError(f"Missing CUDA runtime DLLs: {missing}")
        return model_class(
            self._config.whisper_model,
            device=device,
            compute_type=compute_type,
            local_files_only=self._config.local_files_only,
            use_auth_token=os.environ.get("HF_TOKEN") or None,
            cpu_threads=self._config.cpu_threads,
            num_workers=self._config.num_workers,
        )

    def _wrap_model(self, model, batched_pipeline_class):
        if not self._config.use_batched_inference:
            return model
        return batched_pipeline_class(model)

    def _should_fallback_to_cpu(self, exc: Exception) -> bool:
        if not self._config.allow_cpu_fallback or self._config.device != "cuda":
            return False

        message = str(exc).lower()
        cuda_markers = ("cuda", "cublas", "cudnn", "nvidia")
        return any(marker in message for marker in cuda_markers)

    def transcribe(
        self,
        audio_path: Path,
        on_segment: SegmentProgressCallback | None = None,
    ) -> tuple[list[TranscriptSegment], str | None, float | None]:
        model = self._load_model()
        transcribe_options = {
            "language": self._config.language,
            "vad_filter": self._config.vad_filter,
            "beam_size": self._config.beam_size,
            "best_of": self._config.best_of,
        }
        if self._config.use_batched_inference:
            transcribe_options["batch_size"] = self._config.batch_size

        segments_iter, info = model.transcribe(str(audio_path), **transcribe_options)

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
