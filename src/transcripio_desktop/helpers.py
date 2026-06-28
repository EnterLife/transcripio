from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

from transcripio.config import AppConfig
from transcripio.formatters import to_docx, to_json, to_srt, to_txt, to_vtt, to_words_csv
from transcripio.health import HealthCheck
from transcripio.models import TranscriptionResult


@dataclass(frozen=True, slots=True)
class ExportArtifact:
    file_name: str
    data: bytes
    mime: str


EXPORT_FORMATS = ("TXT", "SRT", "VTT", "DOCX", "JSON", "Words CSV")


def blocking_health_message(checks: list[HealthCheck]) -> str | None:
    errors = [check for check in checks if check.status == "error"]
    if not errors:
        return None

    lines = ["Fix these issues before processing:"]
    lines.extend(f"- {check.name}: {check.message}" for check in errors)
    return "\n".join(lines)


def health_log_lines(checks: list[HealthCheck]) -> list[str]:
    return [f"  {check.name}: {check.status} - {check.message}" for check in checks]


def build_desktop_config(
    base_config: AppConfig,
    *,
    whisper_model: str,
    device: str,
    compute_type: str,
    language: str,
    diarization_model_path: str,
    ffmpeg_path: str,
    output_dir: Path,
    history_dir: Path,
    local_files_only: bool,
    vad_filter: bool,
    word_timestamps: bool,
    beam_size: int,
    best_of: int,
    cpu_threads: int,
    num_workers: int,
    initial_prompt: str = "",
    hotwords: str = "",
) -> AppConfig:
    return AppConfig(
        whisper_model=whisper_model.strip() or base_config.whisper_model,
        device=device.strip() or base_config.device,
        compute_type=compute_type.strip() or base_config.compute_type,
        language=language.strip() or None,
        diarization_model_path=diarization_model_path.strip() or None,
        diarization_repo_id=base_config.diarization_repo_id,
        diarization_output_dir=base_config.diarization_output_dir,
        ffmpeg_path=ffmpeg_path.strip() or base_config.ffmpeg_path,
        output_dir=output_dir,
        history_dir=history_dir,
        local_files_only=local_files_only,
        allow_cpu_fallback=base_config.allow_cpu_fallback,
        auto_install_cuda_runtime=base_config.auto_install_cuda_runtime,
        use_batched_inference=base_config.use_batched_inference,
        batch_size=base_config.batch_size,
        beam_size=max(1, int(beam_size)),
        best_of=max(1, int(best_of)),
        cpu_threads=max(0, int(cpu_threads)),
        num_workers=max(1, int(num_workers)),
        vad_filter=vad_filter,
        initial_prompt=initial_prompt.strip() or None,
        hotwords=hotwords.strip() or None,
        word_timestamps=word_timestamps,
        condition_on_previous_text=base_config.condition_on_previous_text,
        no_speech_threshold=base_config.no_speech_threshold,
        language_detection_threshold=base_config.language_detection_threshold,
        hallucination_silence_threshold=base_config.hallucination_silence_threshold,
    )


def result_title(result: TranscriptionResult) -> str:
    created_at = result.created_at[:19].replace("T", " ") if result.created_at else "unknown"
    return f"{result.source_name} | {created_at} | {result.job_id[:8]}"


def result_metrics(result: TranscriptionResult) -> dict[str, str]:
    speakers = {segment.speaker for segment in result.segments if segment.speaker}
    return {
        "Language": result.language or "Unknown",
        "Duration": format_duration(result.duration),
        "Segments": str(len(result.segments)),
        "Speakers": str(len(speakers)) if speakers else "None",
        "Created": result.created_at[:19].replace("T", " ") if result.created_at else "Unknown",
    }


def format_duration(duration: float | None) -> str:
    if duration is None:
        return "Unknown"
    total_seconds = max(0, int(round(duration)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def format_file_size(size_bytes: int) -> str:
    size = max(0, float(size_bytes))
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def export_result(result: TranscriptionResult, format_name: str) -> ExportArtifact:
    base_name = safe_file_stem(Path(result.source_name).stem or "transcript")
    normalized = format_name.strip().lower()
    if normalized == "txt":
        return _text_artifact(f"{base_name}.txt", to_txt(result.segments), "text/plain")
    if normalized == "srt":
        return _text_artifact(f"{base_name}.srt", to_srt(result.segments), "application/x-subrip")
    if normalized == "vtt":
        return _text_artifact(f"{base_name}.vtt", to_vtt(result.segments), "text/vtt")
    if normalized == "json":
        return _text_artifact(f"{base_name}.json", to_json(result), "application/json")
    if normalized == "docx":
        return ExportArtifact(
            file_name=f"{base_name}.docx",
            data=to_docx(result),
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )
    if normalized in {"words csv", "csv"}:
        return _text_artifact(f"{base_name}-words.csv", to_words_csv(result.segments), "text/csv")
    raise ValueError(f"Unsupported export format: {format_name}")


def safe_file_stem(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value).strip(" .")
    return cleaned or "transcript"


def _text_artifact(file_name: str, text: str, mime: str) -> ExportArtifact:
    return ExportArtifact(file_name=file_name, data=text.encode("utf-8"), mime=mime)
