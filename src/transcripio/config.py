from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SETTINGS_PATH = Path("settings.json")


@dataclass(slots=True)
class AppConfig:
    whisper_model: str = "small"
    device: str = "cpu"
    compute_type: str = "int8"
    language: str | None = None
    diarization_model_path: str | None = None
    ffmpeg_path: str = "ffmpeg"
    output_dir: Path = Path("data/output")
    history_dir: Path = Path("data/history")
    local_files_only: bool = False
    allow_cpu_fallback: bool = True
    auto_install_cuda_runtime: bool = True
    use_batched_inference: bool = False
    batch_size: int = 8
    beam_size: int = 1
    best_of: int = 1
    cpu_threads: int = 0
    num_workers: int = 1
    vad_filter: bool = True


@dataclass(slots=True)
class AppSettings:
    page_title: str = "Transcripio"
    page_icon: str = "T"
    caption: str = "Local audio and video transcription with optional local speaker diarization."
    upload_types: tuple[str, ...] = (
        "mp3",
        "wav",
        "m4a",
        "flac",
        "ogg",
        "aac",
        "mp4",
        "mov",
        "mkv",
        "avi",
        "webm",
        "m4v",
    )
    whisper_models: tuple[str, ...] = (
        "tiny",
        "base",
        "small",
        "medium",
        "large-v3",
        "Systran/faster-whisper-small",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    )
    config: AppConfig = field(default_factory=AppConfig)


class SettingsError(ValueError):
    pass


def load_settings(path: Path = SETTINGS_PATH) -> AppSettings:
    if not path.exists():
        return AppSettings()

    try:
        raw_settings = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SettingsError(f"Settings file {path} contains invalid JSON: {exc}") from exc

    if not isinstance(raw_settings, dict):
        raise SettingsError(f"Settings file {path} must contain a JSON object.")

    ui_settings = _section(raw_settings, "ui")
    transcription_settings = _section(raw_settings, "transcription")
    storage_settings = _section(raw_settings, "storage")

    default_config = AppConfig()
    config = AppConfig(
        whisper_model=_text(
            transcription_settings.get("whisper_model"),
            default_config.whisper_model,
        ),
        device=_text(transcription_settings.get("device"), default_config.device),
        compute_type=_text(
            transcription_settings.get("compute_type"),
            default_config.compute_type,
        ),
        language=_optional_text(transcription_settings.get("language"), default_config.language),
        diarization_model_path=_optional_text(
            transcription_settings.get("diarization_model_path"),
            default_config.diarization_model_path,
        ),
        ffmpeg_path=_text(transcription_settings.get("ffmpeg_path"), default_config.ffmpeg_path),
        output_dir=Path(_text(storage_settings.get("output_dir"), str(default_config.output_dir))),
        history_dir=Path(_text(storage_settings.get("history_dir"), str(default_config.history_dir))),
        local_files_only=_bool(
            transcription_settings.get("local_files_only"),
            default_config.local_files_only,
        ),
        allow_cpu_fallback=_bool(
            transcription_settings.get("allow_cpu_fallback"),
            default_config.allow_cpu_fallback,
        ),
        auto_install_cuda_runtime=_bool(
            transcription_settings.get("auto_install_cuda_runtime"),
            default_config.auto_install_cuda_runtime,
        ),
        use_batched_inference=_bool(
            transcription_settings.get("use_batched_inference"),
            default_config.use_batched_inference,
        ),
        batch_size=_positive_int(
            transcription_settings.get("batch_size"),
            default_config.batch_size,
        ),
        beam_size=_positive_int(
            transcription_settings.get("beam_size"),
            default_config.beam_size,
        ),
        best_of=_positive_int(
            transcription_settings.get("best_of"),
            default_config.best_of,
        ),
        cpu_threads=_non_negative_int(
            transcription_settings.get("cpu_threads"),
            default_config.cpu_threads,
        ),
        num_workers=_positive_int(
            transcription_settings.get("num_workers"),
            default_config.num_workers,
        ),
        vad_filter=_bool(
            transcription_settings.get("vad_filter"),
            default_config.vad_filter,
        ),
    )

    defaults = AppSettings()
    return AppSettings(
        page_title=_text(ui_settings.get("page_title"), defaults.page_title),
        page_icon=_text(ui_settings.get("page_icon"), defaults.page_icon),
        caption=_text(ui_settings.get("caption"), defaults.caption),
        upload_types=_upload_types(ui_settings.get("upload_types"), defaults.upload_types),
        whisper_models=_text_list(
            transcription_settings.get("whisper_models"),
            defaults.whisper_models,
        ),
        config=config,
    )


def _section(settings: dict[str, Any], name: str) -> dict[str, Any]:
    value = settings.get(name, {})
    if not isinstance(value, dict):
        raise SettingsError(f"Settings section '{name}' must be an object.")
    return value


def _text(value: Any, default: str) -> str:
    if value is None:
        return default
    return str(value).strip() or default


def _optional_text(value: Any, default: str | None) -> str | None:
    if value is None:
        return default
    text = str(value).strip()
    return text or None


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


def _positive_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _non_negative_int(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed >= 0 else default


def _upload_types(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list):
        raise SettingsError("Settings value 'ui.upload_types' must be a list.")

    upload_types = tuple(
        str(item).strip().lower().lstrip(".") for item in value if str(item).strip()
    )
    return upload_types or default


def _text_list(value: Any, default: tuple[str, ...]) -> tuple[str, ...]:
    if value is None:
        return default
    if not isinstance(value, list):
        raise SettingsError("Settings value must be a list of text values.")

    text_values = tuple(str(item).strip() for item in value if str(item).strip())
    return text_values or default
