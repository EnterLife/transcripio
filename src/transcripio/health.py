from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from transcripio.config import AppConfig
from transcripio.cuda_runtime import configure_cuda_dll_paths


HealthStatus = Literal["ok", "warning", "error"]


@dataclass(frozen=True, slots=True)
class HealthCheck:
    name: str
    status: HealthStatus
    message: str


def run_environment_checks(config: AppConfig) -> list[HealthCheck]:
    checks = [
        check_ffmpeg(config.ffmpeg_path),
        check_writable_dir("Output directory", config.output_dir),
        check_writable_dir("History directory", config.history_dir),
        check_whisper_model(config.whisper_model, config.local_files_only),
    ]
    proxy_check = check_network_proxy(config.whisper_model, config.local_files_only)
    if proxy_check is not None:
        checks.append(proxy_check)

    if config.diarization_model_path:
        checks.append(check_existing_file("Diarization model", Path(config.diarization_model_path)))

    if config.device == "cuda":
        checks.append(check_cuda_runtime(config.allow_cpu_fallback))

    return checks


def check_ffmpeg(ffmpeg_path: str) -> HealthCheck:
    try:
        completed = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except FileNotFoundError:
        return HealthCheck(
            name="ffmpeg",
            status="error",
            message="ffmpeg was not found. Install ffmpeg or set transcription.ffmpeg_path.",
        )
    except subprocess.TimeoutExpired:
        return HealthCheck(
            name="ffmpeg",
            status="error",
            message="ffmpeg did not respond within 5 seconds.",
        )

    if completed.returncode != 0:
        details = _first_line(completed.stderr) or _first_line(completed.stdout)
        return HealthCheck(
            name="ffmpeg",
            status="error",
            message=f"ffmpeg check failed: {details or 'unknown error'}",
        )

    return HealthCheck(name="ffmpeg", status="ok", message="ffmpeg is available.")


def check_writable_dir(name: str, directory: Path) -> HealthCheck:
    probe_path = directory / ".transcripio-write-test"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        probe_path.write_text("ok", encoding="utf-8")
        probe_path.unlink()
    except OSError as exc:
        return HealthCheck(
            name=name,
            status="error",
            message=f"Cannot write to {directory}: {exc}",
        )

    return HealthCheck(name=name, status="ok", message=f"{directory} is writable.")


def check_whisper_model(model_name_or_path: str, local_files_only: bool) -> HealthCheck:
    model_path = Path(model_name_or_path)
    if _looks_like_local_path(model_name_or_path):
        if model_path.exists():
            return HealthCheck(
                name="Whisper model",
                status="ok",
                message=f"Local model path exists: {model_path}",
            )
        return HealthCheck(
            name="Whisper model",
            status="error",
            message=f"Local model path was not found: {model_path}",
        )

    if local_files_only:
        return HealthCheck(
            name="Whisper model",
            status="warning",
            message=(
                "A model name is selected with offline-only mode. Make sure it is already "
                "downloaded in the Hugging Face cache."
            ),
        )

    return HealthCheck(
        name="Whisper model",
        status="ok",
        message="Model name is configured. It may be downloaded by faster-whisper on first use.",
    )


def check_network_proxy(model_name_or_path: str, local_files_only: bool) -> HealthCheck | None:
    if local_files_only or _looks_like_local_path(model_name_or_path):
        return None

    for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY"):
        value = _proxy_env_value(name)
        if not value:
            continue
        parsed = urlparse(value)
        scheme = parsed.scheme.lower()
        if scheme in {"http", "https"}:
            return HealthCheck(
                name="Network proxy",
                status="ok",
                message=f"{name} uses a supported {scheme} proxy.",
            )
        if scheme:
            return HealthCheck(
                name="Network proxy",
                status="error",
                message=(
                    f"{name} uses unsupported proxy scheme '{scheme}'. "
                    "Use an http:// or https:// proxy, unset the proxy variable, "
                    "or select a downloaded/local Whisper model with offline-only mode."
                ),
            )
    return None


def check_existing_file(name: str, path: Path) -> HealthCheck:
    if path.exists():
        return HealthCheck(name=name, status="ok", message=f"{path} exists.")
    return HealthCheck(name=name, status="error", message=f"{path} was not found.")


def check_cuda_runtime(allow_cpu_fallback: bool) -> HealthCheck:
    status = configure_cuda_dll_paths()
    if status.is_ready:
        return HealthCheck(
            name="CUDA runtime",
            status="ok",
            message="CUDA runtime DLLs were found.",
        )

    missing = ", ".join(status.missing_dlls)
    health_status: HealthStatus = "warning" if allow_cpu_fallback else "error"
    fallback = " Transcription can fall back to CPU." if allow_cpu_fallback else ""
    return HealthCheck(
        name="CUDA runtime",
        status=health_status,
        message=f"Missing CUDA runtime DLLs: {missing}.{fallback}",
    )


def _looks_like_local_path(value: str) -> bool:
    cleaned = value.strip()
    path = Path(cleaned)
    if path.exists() or path.is_absolute():
        return True
    if cleaned.startswith((".", "~")) or "\\" in cleaned:
        return True
    if "/" in cleaned:
        first_part = cleaned.split("/", 1)[0].lower()
        return first_part in {"data", "models"}
    return False


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0] if text.strip() else ""


def _proxy_env_value(name: str) -> str | None:
    return os.environ.get(name) or os.environ.get(name.lower())
