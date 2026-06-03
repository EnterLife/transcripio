from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


SHORT_MODEL_REPOS = {
    "tiny": "Systran/faster-whisper-tiny",
    "base": "Systran/faster-whisper-base",
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
    "large-v3": "Systran/faster-whisper-large-v3",
}


@dataclass(frozen=True, slots=True)
class WhisperModelOption:
    label: str
    value: str
    source: str
    is_downloaded: bool


@dataclass(frozen=True, slots=True)
class DiarizationModelOption:
    label: str
    value: str


def list_whisper_model_options(
    configured_models: tuple[str, ...],
    models_dir: Path = Path("models"),
    hf_cache_dir: Path | None = None,
) -> list[WhisperModelOption]:
    cached_repos = discover_cached_hf_model_repos(hf_cache_dir)
    local_paths = discover_local_whisper_model_paths(models_dir)

    options: list[WhisperModelOption] = []
    seen_values: set[str] = set()

    for model in configured_models:
        downloaded = model in cached_repos or SHORT_MODEL_REPOS.get(model) in cached_repos
        source = "downloaded" if downloaded else "available"
        _append_option(options, seen_values, model, model, source, downloaded)

    for repo_id in sorted(cached_repos):
        _append_option(options, seen_values, repo_id, repo_id, "downloaded", True)

    for path in local_paths:
        value = str(path)
        _append_option(options, seen_values, path.name, value, "local", True)

    return options


def discover_cached_hf_model_repos(cache_dir: Path | None = None) -> set[str]:
    cache_root = cache_dir or _default_hf_cache_dir()
    if not cache_root.exists():
        return set()

    repos: set[str] = set()
    for path in cache_root.glob("models--*--*"):
        snapshots_dir = path / "snapshots"
        if not snapshots_dir.exists() or not any(snapshots_dir.iterdir()):
            continue
        repo_id = path.name.removeprefix("models--").replace("--", "/", 1)
        repos.add(repo_id)
    return repos


def discover_local_whisper_model_paths(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []

    model_paths: list[Path] = []
    for path in models_dir.rglob("*"):
        if not path.is_dir():
            continue
        if (path / "model.bin").exists() and (path / "config.json").exists():
            model_paths.append(path)
    return sorted(model_paths)


def list_diarization_model_options(models_dir: Path = Path("models")) -> list[DiarizationModelOption]:
    return [
        DiarizationModelOption(label=str(path), value=str(path))
        for path in discover_local_diarization_config_paths(models_dir)
    ]


def discover_local_diarization_config_paths(models_dir: Path) -> list[Path]:
    if not models_dir.exists():
        return []

    return sorted(
        path
        for path in models_dir.rglob("config.yaml")
        if path.is_file() and _looks_like_pyannote_config(path)
    )


def _looks_like_pyannote_config(path: Path) -> bool:
    path_text = str(path).lower()
    if "pyannote" in path_text or "diarization" in path_text:
        return True

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return "pyannote" in text.lower() or "diarization" in text.lower()


def _default_hf_cache_dir() -> Path:
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _append_option(
    options: list[WhisperModelOption],
    seen_values: set[str],
    label: str,
    value: str,
    source: str,
    is_downloaded: bool,
) -> None:
    if value in seen_values:
        return
    seen_values.add(value)
    status = "downloaded" if is_downloaded else source
    options.append(
        WhisperModelOption(
            label=f"{label} ({status})",
            value=value,
            source=source,
            is_downloaded=is_downloaded,
        )
    )
