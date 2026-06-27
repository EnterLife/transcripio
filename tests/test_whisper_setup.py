from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from transcripio.whisper_setup import (
    default_whisper_output_dir,
    download_whisper_model,
    resolve_whisper_repo_id,
)


def test_resolve_whisper_repo_id_expands_short_names() -> None:
    assert resolve_whisper_repo_id("small") == "Systran/faster-whisper-small"
    assert resolve_whisper_repo_id("custom/repo") == "custom/repo"


def test_default_whisper_output_dir_is_repo_based() -> None:
    assert default_whisper_output_dir("Systran/faster-whisper-small", Path("models")) == Path(
        "models/Systran--faster-whisper-small"
    )


def test_download_whisper_model_returns_local_model_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_snapshot_download(repo_id: str, local_dir: str, token: str | None) -> str:
        assert repo_id == "Systran/faster-whisper-small"
        assert token == "hf_token"
        path = Path(local_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "model.bin").write_bytes(b"model")
        (path / "config.json").write_text("{}", encoding="utf-8")
        return str(path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    result = download_whisper_model("small", output_dir=tmp_path / "model", token="hf_token")

    assert result.repo_id == "Systran/faster-whisper-small"
    assert result.model_path == tmp_path / "model"


def test_download_whisper_model_requires_model_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    def fake_snapshot_download(**kwargs) -> str:
        path = Path(kwargs["local_dir"])
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    with pytest.raises(RuntimeError, match="model.bin and config.json"):
        download_whisper_model("small", output_dir=tmp_path / "model")
