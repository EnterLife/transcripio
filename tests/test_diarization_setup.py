import sys
import types
from pathlib import Path

import pytest

from transcripio.diarization_setup import download_diarization_pipeline


def test_download_diarization_pipeline_requires_token(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Missing Hugging Face token"):
        download_diarization_pipeline(
            repo_id="pyannote/speaker-diarization-3.1",
            output_dir=tmp_path,
            token="",
        )


def test_download_diarization_pipeline_returns_config_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(repo_id: str, local_dir: str, token: str) -> str:
        assert repo_id == "pyannote/speaker-diarization-3.1"
        assert token == "hf_token"
        snapshot_path = Path(local_dir)
        snapshot_path.mkdir(parents=True, exist_ok=True)
        (snapshot_path / "config.yaml").write_text("pipeline: pyannote", encoding="utf-8")
        return str(snapshot_path)

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    result = download_diarization_pipeline(
        repo_id="pyannote/speaker-diarization-3.1",
        output_dir=tmp_path / "pyannote",
        token="hf_token",
    )

    assert result.config_path == tmp_path / "pyannote" / "config.yaml"
