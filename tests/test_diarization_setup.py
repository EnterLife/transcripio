import sys
import types
from pathlib import Path

import pytest

from transcripio.diarization_setup import (
    COMMUNITY_DIARIZATION_ACCESS_REPOS,
    LEGACY_DIARIZATION_ACCESS_REPOS,
    check_huggingface_diarization_access,
    download_diarization_pipeline,
    required_access_repos,
)


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


def test_download_diarization_pipeline_explains_gated_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_snapshot_download(**_kwargs) -> str:
        raise RuntimeError("403 Client Error. Cannot access gated repo.")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    with pytest.raises(RuntimeError, match="accept the model terms"):
        download_diarization_pipeline(
            repo_id="pyannote/speaker-diarization-3.1",
            output_dir=tmp_path / "pyannote",
            token="hf_token",
        )


def test_check_huggingface_diarization_access_reports_each_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeApi:
        def whoami(self, token: str) -> dict[str, str]:
            assert token == "hf_token"
            return {"name": "test-user"}

    checked_repos: list[str] = []

    def fake_hf_hub_download(**kwargs) -> str:
        checked_repos.append(kwargs["repo_id"])
        if kwargs["repo_id"] == "pyannote/segmentation-3.0":
            raise RuntimeError("403 Client Error. Cannot access gated repo.")
        return "ok"

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda: FakeApi(), hf_hub_download=fake_hf_hub_download),
    )

    result = check_huggingface_diarization_access(
        "hf_token",
        repo_id="pyannote/speaker-diarization-3.1",
    )

    assert result.username == "test-user"
    assert checked_repos == list(LEGACY_DIARIZATION_ACCESS_REPOS)
    assert result.repos[0].has_access is False
    assert result.repos[1].has_access is True


def test_required_access_repos_uses_community_repo_only() -> None:
    assert required_access_repos("pyannote/speaker-diarization-community-1") == (
        "pyannote/speaker-diarization-community-1",
    )
    assert COMMUNITY_DIARIZATION_ACCESS_REPOS == ("pyannote/speaker-diarization-community-1",)
