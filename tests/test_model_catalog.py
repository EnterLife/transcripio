from pathlib import Path

from transcripio.model_catalog import (
    discover_cached_hf_model_repos,
    discover_local_diarization_config_paths,
    discover_local_whisper_model_paths,
    list_whisper_model_options,
)


def test_discover_cached_hf_model_repos(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "models--Systran--faster-whisper-small" / "snapshots" / "abc"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "config.json").write_text("{}", encoding="utf-8")

    repos = discover_cached_hf_model_repos(tmp_path)

    assert repos == {"Systran/faster-whisper-small"}


def test_discover_local_whisper_model_paths(tmp_path: Path) -> None:
    model_dir = tmp_path / "whisper-small"
    model_dir.mkdir()
    (model_dir / "model.bin").write_bytes(b"model")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")

    assert discover_local_whisper_model_paths(tmp_path) == [model_dir]


def test_model_options_mark_short_cached_model_downloaded(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "models--Systran--faster-whisper-small" / "snapshots" / "abc"
    snapshot_dir.mkdir(parents=True)

    options = list_whisper_model_options(("small", "medium"), hf_cache_dir=tmp_path)

    status_by_value = {option.value: option.is_downloaded for option in options}
    assert status_by_value["small"] is True
    assert status_by_value["medium"] is False


def test_model_options_do_not_duplicate_configured_cached_repos(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "models--Systran--faster-whisper-small" / "snapshots" / "abc"
    snapshot_dir.mkdir(parents=True)

    options = list_whisper_model_options(
        ("Systran/faster-whisper-small",),
        hf_cache_dir=tmp_path,
    )

    assert [option.value for option in options] == ["Systran/faster-whisper-small"]


def test_discover_local_diarization_config_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "pyannote-speaker-diarization" / "config.yaml"
    config_path.parent.mkdir()
    config_path.write_text("pipeline: pyannote.audio.Pipeline", encoding="utf-8")

    assert discover_local_diarization_config_paths(tmp_path) == [config_path]
