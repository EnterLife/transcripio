import json
from pathlib import Path

import pytest

from transcripio.config import SettingsError, load_settings


def test_load_settings_from_json(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(
        json.dumps(
            {
                "ui": {
                    "page_title": "Local Transcriber",
                    "upload_types": [".mp3", "wav"],
                },
                "transcription": {
                    "whisper_model": "models/whisper-small",
                    "language": "",
                    "diarization_repo_id": "pyannote/test",
                    "diarization_output_dir": "tmp/pyannote",
                    "ffmpeg_path": "bin/ffmpeg.exe",
                    "use_batched_inference": True,
                    "batch_size": 16,
                    "beam_size": 2,
                    "best_of": 2,
                    "cpu_threads": 4,
                    "num_workers": 2,
                    "vad_filter": False,
                },
                "storage": {
                    "output_dir": "tmp/output",
                    "history_dir": "tmp/history",
                },
            }
        ),
        encoding="utf-8",
    )

    settings = load_settings(settings_path)

    assert settings.page_title == "Local Transcriber"
    assert settings.upload_types == ("mp3", "wav")
    assert settings.config.whisper_model == "models/whisper-small"
    assert settings.config.language is None
    assert settings.config.diarization_repo_id == "pyannote/test"
    assert settings.config.diarization_output_dir == Path("tmp/pyannote")
    assert settings.config.ffmpeg_path == "bin/ffmpeg.exe"
    assert settings.config.use_batched_inference is True
    assert settings.config.batch_size == 16
    assert settings.config.beam_size == 2
    assert settings.config.best_of == 2
    assert settings.config.cpu_threads == 4
    assert settings.config.num_workers == 2
    assert settings.config.vad_filter is False
    assert settings.config.output_dir == Path("tmp/output")
    assert settings.config.history_dir == Path("tmp/history")


def test_load_settings_rejects_invalid_sections(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"ui": []}), encoding="utf-8")

    with pytest.raises(SettingsError):
        load_settings(settings_path)
