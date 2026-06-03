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
                    "ffmpeg_path": "bin/ffmpeg.exe",
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
    assert settings.config.ffmpeg_path == "bin/ffmpeg.exe"
    assert settings.config.output_dir == Path("tmp/output")
    assert settings.config.history_dir == Path("tmp/history")


def test_load_settings_rejects_invalid_sections(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"ui": []}), encoding="utf-8")

    with pytest.raises(SettingsError):
        load_settings(settings_path)
