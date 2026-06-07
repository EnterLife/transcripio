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
                "llm": {
                    "default_provider": "LM Studio",
                    "providers": [
                        {
                            "name": "LM Studio",
                            "base_url": "http://localhost:1234/v1",
                            "model": "qwen2.5",
                            "api_key_env": "LM_STUDIO_API_KEY",
                            "requires_api_key": False,
                            "temperature": 0.1,
                            "max_tokens": 500,
                        }
                    ],
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
    assert settings.default_llm_provider == "LM Studio"
    assert settings.llm_providers[0].name == "LM Studio"
    assert settings.llm_providers[0].base_url == "http://localhost:1234/v1"
    assert settings.llm_providers[0].model == "qwen2.5"
    assert settings.llm_providers[0].api_key_env == "LM_STUDIO_API_KEY"
    assert settings.llm_providers[0].requires_api_key is False
    assert settings.llm_providers[0].temperature == 0.1
    assert settings.llm_providers[0].max_tokens == 500


def test_load_settings_rejects_invalid_sections(tmp_path: Path) -> None:
    settings_path = tmp_path / "settings.json"
    settings_path.write_text(json.dumps({"ui": []}), encoding="utf-8")

    with pytest.raises(SettingsError):
        load_settings(settings_path)
