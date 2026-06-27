from __future__ import annotations

import subprocess
from pathlib import Path

from transcripio.config import AppConfig
from transcripio.health import (
    HealthCheck,
    check_ffmpeg,
    check_whisper_model,
    run_environment_checks,
)


def test_check_ffmpeg_reports_available_binary(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        assert command == ["ffmpeg", "-version"]
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(command, 0, stdout="ffmpeg version 1", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    check = check_ffmpeg("ffmpeg")

    assert check == HealthCheck("ffmpeg", "ok", "ffmpeg is available.")


def test_check_ffmpeg_reports_failed_binary(monkeypatch) -> None:
    def fake_run(command, **_kwargs):
        return subprocess.CompletedProcess(command, 1, stdout="", stderr="bad binary\nmore")

    monkeypatch.setattr(subprocess, "run", fake_run)

    check = check_ffmpeg("ffmpeg")

    assert check.status == "error"
    assert check.message == "ffmpeg check failed: bad binary"


def test_check_whisper_model_reports_missing_local_path(tmp_path: Path) -> None:
    check = check_whisper_model(str(tmp_path / "missing-model"), local_files_only=True)

    assert check.status == "error"
    assert "was not found" in check.message


def test_check_whisper_model_treats_huggingface_repo_as_model_name() -> None:
    check = check_whisper_model("Systran/faster-whisper-small", local_files_only=False)

    assert check.status == "ok"


def test_run_environment_checks_includes_writable_dirs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "transcripio.health.check_ffmpeg",
        lambda _path: HealthCheck("ffmpeg", "ok", "ok"),
    )

    checks = run_environment_checks(
        AppConfig(
            whisper_model="small",
            output_dir=tmp_path / "output",
            history_dir=tmp_path / "history",
        )
    )

    checks_by_name = {check.name: check for check in checks}
    assert checks_by_name["Output directory"].status == "ok"
    assert checks_by_name["History directory"].status == "ok"
    assert (tmp_path / "output").exists()
    assert (tmp_path / "history").exists()
