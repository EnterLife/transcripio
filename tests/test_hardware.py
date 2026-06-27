from __future__ import annotations

import subprocess
from pathlib import Path

from transcripio.config import AppConfig
from transcripio.hardware import (
    GpuInfo,
    HardwareProfile,
    _detect_nvidia_smi_gpus,
    recommend_transcription_profiles,
)


def test_recommend_profiles_uses_cuda_when_gpu_is_available() -> None:
    hardware = HardwareProfile(
        cpu_count=12,
        memory_total_mb=32768,
        disk_free_mb=100000,
        gpus=(GpuInfo(name="RTX", memory_total_mb=12000, cuda_available=True),),
    )

    profiles = recommend_transcription_profiles(
        hardware,
        downloaded_models={
            "Systran/faster-whisper-small",
            "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        },
    )

    balanced = profiles[1]
    assert balanced.device == "cuda"
    assert balanced.compute_type == "float16"
    assert balanced.use_batched_inference is True
    assert balanced.whisper_model == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert balanced.local_files_only is True


def test_recommend_profiles_uses_cpu_safe_defaults_without_gpu() -> None:
    hardware = HardwareProfile(
        cpu_count=8,
        memory_total_mb=16384,
        disk_free_mb=100000,
        gpus=(),
    )

    profiles = recommend_transcription_profiles(hardware, downloaded_models=set())

    fast = profiles[0]
    assert fast.device == "cpu"
    assert fast.compute_type == "int8"
    assert fast.use_batched_inference is False
    assert fast.beam_size == 1
    assert fast.local_files_only is False


def test_recommendation_can_apply_to_app_config() -> None:
    hardware = HardwareProfile(cpu_count=8, memory_total_mb=16000, disk_free_mb=100000)
    recommendation = recommend_transcription_profiles(
        hardware,
        downloaded_models={"Systran/faster-whisper-small"},
    )[0]
    config = recommendation.apply_to(
        AppConfig(
            language="ru",
            diarization_model_path="models/pyannote/config.yaml",
            output_dir=Path("out"),
            history_dir=Path("history"),
        )
    )

    assert config.language == "ru"
    assert config.diarization_model_path == "models/pyannote/config.yaml"
    assert config.output_dir == Path("out")
    assert config.whisper_model == "small"


def test_detect_nvidia_smi_gpus_parses_query_output(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        assert command[0] == "nvidia-smi"
        assert kwargs["timeout"] == 5
        return subprocess.CompletedProcess(
            command,
            0,
            stdout="NVIDIA GeForce RTX 4070, 12282\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert _detect_nvidia_smi_gpus() == (
        GpuInfo(name="NVIDIA GeForce RTX 4070", memory_total_mb=12282, cuda_available=True),
    )
