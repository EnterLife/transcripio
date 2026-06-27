from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from transcripio.config import AppConfig
from transcripio.model_catalog import SHORT_MODEL_REPOS


@dataclass(frozen=True, slots=True)
class GpuInfo:
    name: str
    memory_total_mb: int | None = None
    cuda_available: bool = False


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    cpu_count: int
    memory_total_mb: int | None
    disk_free_mb: int
    gpus: tuple[GpuInfo, ...] = ()

    @property
    def has_cuda_gpu(self) -> bool:
        return any(gpu.cuda_available for gpu in self.gpus)

    @property
    def best_gpu_memory_mb(self) -> int | None:
        memory_values = [gpu.memory_total_mb for gpu in self.gpus if gpu.memory_total_mb]
        return max(memory_values) if memory_values else None


@dataclass(frozen=True, slots=True)
class TranscriptionRecommendation:
    name: str
    description: str
    whisper_model: str
    device: str
    compute_type: str
    use_batched_inference: bool
    batch_size: int
    beam_size: int
    best_of: int
    cpu_threads: int
    num_workers: int
    vad_filter: bool
    local_files_only: bool

    def apply_to(self, config: AppConfig) -> AppConfig:
        return AppConfig(
            whisper_model=self.whisper_model,
            device=self.device,
            compute_type=self.compute_type,
            language=config.language,
            diarization_model_path=config.diarization_model_path,
            diarization_repo_id=config.diarization_repo_id,
            diarization_output_dir=config.diarization_output_dir,
            ffmpeg_path=config.ffmpeg_path,
            output_dir=config.output_dir,
            history_dir=config.history_dir,
            local_files_only=self.local_files_only,
            allow_cpu_fallback=config.allow_cpu_fallback,
            auto_install_cuda_runtime=config.auto_install_cuda_runtime,
            use_batched_inference=self.use_batched_inference,
            batch_size=self.batch_size,
            beam_size=self.beam_size,
            best_of=self.best_of,
            cpu_threads=self.cpu_threads,
            num_workers=self.num_workers,
            vad_filter=self.vad_filter,
        )


def detect_hardware_profile(output_dir: Path = Path("data/output")) -> HardwareProfile:
    return HardwareProfile(
        cpu_count=os.cpu_count() or 1,
        memory_total_mb=_total_memory_mb(),
        disk_free_mb=shutil.disk_usage(output_dir if output_dir.exists() else Path.cwd()).free
        // 1024
        // 1024,
        gpus=_detect_gpus(),
    )


def recommend_transcription_profiles(
    hardware: HardwareProfile,
    downloaded_models: set[str],
) -> tuple[TranscriptionRecommendation, ...]:
    fast_model = _best_available_model(
        downloaded_models,
        ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "small", "base", "tiny"),
        fallback="small",
    )
    balanced_model = _best_available_model(
        downloaded_models,
        ("mobiuslabsgmbh/faster-whisper-large-v3-turbo", "medium", "small"),
        fallback="small",
    )
    quality_model = _quality_model(downloaded_models)

    gpu_memory_mb = hardware.best_gpu_memory_mb or 0
    use_cuda = hardware.has_cuda_gpu
    cpu_threads = 0 if hardware.cpu_count <= 4 else max(1, hardware.cpu_count - 2)

    if use_cuda:
        fast_batch = 16 if gpu_memory_mb >= 12000 else 8
        balanced_batch = 12 if gpu_memory_mb >= 12000 else 8
        quality_batch = 8 if gpu_memory_mb >= 10000 else 4
        return (
            TranscriptionRecommendation(
                name="Fast",
                description="Prioritizes speed for drafts and long queues.",
                whisper_model=fast_model,
                device="cuda",
                compute_type="float16",
                use_batched_inference=True,
                batch_size=fast_batch,
                beam_size=1,
                best_of=1,
                cpu_threads=cpu_threads,
                num_workers=1,
                vad_filter=True,
                local_files_only=fast_model in downloaded_models or _short_repo(fast_model) in downloaded_models,
            ),
            TranscriptionRecommendation(
                name="Balanced",
                description="Good default for local work: fast, but less aggressive.",
                whisper_model=balanced_model,
                device="cuda",
                compute_type="float16",
                use_batched_inference=True,
                batch_size=balanced_batch,
                beam_size=2,
                best_of=2,
                cpu_threads=cpu_threads,
                num_workers=1,
                vad_filter=True,
                local_files_only=balanced_model in downloaded_models
                or _short_repo(balanced_model) in downloaded_models,
            ),
            TranscriptionRecommendation(
                name="Quality",
                description="Uses the strongest practical local model and slower decoding.",
                whisper_model=quality_model,
                device="cuda",
                compute_type="float16",
                use_batched_inference=True,
                batch_size=quality_batch,
                beam_size=5,
                best_of=5,
                cpu_threads=cpu_threads,
                num_workers=1,
                vad_filter=True,
                local_files_only=quality_model in downloaded_models
                or _short_repo(quality_model) in downloaded_models,
            ),
        )

    return (
        TranscriptionRecommendation(
            name="Fast",
            description="CPU-friendly draft mode.",
            whisper_model=fast_model,
            device="cpu",
            compute_type="int8",
            use_batched_inference=False,
            batch_size=8,
            beam_size=1,
            best_of=1,
            cpu_threads=cpu_threads,
            num_workers=1,
            vad_filter=True,
            local_files_only=fast_model in downloaded_models or _short_repo(fast_model) in downloaded_models,
        ),
        TranscriptionRecommendation(
            name="Balanced",
            description="CPU-safe default with a little more decoding quality.",
            whisper_model=balanced_model,
            device="cpu",
            compute_type="int8",
            use_batched_inference=False,
            batch_size=8,
            beam_size=2,
            best_of=2,
            cpu_threads=cpu_threads,
            num_workers=1,
            vad_filter=True,
            local_files_only=balanced_model in downloaded_models
            or _short_repo(balanced_model) in downloaded_models,
        ),
        TranscriptionRecommendation(
            name="Quality",
            description="Best CPU quality when you can wait longer.",
            whisper_model=quality_model,
            device="cpu",
            compute_type="int8",
            use_batched_inference=False,
            batch_size=8,
            beam_size=5,
            best_of=5,
            cpu_threads=cpu_threads,
            num_workers=1,
            vad_filter=True,
            local_files_only=quality_model in downloaded_models
            or _short_repo(quality_model) in downloaded_models,
        ),
    )


def _detect_gpus() -> tuple[GpuInfo, ...]:
    nvidia_gpus = _detect_nvidia_smi_gpus()
    if nvidia_gpus:
        return nvidia_gpus

    try:
        import torch
    except ImportError:
        return ()

    if not torch.cuda.is_available():
        return ()

    gpus: list[GpuInfo] = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        gpus.append(
            GpuInfo(
                name=props.name,
                memory_total_mb=int(props.total_memory // 1024 // 1024),
                cuda_available=True,
            )
        )
    return tuple(gpus)


def _detect_nvidia_smi_gpus() -> tuple[GpuInfo, ...]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ()

    if completed.returncode != 0:
        return ()

    gpus: list[GpuInfo] = []
    for line in completed.stdout.splitlines():
        name, separator, memory_text = line.partition(",")
        if not separator:
            continue
        try:
            memory_mb = int(memory_text.strip())
        except ValueError:
            memory_mb = None
        gpus.append(GpuInfo(name=name.strip(), memory_total_mb=memory_mb, cuda_available=True))
    return tuple(gpus)


def _total_memory_mb() -> int | None:
    if os.name != "nt":
        return None

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys // 1024 // 1024)


def _best_available_model(
    downloaded_models: set[str],
    preferred_models: tuple[str, ...],
    fallback: str,
) -> str:
    for model in preferred_models:
        if model in downloaded_models or _short_repo(model) in downloaded_models:
            return model
    return fallback


def _quality_model(downloaded_models: set[str]) -> str:
    for model in (
        "large-v3",
        "Systran/faster-whisper-large-v3",
        "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
        "medium",
        "small",
    ):
        if model in downloaded_models or _short_repo(model) in downloaded_models:
            return model
    return "large-v3"


def _short_repo(model: str) -> str | None:
    return SHORT_MODEL_REPOS.get(model)
