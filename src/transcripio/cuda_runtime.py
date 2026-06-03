from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from dataclasses import dataclass
from pathlib import Path


CUDA_RUNTIME_PACKAGES = (
    "nvidia-cublas-cu12",
    "nvidia-cudnn-cu12==9.*",
)

REQUIRED_CUDA_DLLS = (
    "cublas64_12.dll",
    "cudnn64_9.dll",
)


@dataclass(frozen=True, slots=True)
class CudaRuntimeStatus:
    is_ready: bool
    missing_dlls: tuple[str, ...]
    dll_dirs: tuple[Path, ...]


def configure_cuda_dll_paths() -> CudaRuntimeStatus:
    dll_dirs = discover_nvidia_dll_dirs()
    for dll_dir in dll_dirs:
        dll_dir_text = str(dll_dir)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir_text)
        path_parts = os.environ.get("PATH", "").split(os.pathsep)
        if dll_dir_text not in path_parts:
            os.environ["PATH"] = dll_dir_text + os.pathsep + os.environ.get("PATH", "")

    missing_dlls = tuple(dll_name for dll_name in REQUIRED_CUDA_DLLS if not _can_find_dll(dll_name))
    return CudaRuntimeStatus(
        is_ready=not missing_dlls,
        missing_dlls=missing_dlls,
        dll_dirs=dll_dirs,
    )


def discover_nvidia_dll_dirs() -> tuple[Path, ...]:
    nvidia_root = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    if not nvidia_root.exists():
        return ()

    dll_dirs: list[Path] = []
    for path in nvidia_root.glob("*"):
        if not path.is_dir():
            continue
        bin_dir = path / "bin"
        if bin_dir.exists() and any(bin_dir.glob("*.dll")):
            dll_dirs.append(bin_dir)
        lib_dir = path / "lib"
        if lib_dir.exists() and any(lib_dir.glob("*.dll")):
            dll_dirs.append(lib_dir)
    return tuple(sorted(dll_dirs))


def install_cuda_runtime_packages() -> subprocess.CompletedProcess[str]:
    command = [sys.executable, "-m", "pip", "install", *CUDA_RUNTIME_PACKAGES]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def _can_find_dll(dll_name: str) -> bool:
    search_dirs = [Path.cwd(), *discover_nvidia_dll_dirs()]
    search_dirs.extend(Path(part) for part in os.environ.get("PATH", "").split(os.pathsep) if part)
    return any((search_dir / dll_name).exists() for search_dir in search_dirs)
