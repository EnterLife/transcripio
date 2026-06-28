from __future__ import annotations

import sys

from transcripio.cuda_runtime import (
    CUDA_RUNTIME_PACKAGES,
    configure_cuda_dll_paths,
    has_cuda_capable_gpu,
    install_cuda_runtime_packages,
)


def install_cuda_runtime_if_needed() -> bool:
    if not has_cuda_capable_gpu():
        print("No CUDA-capable NVIDIA GPU detected. Skipping CUDA runtime packages.")
        return False

    cuda_status = configure_cuda_dll_paths()
    if cuda_status.is_ready:
        print("CUDA runtime DLLs are already available.")
        return False

    print("CUDA-capable NVIDIA GPU detected.")
    print("Installing CUDA runtime packages: " + ", ".join(CUDA_RUNTIME_PACKAGES))
    completed = install_cuda_runtime_packages()
    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"Could not install CUDA runtime packages: {details}")

    cuda_status = configure_cuda_dll_paths()
    if not cuda_status.is_ready:
        missing = ", ".join(cuda_status.missing_dlls)
        raise RuntimeError(f"CUDA runtime packages installed, but DLLs are still missing: {missing}")

    print("CUDA runtime packages installed.")
    return True


def main() -> int:
    try:
        install_cuda_runtime_if_needed()
    except Exception as exc:  # noqa: BLE001 - setup should print a concise failure.
        print(f"CUDA runtime setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
