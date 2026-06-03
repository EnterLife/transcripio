import sys
import types
from pathlib import Path

from transcripio.config import AppConfig
from transcripio.cuda_runtime import CudaRuntimeStatus
from transcripio.transcriber import WhisperTranscriber


def test_cuda_cublas_error_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(WhisperModel=object),
    )
    transcriber = WhisperTranscriber(AppConfig(device="cuda", compute_type="float16"))
    calls: list[tuple[str, str]] = []
    cpu_model = object()

    def fake_create_model(_model_class, device: str, compute_type: str):
        calls.append((device, compute_type))
        if device == "cuda":
            raise RuntimeError("Library cublas64_12.dll is not found or cannot be loaded")
        return cpu_model

    monkeypatch.setattr(transcriber, "_create_model", fake_create_model)

    assert transcriber._load_model() is cpu_model
    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    assert transcriber.runtime_notice is not None


def test_cuda_model_auto_installs_missing_runtime(monkeypatch) -> None:
    transcriber = WhisperTranscriber(AppConfig(device="cuda", auto_install_cuda_runtime=True))
    configure_calls = 0
    install_calls = 0

    def fake_configure_cuda_dll_paths():
        nonlocal configure_calls
        configure_calls += 1
        if configure_calls == 1:
            return CudaRuntimeStatus(
                is_ready=False,
                missing_dlls=("cublas64_12.dll",),
                dll_dirs=(),
            )
        return CudaRuntimeStatus(
            is_ready=True,
            missing_dlls=(),
            dll_dirs=(Path("nvidia/cublas/bin"),),
        )

    def fake_install_cuda_runtime_packages():
        nonlocal install_calls
        install_calls += 1
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    class FakeModel:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    monkeypatch.setattr(
        "transcripio.transcriber.configure_cuda_dll_paths",
        fake_configure_cuda_dll_paths,
    )
    monkeypatch.setattr(
        "transcripio.transcriber.install_cuda_runtime_packages",
        fake_install_cuda_runtime_packages,
    )

    transcriber._create_model(FakeModel, device="cuda", compute_type="float16")

    assert configure_calls == 2
    assert install_calls == 1
