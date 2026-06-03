import sys
import types

from transcripio.config import AppConfig
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
