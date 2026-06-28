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
        types.SimpleNamespace(BatchedInferencePipeline=object, WhisperModel=object),
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
    monkeypatch.setattr(transcriber, "_wrap_model", lambda model, _pipeline_class: model)

    assert transcriber._load_model() is cpu_model
    assert calls == [("cuda", "float16"), ("cpu", "int8")]
    assert transcriber.runtime_notice is not None


def test_model_load_reports_unsupported_proxy_cleanly(monkeypatch) -> None:
    monkeypatch.setitem(
        sys.modules,
        "faster_whisper",
        types.SimpleNamespace(BatchedInferencePipeline=object, WhisperModel=object),
    )
    transcriber = WhisperTranscriber(AppConfig())

    def fake_create_model(_model_class, device: str, compute_type: str):
        raise RuntimeError("Unknown scheme for proxy URL URL('socks4://127.0.0.1:10808')")

    monkeypatch.setattr(transcriber, "_create_model", fake_create_model)

    try:
        transcriber._load_model()
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected proxy model load failure.")

    assert "configured network proxy is unsupported" in message
    assert "socks4://127.0.0.1:10808" not in message


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
    monkeypatch.setattr("transcripio.transcriber.has_cuda_capable_gpu", lambda _output_dir: True)

    transcriber._create_model(FakeModel, device="cuda", compute_type="float16")

    assert configure_calls == 2
    assert install_calls == 1


def test_cuda_model_does_not_auto_install_runtime_without_gpu(monkeypatch) -> None:
    transcriber = WhisperTranscriber(AppConfig(device="cuda", auto_install_cuda_runtime=True))
    install_calls = 0

    class FakeModel:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

    def fake_install_cuda_runtime_packages():
        nonlocal install_calls
        install_calls += 1

    monkeypatch.setattr(
        "transcripio.transcriber.configure_cuda_dll_paths",
        lambda: CudaRuntimeStatus(
            is_ready=False,
            missing_dlls=("cublas64_12.dll",),
            dll_dirs=(),
        ),
    )
    monkeypatch.setattr("transcripio.transcriber.has_cuda_capable_gpu", lambda _output_dir: False)
    monkeypatch.setattr(
        "transcripio.transcriber.install_cuda_runtime_packages",
        fake_install_cuda_runtime_packages,
    )

    try:
        transcriber._create_model(FakeModel, device="cuda", compute_type="float16")
    except RuntimeError as exc:
        assert "no CUDA-capable NVIDIA GPU" in str(exc)
    else:
        raise AssertionError("Expected missing GPU to stop CUDA runtime auto-install.")

    assert install_calls == 0


def test_batched_transcribe_passes_batch_size(monkeypatch, tmp_path: Path) -> None:
    transcriber = WhisperTranscriber(
        AppConfig(use_batched_inference=True, batch_size=12, beam_size=1, best_of=1)
    )
    captured_options = {}

    class FakeInfo:
        duration = 1.0
        language = "en"

    class FakeModel:
        def transcribe(self, _audio_path, **options):
            captured_options.update(options)
            return iter(()), FakeInfo()

    monkeypatch.setattr(transcriber, "_load_model", lambda: FakeModel())

    transcriber.transcribe(tmp_path / "audio.wav")

    assert captured_options["batch_size"] == 12
    assert captured_options["beam_size"] == 1
    assert captured_options["best_of"] == 1


def test_transcribe_passes_quality_options_and_collects_words(
    monkeypatch,
    tmp_path: Path,
) -> None:
    transcriber = WhisperTranscriber(
        AppConfig(
            initial_prompt="Names: Transcripio, Pavel.",
            hotwords="Transcripio Pavel",
            word_timestamps=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.4,
            language_detection_threshold=0.7,
            hallucination_silence_threshold=1.5,
        )
    )
    captured_options = {}

    class FakeWord:
        start = 0.0
        end = 0.5
        word = " hello"
        probability = 0.9

    class FakeSegment:
        start = 0.0
        end = 0.5
        text = " hello"
        words = [FakeWord()]

    class FakeInfo:
        duration = 1.0
        language = "en"

    class FakeModel:
        def transcribe(self, _audio_path, **options):
            captured_options.update(options)
            return iter([FakeSegment()]), FakeInfo()

    monkeypatch.setattr(transcriber, "_load_model", lambda: FakeModel())

    segments, _language, _duration = transcriber.transcribe(tmp_path / "audio.wav")

    assert captured_options["initial_prompt"] == "Names: Transcripio, Pavel."
    assert captured_options["hotwords"] == "Transcripio Pavel"
    assert captured_options["word_timestamps"] is True
    assert captured_options["condition_on_previous_text"] is False
    assert captured_options["no_speech_threshold"] == 0.4
    assert captured_options["language_detection_threshold"] == 0.7
    assert captured_options["hallucination_silence_threshold"] == 1.5
    assert segments[0].words[0].text == "hello"
    assert segments[0].words[0].probability == 0.9
