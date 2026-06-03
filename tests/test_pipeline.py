from pathlib import Path

from transcripio.config import AppConfig
from transcripio.models import TranscriptSegment
from transcripio.pipeline import TranscriptionPipeline


def test_pipeline_keeps_transcript_when_diarization_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "call.mp3"
    input_path.write_bytes(b"media")
    audio_path = tmp_path / "call.wav"
    audio_path.write_bytes(b"wav")

    monkeypatch.setattr("transcripio.pipeline.ensure_audio_wav", lambda *_args: audio_path)

    class FakeTranscriber:
        runtime_notice = None

        def __init__(self, _config) -> None:
            pass

        def transcribe(self, _audio_path, on_segment=None):
            segment = TranscriptSegment(start=0.0, end=1.0, text="hello")
            if on_segment:
                on_segment(segment, 1.0)
            return [segment], "en", 1.0

    class FakeDiarizer:
        def __init__(self, _config) -> None:
            pass

        def is_enabled(self) -> bool:
            return True

        def diarize(self, _audio_path):
            raise RuntimeError("Could not load libtorchcodec.")

    monkeypatch.setattr("transcripio.pipeline.WhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr("transcripio.pipeline.LocalPyannoteDiarizer", FakeDiarizer)

    messages: list[str] = []
    config = AppConfig(
        diarization_model_path="models/pyannote/config.yaml",
        output_dir=tmp_path / "output",
        history_dir=tmp_path / "history",
    )
    result = TranscriptionPipeline(config).run(
        input_path,
        on_step=lambda message, _value: messages.append(message),
    )

    assert result.segments[0].text == "hello"
    assert result.segments[0].speaker is None
    assert any("Speaker assignment skipped" in message for message in messages)
