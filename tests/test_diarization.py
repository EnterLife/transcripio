from pathlib import Path
import sys
import types
import wave

import torch

from transcripio.config import AppConfig
from transcripio.diarization import LocalPyannoteDiarizer, assign_speakers
from transcripio.models import DiarizationSegment, TranscriptSegment


def test_assign_speakers_uses_largest_overlap() -> None:
    transcript = [TranscriptSegment(start=0.0, end=10.0, text="hello")]
    diarization = [
        DiarizationSegment(start=0.0, end=2.0, speaker="SPEAKER_00"),
        DiarizationSegment(start=2.0, end=10.0, speaker="SPEAKER_01"),
    ]

    result = assign_speakers(transcript, diarization)

    assert result[0].speaker == "SPEAKER_01"


def test_diarizer_passes_preloaded_waveform_to_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("pipeline: pyannote", encoding="utf-8")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"wav")

    received_file = {}

    class FakeTurn:
        start = 0.0
        end = 1.0

    class FakeAnnotation:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield FakeTurn(), None, "SPEAKER_00"

    class FakePipeline:
        def __call__(self, file):
            received_file.update(file)
            return FakeAnnotation()

    fake_pipeline = FakePipeline()

    class FakePipelineFactory:
        @staticmethod
        def from_pretrained(_path: str):
            return fake_pipeline

    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        types.SimpleNamespace(Pipeline=FakePipelineFactory),
    )
    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        types.SimpleNamespace(load=lambda _path: (torch.ones(2, 16000), 16000)),
    )

    diarizer = LocalPyannoteDiarizer(AppConfig(diarization_model_path=str(config_path)))

    segments = diarizer.diarize(audio_path)

    assert segments == [DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_00")]
    assert set(received_file) == {"waveform", "sample_rate", "uri"}
    assert received_file["waveform"].shape == (1, 16000)
    assert received_file["sample_rate"] == 16000
    assert received_file["uri"] == "audio"


def test_diarizer_accepts_community_pipeline_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("pipeline: pyannote", encoding="utf-8")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"wav")

    class FakeTurn:
        start = 0.0
        end = 1.0

    class FakeAnnotation:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            yield FakeTurn(), None, "SPEAKER_00"

    class FakeCommunityOutput:
        speaker_diarization = FakeAnnotation()

    class FakePipeline:
        def __call__(self, _file):
            return FakeCommunityOutput()

    fake_pipeline = FakePipeline()

    class FakePipelineFactory:
        @staticmethod
        def from_pretrained(_path: str):
            return fake_pipeline

    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        types.SimpleNamespace(Pipeline=FakePipelineFactory),
    )
    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        types.SimpleNamespace(load=lambda _path: (torch.ones(1, 16000), 16000)),
    )

    diarizer = LocalPyannoteDiarizer(AppConfig(diarization_model_path=str(config_path)))

    assert diarizer.diarize(audio_path) == [
        DiarizationSegment(start=0.0, end=1.0, speaker="SPEAKER_00")
    ]


def test_diarizer_falls_back_to_standard_wav_reader_when_torchcodec_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("pipeline: pyannote", encoding="utf-8")
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes((b"\x00\x00") * 160)

    received_file = {}

    class FakeAnnotation:
        def itertracks(self, yield_label: bool = False):
            assert yield_label is True
            return iter(())

    class FakePipeline:
        def __call__(self, file):
            received_file.update(file)
            return FakeAnnotation()

    fake_pipeline = FakePipeline()

    class FakePipelineFactory:
        @staticmethod
        def from_pretrained(_path: str):
            return fake_pipeline

    def fail_with_torchcodec(_path):
        raise RuntimeError("Could not load libtorchcodec.")

    monkeypatch.setitem(
        sys.modules,
        "pyannote.audio",
        types.SimpleNamespace(Pipeline=FakePipelineFactory),
    )
    monkeypatch.setitem(
        sys.modules,
        "torchaudio",
        types.SimpleNamespace(load=fail_with_torchcodec),
    )

    diarizer = LocalPyannoteDiarizer(AppConfig(diarization_model_path=str(config_path)))

    assert diarizer.diarize(audio_path) == []
    assert received_file["waveform"].shape == (1, 160)
    assert received_file["sample_rate"] == 16000
