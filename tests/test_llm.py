import pytest

from transcripio.config import LlmProviderConfig
from transcripio.llm import LlmError, OpenAICompatibleLlm
from transcripio.models import TranscriptSegment


def test_openai_compatible_llm_sends_transcript_request() -> None:
    captured: dict[str, object] = {}

    class FakeCompletions:
        @staticmethod
        def create(**kwargs):
            captured.update(kwargs)
            message = type("Message", (), {"content": "summary"})
            choice = type("Choice", (), {"message": message})
            return type("Response", (), {"choices": [choice]})

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    def client_factory(**kwargs):
        captured["client_kwargs"] = kwargs
        return FakeClient()

    provider = LlmProviderConfig(
        name="LM Studio",
        base_url="http://localhost:1234/v1",
        model="local-model",
        requires_api_key=False,
        temperature=0.1,
        max_tokens=500,
    )
    segments = [TranscriptSegment(start=0.0, end=1.0, speaker="A", text="hello")]

    result = OpenAICompatibleLlm(
        provider,
        client_factory=client_factory,
    ).generate_transcript_note(segments, "Summarize")

    assert result == "summary"
    assert captured["client_kwargs"] == {
        "api_key": "not-needed",
        "base_url": "http://localhost:1234/v1",
    }
    assert captured["model"] == "local-model"
    assert captured["temperature"] == 0.1
    assert captured["max_tokens"] == 500
    assert captured["messages"][1]["content"].endswith("[00:00:00] [00:00:01]  A: hello")


def test_llm_requires_api_key_for_remote_provider() -> None:
    provider = LlmProviderConfig(
        name="Yandex AI Studio",
        base_url="https://llm.api.cloud.yandex.net/v1",
        model="gpt://folder/yandexgpt/latest",
        api_key_env="YANDEX_API_KEY",
        requires_api_key=True,
    )

    with pytest.raises(LlmError, match="YANDEX_API_KEY"):
        OpenAICompatibleLlm(provider).generate("system", "user")
