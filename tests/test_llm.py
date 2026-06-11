import pytest

from transcripio.config import LlmProviderConfig
from transcripio.llm import LlmError, OpenAICompatibleLlm
from transcripio.models import TranscriptSegment


def test_openai_compatible_llm_sends_transcript_request(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr("transcripio.llm._fetch_available_model_ids", lambda _base_url: [])

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


def test_lm_studio_local_model_alias_uses_available_chat_model(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "transcripio.llm._fetch_available_model_ids",
        lambda _base_url: ["text-embedding-nomic-embed-text-v1.5", "openai/gpt-oss-20b"],
    )

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

    provider = LlmProviderConfig(
        name="LM Studio",
        base_url="http://localhost:1234/v1",
        model="local-model",
        requires_api_key=False,
    )

    result = OpenAICompatibleLlm(
        provider,
        client_factory=lambda **_kwargs: FakeClient(),
    ).generate("system", "user")

    assert result == "summary"
    assert captured["model"] == "openai/gpt-oss-20b"


def test_lm_studio_503_error_explains_model_loading(monkeypatch) -> None:
    monkeypatch.setattr(
        "transcripio.llm._fetch_available_model_ids",
        lambda _base_url: ["openai/gpt-oss-20b"],
    )

    class FakeOpenAiError(RuntimeError):
        status_code = 503

    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            raise FakeOpenAiError("Error code: 503")

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    provider = LlmProviderConfig(
        name="LM Studio",
        base_url="http://localhost:1234/v1",
        model="local-model",
        requires_api_key=False,
    )

    with pytest.raises(LlmError, match="load a chat model"):
        OpenAICompatibleLlm(
            provider,
            client_factory=lambda **_kwargs: FakeClient(),
        ).generate("system", "user")
