from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from transcripio.config import LlmProviderConfig
from transcripio.formatters import to_txt
from transcripio.models import TranscriptSegment


ClientFactory = Callable[..., Any]

SYSTEM_PROMPT = (
    "You help analyze audio and video transcripts. Keep the answer grounded in the "
    "provided transcript and do not invent missing details."
)


class LlmError(RuntimeError):
    pass


class OpenAICompatibleLlm:
    def __init__(
        self,
        provider: LlmProviderConfig,
        api_key: str | None = None,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._provider = provider
        self._api_key = api_key.strip() if api_key else None
        self._client_factory = client_factory or _openai_client_factory

    def generate_transcript_note(
        self,
        segments: Sequence[TranscriptSegment],
        instruction: str,
    ) -> str:
        clean_instruction = instruction.strip()
        if not clean_instruction:
            raise LlmError("Enter an instruction for the LLM.")

        transcript_text = to_txt(list(segments)).strip()
        if not transcript_text:
            raise LlmError("Transcript is empty.")

        user_prompt = "\n\n".join(
            [
                clean_instruction,
                "Transcript:",
                transcript_text,
            ]
        )
        return self.generate(SYSTEM_PROMPT, user_prompt)

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self._provider.base_url.strip():
            raise LlmError("LLM provider base URL is empty.")
        if not self._provider.model.strip():
            raise LlmError("LLM provider model is empty.")

        api_key = self._api_key or "not-needed"
        if self._provider.requires_api_key and not self._api_key:
            if self._provider.api_key_env:
                raise LlmError(f"Set {self._provider.api_key_env} before using this provider.")
            raise LlmError("Set an API key before using this provider.")

        client = self._client_factory(api_key=api_key, base_url=self._provider.base_url)
        try:
            response = client.chat.completions.create(
                model=self._provider.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self._provider.temperature,
                max_tokens=self._provider.max_tokens,
            )
        except Exception as exc:  # noqa: BLE001 - adapters should expose concise UI errors.
            raise LlmError(str(exc)) from exc

        content = response.choices[0].message.content
        return str(content or "").strip()


def _openai_client_factory(**kwargs: Any) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LlmError("openai is not installed. Run: pip install -r requirements.txt") from exc

    return OpenAI(**kwargs)
