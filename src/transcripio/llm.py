from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
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
        self._uses_default_client_factory = client_factory is None

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

        model = _resolve_model_alias(self._provider)
        if self._uses_default_client_factory and _is_local_lm_studio_provider(self._provider):
            try:
                content = _create_lm_studio_chat_completion(
                    self._provider,
                    model,
                    system_prompt,
                    user_prompt,
                    api_key=self._api_key,
                )
            except Exception as exc:  # noqa: BLE001 - adapters should expose concise UI errors.
                raise LlmError(_format_llm_exception(exc, self._provider, model)) from exc
        else:
            client = self._client_factory(api_key=api_key, base_url=self._provider.base_url)
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=self._provider.temperature,
                    max_tokens=self._provider.max_tokens,
                )
            except Exception as exc:  # noqa: BLE001 - adapters should expose concise UI errors.
                raise LlmError(_format_llm_exception(exc, self._provider, model)) from exc
            content = response.choices[0].message.content

        generated_text = str(content or "").strip()
        if not generated_text:
            raise LlmError(
                f"{self._provider.name} returned an empty response. "
                "Try increasing max_tokens or choose a non-reasoning chat model."
            )
        return generated_text


def _resolve_model_alias(provider: LlmProviderConfig) -> str:
    configured_model = provider.model.strip()
    if configured_model != "local-model" or not _is_local_lm_studio_provider(provider):
        return configured_model

    model_ids = _fetch_available_model_ids(provider.base_url)
    chat_model_ids = [model_id for model_id in model_ids if "embedding" not in model_id.lower()]
    return (chat_model_ids or model_ids or [configured_model])[0]


def _is_local_lm_studio_provider(provider: LlmProviderConfig) -> bool:
    if provider.name.strip().lower() != "lm studio":
        return False

    parsed_base_url = urlparse(provider.base_url.strip())
    return parsed_base_url.hostname in {"localhost", "127.0.0.1", "::1"}


def _format_llm_exception(exc: Exception, provider: LlmProviderConfig, model: str) -> str:
    status_code = getattr(exc, "status_code", None)
    message = str(exc).strip()

    if status_code == 503 and _is_local_lm_studio_provider(provider):
        return (
            f"{provider.name} returned 503 for model '{model}'. "
            "Open LM Studio, load a chat model, start the local server, then try again. "
            "If a model is already loaded, set llm.providers[].model in settings.json "
            "to the exact model id shown by http://localhost:1234/v1/models."
        )

    return message or f"{provider.name} request failed."


class _LlmHttpError(RuntimeError):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


def _create_lm_studio_chat_completion(
    provider: LlmProviderConfig,
    model: str,
    system_prompt: str,
    user_prompt: str,
    api_key: str | None = None,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": provider.temperature,
        "max_tokens": provider.max_tokens,
    }
    response_payload = _post_json(
        f"{provider.base_url.rstrip('/')}/chat/completions",
        payload,
        api_key=api_key,
    )
    try:
        return str(response_payload["choices"][0]["message"].get("content") or "")
    except (KeyError, IndexError, TypeError) as exc:
        raise LlmError(f"{provider.name} returned an unexpected response format.") from exc


def _post_json(url: str, payload: dict[str, Any], api_key: str | None = None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=600) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace").strip()
        raise _LlmHttpError(exc.code, details or exc.reason) from exc


def _fetch_available_model_ids(base_url: str) -> list[str]:
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        with urlopen(models_url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return []

    models = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(models, list):
        return []

    model_ids: list[str] = []
    for model in models:
        if not isinstance(model, dict):
            continue
        model_id = str(model.get("id", "")).strip()
        if model_id:
            model_ids.append(model_id)
    return model_ids


def _openai_client_factory(**kwargs: Any) -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LlmError("openai is not installed. Run: pip install -r requirements.txt") from exc

    return OpenAI(**kwargs)
