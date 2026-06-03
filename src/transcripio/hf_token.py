from __future__ import annotations

import os
from pathlib import Path


ENV_PATH = Path(".env")
HF_TOKEN_KEY = "HF_TOKEN"


def load_saved_hf_token(env_path: Path = ENV_PATH) -> str | None:
    if not env_path.exists():
        return None

    for line in env_path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == HF_TOKEN_KEY:
            return _clean_env_value(value)
    return None


def resolve_hf_token(env_path: Path = ENV_PATH) -> str | None:
    return os.environ.get(HF_TOKEN_KEY) or load_saved_hf_token(env_path)


def apply_saved_hf_token(env_path: Path = ENV_PATH) -> str | None:
    token = resolve_hf_token(env_path)
    if token:
        os.environ[HF_TOKEN_KEY] = token
    return token


def save_hf_token(token: str, env_path: Path = ENV_PATH) -> None:
    cleaned_token = token.strip()
    if not cleaned_token:
        raise ValueError("HF token cannot be empty.")

    lines = _read_env_lines(env_path)
    replacement = f"{HF_TOKEN_KEY}={cleaned_token}"
    for index, line in enumerate(lines):
        key, separator, _value = line.partition("=")
        if separator and key.strip() == HF_TOKEN_KEY:
            lines[index] = replacement
            break
    else:
        lines.append(replacement)

    _write_env_lines(lines, env_path)
    os.environ[HF_TOKEN_KEY] = cleaned_token


def clear_saved_hf_token(env_path: Path = ENV_PATH) -> None:
    lines = [
        line
        for line in _read_env_lines(env_path)
        if not (line.partition("=")[1] and line.partition("=")[0].strip() == HF_TOKEN_KEY)
    ]
    _write_env_lines(lines, env_path)
    os.environ.pop(HF_TOKEN_KEY, None)


def _read_env_lines(env_path: Path) -> list[str]:
    if not env_path.exists():
        return []
    return env_path.read_text(encoding="utf-8").splitlines()


def _write_env_lines(lines: list[str], env_path: Path) -> None:
    if lines:
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    elif env_path.exists():
        env_path.unlink()


def _clean_env_value(value: str) -> str:
    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    return cleaned
