from pathlib import Path

from transcripio.hf_token import clear_saved_hf_token, load_saved_hf_token, save_hf_token


def test_save_hf_token_updates_env_file(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OTHER=value\nHF_TOKEN=old\n", encoding="utf-8")
    monkeypatch.delenv("HF_TOKEN", raising=False)

    save_hf_token("hf_new", env_path)

    assert env_path.read_text(encoding="utf-8") == "OTHER=value\nHF_TOKEN=hf_new\n"
    assert load_saved_hf_token(env_path) == "hf_new"


def test_clear_saved_hf_token_preserves_other_env_values(tmp_path: Path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("OTHER=value\nHF_TOKEN=hf_saved\n", encoding="utf-8")
    monkeypatch.setenv("HF_TOKEN", "hf_saved")

    clear_saved_hf_token(env_path)

    assert env_path.read_text(encoding="utf-8") == "OTHER=value\n"
