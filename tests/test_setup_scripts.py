from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_script_module(script_name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_install_ffmpeg_copies_bundled_binary_to_venv_scripts(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = _load_script_module("install_ffmpeg.py")
    source = tmp_path / "package" / "ffmpeg.exe"
    source.parent.mkdir()
    source.write_bytes(b"binary")
    scripts_dir = tmp_path / ".venv" / "Scripts"
    scripts_dir.mkdir(parents=True)

    monkeypatch.setattr(module.sys, "executable", str(scripts_dir / "python.exe"))
    monkeypatch.setattr(module.sys, "platform", "win32")
    monkeypatch.setitem(
        sys.modules,
        "imageio_ffmpeg",
        types.SimpleNamespace(get_ffmpeg_exe=lambda: str(source)),
    )
    monkeypatch.setattr(module, "_verify_ffmpeg", lambda _path: None)

    installed_path = module.install_ffmpeg()

    assert installed_path == scripts_dir / "ffmpeg.exe"
    assert installed_path.read_bytes() == b"binary"


def test_cuda_runtime_setup_skips_install_without_gpu(monkeypatch) -> None:
    module = _load_script_module("install_cuda_runtime_if_needed.py")
    install_called = False

    def fake_install_cuda_runtime_packages():
        nonlocal install_called
        install_called = True

    monkeypatch.setattr(module, "has_cuda_capable_gpu", lambda: False)
    monkeypatch.setattr(module, "install_cuda_runtime_packages", fake_install_cuda_runtime_packages)

    assert module.install_cuda_runtime_if_needed() is False
    assert install_called is False
