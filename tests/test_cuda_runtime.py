from pathlib import Path

from transcripio import cuda_runtime


def test_discover_nvidia_dll_dirs(tmp_path: Path, monkeypatch) -> None:
    purelib = tmp_path / "site-packages"
    bin_dir = purelib / "nvidia" / "cublas" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "cublas64_12.dll").write_bytes(b"dll")

    monkeypatch.setattr(cuda_runtime.sysconfig, "get_paths", lambda: {"purelib": str(purelib)})

    assert cuda_runtime.discover_nvidia_dll_dirs() == (bin_dir,)


def test_configure_cuda_dll_paths_reports_missing_dlls(tmp_path: Path, monkeypatch) -> None:
    purelib = tmp_path / "site-packages"
    bin_dir = purelib / "nvidia" / "cublas" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "cublas64_12.dll").write_bytes(b"dll")

    monkeypatch.setattr(cuda_runtime.sysconfig, "get_paths", lambda: {"purelib": str(purelib)})
    monkeypatch.setattr(cuda_runtime.os, "add_dll_directory", lambda _path: None, raising=False)
    monkeypatch.setenv("PATH", "")

    status = cuda_runtime.configure_cuda_dll_paths()

    assert status.is_ready is False
    assert status.missing_dlls == ("cudnn64_9.dll",)
    assert status.dll_dirs == (bin_dir,)
