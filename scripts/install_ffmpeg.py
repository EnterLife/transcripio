from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def install_ffmpeg() -> Path:
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "imageio-ffmpeg is not installed. Run pip install -r requirements.txt first."
        ) from exc

    source_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
    scripts_dir = Path(sys.executable).resolve().parent
    target_path = scripts_dir / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")

    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)
    target_path.chmod(0o755)

    _verify_ffmpeg(target_path)
    return target_path


def _verify_ffmpeg(ffmpeg_path: Path) -> None:
    try:
        completed = subprocess.run(
            [str(ffmpeg_path), "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Installed ffmpeg could not be started: {exc}") from exc

    if completed.returncode != 0:
        details = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"Installed ffmpeg check failed: {details.splitlines()[0]}")


def main() -> int:
    try:
        ffmpeg_path = install_ffmpeg()
    except Exception as exc:  # noqa: BLE001 - setup should print a concise failure.
        print(f"ffmpeg installation failed: {exc}", file=sys.stderr)
        return 1

    print(f"ffmpeg installed in virtual environment: {ffmpeg_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
