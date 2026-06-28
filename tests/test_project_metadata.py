from __future__ import annotations

from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_pyproject_dependencies_match_requirements() -> None:
    requirements = _normalized_dependencies(
        (ROOT_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
    )
    pyproject_dependencies = _pyproject_dependencies(
        (ROOT_DIR / "pyproject.toml").read_text(encoding="utf-8")
    )

    assert pyproject_dependencies == requirements


def _pyproject_dependencies(pyproject_text: str) -> list[str]:
    dependencies: list[str] = []
    in_dependencies = False
    for raw_line in pyproject_text.splitlines():
        line = raw_line.strip()
        if line == "dependencies = [":
            in_dependencies = True
            continue
        if in_dependencies and line == "]":
            break
        if in_dependencies and line:
            dependencies.append(line.strip(",").strip('"'))

    return _normalized_dependencies(dependencies)


def _normalized_dependencies(dependencies: list[str]) -> list[str]:
    normalized: list[str] = []
    for dependency in dependencies:
        cleaned = dependency.strip()
        if not cleaned or cleaned.startswith("#"):
            continue
        normalized.append(cleaned.lower().replace("_", "-"))
    return normalized
