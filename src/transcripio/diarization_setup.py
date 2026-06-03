from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_DIARIZATION_REPO_ID = "pyannote/speaker-diarization-community-1"
DEFAULT_DIARIZATION_OUTPUT_DIR = Path("models/pyannote-speaker-diarization")
LEGACY_DIARIZATION_REPO_ID = "pyannote/speaker-diarization-3.1"
COMMUNITY_DIARIZATION_REPO_ID = "pyannote/speaker-diarization-community-1"
LEGACY_DIARIZATION_ACCESS_REPOS = (
    "pyannote/segmentation-3.0",
    LEGACY_DIARIZATION_REPO_ID,
)
COMMUNITY_DIARIZATION_ACCESS_REPOS = (COMMUNITY_DIARIZATION_REPO_ID,)
DIARIZATION_REPO_OPTIONS = (
    COMMUNITY_DIARIZATION_REPO_ID,
    LEGACY_DIARIZATION_REPO_ID,
)


@dataclass(frozen=True, slots=True)
class DiarizationDownloadResult:
    snapshot_path: Path
    config_path: Path


@dataclass(frozen=True, slots=True)
class HuggingFaceRepoAccess:
    repo_id: str
    has_access: bool
    message: str


@dataclass(frozen=True, slots=True)
class HuggingFaceTokenCheck:
    username: str | None
    repos: tuple[HuggingFaceRepoAccess, ...]


def check_huggingface_diarization_access(
    token: str,
    repo_id: str = DEFAULT_DIARIZATION_REPO_ID,
) -> HuggingFaceTokenCheck:
    cleaned_token = token.strip()
    if not cleaned_token:
        raise RuntimeError("Enter a Hugging Face token before checking access.")

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is missing. Run setup.bat first.") from exc

    try:
        whoami = HfApi().whoami(token=cleaned_token)
    except Exception as exc:
        raise RuntimeError(f"Hugging Face token check failed: {exc}") from exc

    username = whoami.get("name") or whoami.get("fullname")
    access_results: list[HuggingFaceRepoAccess] = []
    for access_repo_id in required_access_repos(repo_id):
        try:
            hf_hub_download(
                repo_id=access_repo_id,
                filename=".gitattributes",
                token=cleaned_token,
                local_files_only=False,
            )
        except Exception as exc:
            access_results.append(
                HuggingFaceRepoAccess(
                    repo_id=access_repo_id,
                    has_access=False,
                    message=_download_error_message(exc, access_repo_id),
                )
            )
        else:
            access_results.append(
                HuggingFaceRepoAccess(
                    repo_id=access_repo_id,
                    has_access=True,
                    message="Access granted.",
                )
            )

    return HuggingFaceTokenCheck(username=username, repos=tuple(access_results))


def download_diarization_pipeline(
    repo_id: str,
    output_dir: Path,
    token: str,
) -> DiarizationDownloadResult:
    cleaned_repo_id = repo_id.strip() or DEFAULT_DIARIZATION_REPO_ID
    cleaned_token = token.strip()
    if not cleaned_token:
        raise RuntimeError(
            "Missing Hugging Face token. Enter a token after accepting the pyannote model terms."
        )

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is missing. Run setup.bat first.") from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        snapshot_path = Path(
            snapshot_download(
                repo_id=cleaned_repo_id,
                local_dir=str(output_dir),
                token=cleaned_token,
            )
        )
    except Exception as exc:
        raise RuntimeError(_download_error_message(exc, cleaned_repo_id)) from exc

    config_path = snapshot_path / "config.yaml"
    if not config_path.exists():
        raise RuntimeError(
            f"Downloaded {cleaned_repo_id}, but config.yaml was not found under {snapshot_path}."
        )

    return DiarizationDownloadResult(snapshot_path=snapshot_path, config_path=config_path)


def required_access_repos(repo_id: str) -> tuple[str, ...]:
    cleaned_repo_id = repo_id.strip()
    if cleaned_repo_id == LEGACY_DIARIZATION_REPO_ID:
        return LEGACY_DIARIZATION_ACCESS_REPOS
    if cleaned_repo_id == COMMUNITY_DIARIZATION_REPO_ID:
        return COMMUNITY_DIARIZATION_ACCESS_REPOS
    return (cleaned_repo_id or DEFAULT_DIARIZATION_REPO_ID,)


def _download_error_message(exc: Exception, repo_id: str) -> str:
    message = str(exc)
    lowered = message.lower()
    if "403" in lowered or "gated repo" in lowered or "authorized list" in lowered:
        repos = ", ".join(required_access_repos(repo_id))
        return (
            f"Hugging Face denied access to {repo_id}. Log in with the same account that owns "
            "the token, accept the model terms for the required repositories, and use a token "
            f"with read access. Required repositories: {repos}."
        )
    return message
