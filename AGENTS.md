# AGENTS.md

## Project

Transcripio is a local-first Python application for transcribing audio and video files.
The app provides a simple Streamlit UI, extracts audio from video with `ffmpeg`, runs
speech recognition with a local `faster-whisper` model, and optionally assigns speakers
with a local `pyannote.audio` diarization pipeline.

The main goal is to keep the product simple, runnable, local-only by default, and easy
to extend without leaking model, media, or UI details across module boundaries.

## Tech Stack

- Language: Python 3.10+
- UI: Streamlit
- Speech to Text: `faster-whisper`
- Speaker diarization: `pyannote.audio`
- Audio/video preparation: `ffmpeg`
- Packaging layout: `src/transcripio`
- Tests: `pytest`
- Local environment: `.venv`

## Repository Structure

- `app.py` - Streamlit UI entry point.
- `src/transcripio/config.py` - application configuration.
- `src/transcripio/media.py` - media validation and WAV extraction.
- `src/transcripio/transcriber.py` - local Whisper transcription adapter.
- `src/transcripio/diarization.py` - local speaker diarization adapter and speaker assignment.
- `src/transcripio/pipeline.py` - end-to-end transcription workflow.
- `src/transcripio/formatters.py` - TXT, SRT, and JSON export formatting.
- `src/transcripio/models.py` - dataclasses used across the app.
- `tests/` - focused behavior tests.
- `requirements.txt` - single dependency list for the project.
- `scripts/setup.bat` - creates `.venv`, installs dependencies, and installs the package editable.
- `scripts/run.bat` - launches the Streamlit app.
- `scripts/install_gpu_runtime.bat` - optionally installs NVIDIA CUDA runtime packages.
- `README.md` - setup, launch, and model notes.
- `data/` - local runtime input/output/tmp artifacts; do not commit contents.
- `models/` - local model files; do not commit contents.

## Coding Rules

- Keep edits focused on the requested behavior.
- Preserve user changes already present in the working tree.
- Do not refactor unrelated code while fixing a local issue.
- Do not edit `.env` or Streamlit secrets unless explicitly requested.
- Do not commit secrets, tokens, local model weights, local media files, generated output, or runtime artifacts.
- Avoid hardcoded credentials, absolute local paths, tokens, server URLs, or machine-specific model paths.
- Keep all dependencies in the single `requirements.txt`.
- Prefer running commands through `.venv\Scripts\python.exe` on Windows.
- Do not change generated/runtime folders unless the task requires it:
  - `.venv/`
  - `data/`
  - `models/`
  - `.pytest_cache/`
  - `__pycache__/`
- Do not leave unused imports, helpers, parameters, or dead code after changing files.
- Use explicit names that describe transcription behavior over generic helper names.
- Keep comments short and useful; avoid comments for obvious assignments.

## Code Quality Rules

- Before changing behavior, identify the expected user-visible outcome and the most likely failure case.
- For bug fixes, add or update a regression test that would fail before the fix when practical.
- For new behavior, add focused tests for:
  - the successful path;
  - at least one failure, edge, or invalid-input path.
- Do not add tests that only verify mocks, implementation details, or duplicated source logic.
- Prefer tests that assert product behavior: formatted output, pipeline result, validation error, speaker assignment, or exported content.
- If a change is documentation-only, formatting-only, or a trivial internal cleanup, tests are not required unless behavior could change.
- Keep tests small and local; ordinary tests must not require real models, real media, network access, Hugging Face tokens, or large files.

## Architecture Rules

- Keep UI code in `app.py`; keep transcription, media, diarization, and formatting logic in package modules.
- Keep provider-specific model code behind adapter classes such as `WhisperTranscriber` and `LocalPyannoteDiarizer`.
- Keep raw audio, video, transcripts, and model execution local by default.
- Do not introduce cloud transcription, cloud diarization, or cloud AI calls unless explicitly requested.
- If future cloud integrations are added, make them opt-in and isolated behind provider interfaces.
- Prefer the pipeline as the orchestration layer; do not put business workflow logic directly in the Streamlit UI.
- Keep media conversion centralized in `media.py`.
- Keep export formatting centralized in `formatters.py`.
- Add focused tests when changing segmentation, speaker assignment, formatting, or pipeline behavior.

## Local Model Rules

- `faster-whisper` may accept either a known model name or a local CTranslate2 model path.
- For fully offline usage, prefer documenting local model paths instead of auto-downloading during app logic.
- Speaker diarization should use a local pyannote pipeline path supplied by the user.
- Do not commit downloaded models or Hugging Face caches.
- Do not hardcode local model paths in source files.
- Handle missing model paths with clear user-facing errors.

## Media Rules

- Accept audio and video input through the UI.
- Always prepare a mono 16 kHz WAV before model processing.
- Use `ffmpeg` for video-to-audio extraction and audio normalization.
- Keep prepared output under `data/output/` or a temporary working directory.
- Surface `ffmpeg` failures with concise errors that include the useful stderr details.

## UI Rules

- Keep the Streamlit UI practical and task-first: upload file, choose model settings, run, inspect, download.
- Do not add landing-page or marketing sections unless explicitly requested.
- Keep controls simple and understandable for local desktop use.
- Do not expose implementation jargon unless it helps the user configure a local model.
- For long-running transcription work, keep progress messages clear.

## Test Design Rules

- Tests should verify product behavior rather than implementation trivia.
- For behavior changes, include at least one happy-path test and one negative or edge-case test when applicable.
- Prefer focused tests for:
  - speaker assignment by timestamp overlap;
  - TXT/SRT/JSON formatting;
  - media validation;
  - pipeline behavior with mocked adapters.
- Do not require large models or real media files in ordinary unit tests.
- If model-backed integration tests are added later, keep them opt-in and document required local files.
- Update docs when setup commands, model behavior, export formats, or user workflows change.

## Review Rules

- For non-trivial code changes, perform a final review from the perspective of a fresh reader.
- Review only the user request, changed files, and relevant tests.
- Check for:
  - missing negative tests;
  - behavior that is only partially tested;
  - violations of local-only model/media rules;
  - leaked UI/model/media concerns across module boundaries;
  - unused imports, dead code, or speculative abstractions;
  - unclear user-facing errors.
- If using another AI pass or a new context window for review, provide it with the request, diff, and test results, but not the full prior reasoning.
- Treat AI review as advisory; verify any suggested issue against the code before applying changes.

## Completion Criteria

A code change is complete only when:

- the requested behavior is implemented;
- relevant positive and negative tests are added or intentionally skipped with a clear reason;
- focused checks have been run when feasible;
- changed files are reviewed before the final response;
- remaining risks are mentioned, especially untested model-backed transcription or diarization flows.

## Useful Commands

Create or update the local environment:

```powershell
.\scripts\setup.bat
```

Run the app:

```powershell
.\scripts\run.bat
```

Run focused checks:

```powershell
.\.venv\Scripts\python.exe -m compileall app.py src tests
.\.venv\Scripts\python.exe -m pytest
```

Check imports from the virtual environment:

```powershell
.\.venv\Scripts\python.exe -c "import streamlit, faster_whisper, torchaudio; import transcripio; print('ok')"
```

Check `ffmpeg` availability:

```powershell
ffmpeg -version
```

## Commit Message Suggestions

- After each completed work chunk, include a suggested commit message in the final response.
- Use a lowercase prefix and a short lowercase summary.
- Keep the message in one line without a period at the end.
- Choose the prefix by intent:
  - `add:` for new user-visible features, adapters, tests, or app flows.
  - `fix:` for bug fixes, broken behavior, failing checks, or bad errors.
  - `upd:` for updates to existing features, docs, configs, or expected behavior.
  - `refactor:` for internal restructuring without behavior changes.
  - `docs:` for documentation-only changes.
  - `test:` for test-only maintenance that does not add new coverage.
  - `chore:` for tooling, cleanup, dependency, or repository maintenance.

Examples:

- `add: local transcription app scaffold`
- `fix: speaker assignment overlap logic`
- `upd: virtual environment setup`
- `docs: document local model setup`

## Before Finishing Work

- Review changed files.
- Run the most focused relevant checks when feasible.
- For Python code changes, prefer:

```powershell
.\.venv\Scripts\python.exe -m compileall app.py src tests
.\.venv\Scripts\python.exe -m pytest
```

- For docs-only changes, a syntax or test run is usually not required.
- If a required tool is unavailable, say so explicitly in the final response and mention the remaining risk.
- Mention any remaining risk, especially model-backed transcription or diarization flows that were not executed.
