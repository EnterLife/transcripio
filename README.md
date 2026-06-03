# Transcripio

Transcripio is a local-first Python application for transcribing audio and video files.
It provides a Streamlit UI, extracts audio with `ffmpeg`, runs speech recognition with
`faster-whisper`, optionally assigns speakers with a local `pyannote.audio` pipeline,
and exports editable results as TXT, SRT, VTT, DOCX, and JSON.

## Current Features

- Upload one or more audio/video files.
- Process files as a simple queue.
- Extract a mono 16 kHz WAV file before transcription.
- Use a `faster-whisper` model by model name or by local CTranslate2 model path.
- Use an optional local pyannote diarization pipeline for speaker labels.
- Show per-file and queue-level progress.
- Edit transcript segments in the UI before downloading.
- Save transcript history under `data/history/`.
- Save prepared WAV files under `data/output/`.
- Export TXT, SRT, VTT, DOCX, and JSON.

## Project Layout

```text
transcripio/
  app.py                         Streamlit UI
  requirements.txt               single dependency list
  settings.json                  default app settings
  setup.bat                      Windows setup script for .venv
  run.bat                        Windows app launcher
  install_gpu_runtime.bat        optional NVIDIA CUDA runtime installer
  scripts/prepare_pyannote.py    helper for downloading a local pyannote pipeline
  src/transcripio/
    config.py                    app configuration
    media.py                     ffmpeg media preparation
    transcriber.py               faster-whisper adapter
    diarization.py               pyannote adapter and speaker assignment
    pipeline.py                  end-to-end transcription workflow
    formatters.py                TXT/SRT/VTT/DOCX/JSON exports
    storage.py                   transcript history persistence
    models.py                    shared dataclasses
  tests/                         focused unit tests
  data/                          runtime output, ignored by git
  models/                        local model files, ignored by git
```

## Requirements

- Windows Command Prompt or PowerShell.
- Python 3.10 or newer.
- `ffmpeg` available from the command line.
- Enough disk space for ML dependencies and local models.
- Optional: NVIDIA GPU with a compatible PyTorch setup if you want CUDA.

The default setup uses CPU-friendly settings:

- Device: `cpu`
- Compute type: `int8`
- Whisper model: `small`
- Language: auto-detect

## Install ffmpeg

Check whether `ffmpeg` is already available:

```powershell
ffmpeg -version
```

If the command is not found, install ffmpeg and add it to `PATH`. On Windows, one common
option is a full build from Gyan.dev or installation through a package manager such as
Chocolatey, Scoop, or winget. After installation, open a new PowerShell window and run
the check again.

## Quick Start

From the repository root:

```powershell
.\setup.bat
```

The script does all of this:

1. Creates `.venv` if it does not exist.
2. Updates `pip` inside `.venv`.
3. Installs every dependency from `requirements.txt`.
4. Installs this project in editable mode with `pip install -e . --no-deps`.

Then launch the app:

```powershell
.\run.bat
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Manual Setup

Use these commands if you do not want to run `setup.bat`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Run the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

You can also run `python app.py`; the app redirects that direct Python launch into
Streamlit automatically.

## Using the App

1. Open the app with `run.bat`.
2. In the sidebar, choose the Whisper model:
   - Use a model name such as `tiny`, `base`, `small`, `medium`, or `large-v3`.
   - Or use a local CTranslate2 model path such as `models/whisper-large-v3-ct2`.
   - Models marked `downloaded` are already present in the Hugging Face cache or `models/`.
3. Choose the device:
   - `cpu` works everywhere and is the safest first run.
   - `cuda` requires a correctly installed GPU PyTorch stack.
4. Choose the compute type:
   - `int8` is a good CPU default.
   - `float16` is commonly used on GPU.
5. Set the language code, for example `en` or `ru`. Leave it empty for auto-detection.
6. Keep **Assign speakers** off for the fastest first transcription.
7. Upload one or more audio/video files.
8. Click **Process queue**.
9. Edit speaker labels or transcript text in the segment table.
10. Download TXT, SRT, VTT, DOCX, or JSON.

## Supported Input Formats

Audio:

- `mp3`
- `wav`
- `m4a`
- `flac`
- `ogg`
- `aac`

Video:

- `mp4`
- `mov`
- `mkv`
- `avi`
- `webm`
- `m4v`

All supported inputs are converted to a mono 16 kHz WAV file before model processing.

## Local Whisper Models

`faster-whisper` accepts either a model name or a local model path.

Using a model name:

```text
small
```

This is convenient, but the first run may download model files into the local Hugging Face
cache if they are not already present.

The sidebar marks cached models as `downloaded`. If a model is already downloaded, keep
**Use downloaded/local files only** enabled to avoid Hugging Face download checks.

Using a local model path:

```text
models/whisper-large-v3-ct2
```

This is the preferred offline mode. Prepare or download the CTranslate2 model before
launching the app, then paste the local path into the sidebar.

## Local pyannote Speaker Diarization

Speaker diarization is optional. If the diarization path is empty, Transcripio performs
regular transcription without speaker labels.

To use diarization locally:

1. Create a Hugging Face account.
2. Accept the terms for the pyannote diarization repository you want to use.
3. Create a Hugging Face access token.
4. Set the token in PowerShell:

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

5. Download the pipeline snapshot into `models/`:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_pyannote.py
```

6. The script prints a local `config.yaml` path. Paste that path into the app sidebar.

Default output path:

```text
models/pyannote-speaker-diarization/config.yaml
```

After the pipeline is downloaded, the app can use that local path without downloading it
again. Keep the `models/` directory local and do not commit it.

## Offline Mode

For fully offline transcription:

1. Install dependencies while online with `.\setup.bat`.
2. Install `ffmpeg`.
3. Download or prepare a local CTranslate2 Whisper model.
4. Optional: download a local pyannote diarization pipeline.
5. Disconnect from the network.
6. In the app sidebar, use only local model paths.

If you enter a model name such as `small` while offline and the model is not cached yet,
`faster-whisper` may fail because it cannot download the model.

## Output and History

Transcripio stores runtime artifacts locally:

```text
data/output/   prepared WAV files
data/history/  transcript JSON files
```

These directories are ignored by git. The History tab reads saved JSON files from
`data/history/` and lets you reopen, edit, and re-export transcripts.

## Settings

Default UI, model, ffmpeg, and storage settings live in `settings.json`.
The sidebar still lets you override model settings for the current run.

## Running Checks

Compile the Python files:

```powershell
.\.venv\Scripts\python.exe -m compileall app.py src tests
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Check imports:

```powershell
.\.venv\Scripts\python.exe -c "import streamlit, faster_whisper, torchaudio; import transcripio; print('ok')"
```

Check ffmpeg:

```powershell
ffmpeg -version
```

## Troubleshooting

If `ffmpeg` is not found, install it and open a new PowerShell window so `PATH` is refreshed.

If CUDA fails, switch the app sidebar back to:

```text
Device: cpu
Compute type: int8
```

If you see `Library cublas64_12.dll is not found or cannot be loaded`, CUDA runtime
DLLs are missing. Transcripio falls back to CPU when it can, but CPU/int8 is the
recommended first transcription setting.

To use GPU mode, open the CUDA install panel in the sidebar or run:

```powershell
.\install_gpu_runtime.bat
```

This installs official NVIDIA Python packages into `.venv`:

```text
nvidia-cublas-cu12
nvidia-cudnn-cu12==9.*
```

Transcripio automatically adds their DLL directories before loading the Whisper model.
If **Auto-install missing GPU runtime before transcription** is enabled, this install
happens automatically the first time a CUDA transcription needs those DLLs.

If you see a Hugging Face unauthenticated request warning, either use a model marked
`downloaded`, enable **Use downloaded/local files only**, or set `HF_TOKEN` before launch.

If pyannote cannot load the diarization pipeline, confirm that:

- the model terms were accepted on Hugging Face;
- `HF_TOKEN` was set before running `scripts/prepare_pyannote.py`;
- the local `config.yaml` path exists;
- the path pasted into the app points to the local file, not to a web URL.

## License

MIT
