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
- Detect local CPU/GPU/RAM/disk capabilities and suggest Fast, Balanced, or Quality
  transcription settings.
- Download supported faster-whisper models into `models/` for explicit offline use.
- Guide transcription with glossary prompts, hotwords, word timestamps, and silence
  quality controls.
- Use an optional local pyannote diarization pipeline for speaker labels.
- Show per-file and queue-level progress.
- Check the local environment before processing: ffmpeg, writable output folders,
  local model paths, and CUDA runtime status.
- Benchmark current local Whisper settings on a short prepared WAV file.
- Edit transcript segments in the UI before downloading, including find/replace,
  split/merge, and bulk speaker renaming.
- Generate optional transcript notes with OpenAI-compatible LLM providers.
- Save transcript history under `data/history/`.
- Save prepared WAV files under `data/output/`.
- Export TXT, SRT, VTT, DOCX, and JSON.

## Project Layout

```text
transcripio/
  app.py                         Streamlit UI
  requirements.txt               single dependency list
  settings.json                  default app settings
  scripts/setup.bat              Windows setup script for .venv
  scripts/run.bat                Windows app launcher
  scripts/install_gpu_runtime.bat optional NVIDIA CUDA runtime installer
  scripts/install_ffmpeg.py       installs ffmpeg into .venv
  scripts/install_cuda_runtime_if_needed.py conditional CUDA runtime installer
  scripts/prepare_pyannote.py    helper for downloading a local pyannote pipeline
  src/transcripio/
    config.py                    app configuration
    media.py                     ffmpeg media preparation
    transcriber.py               faster-whisper adapter
    diarization.py               pyannote adapter and speaker assignment
    pipeline.py                  end-to-end transcription workflow
    llm.py                       OpenAI-compatible LLM adapter
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
- `ffmpeg` is installed into `.venv` by `scripts\setup.bat`.
- Enough disk space for ML dependencies and local models.
- Optional: NVIDIA GPU with a compatible PyTorch setup if you want CUDA.

The default setup uses CPU-friendly settings:

- Device: `cpu`
- Compute type: `int8`
- Whisper model: `small`
- Language: auto-detect

## Install ffmpeg

`scripts\setup.bat` installs a local `ffmpeg.exe` into `.venv\Scripts` using the
`imageio-ffmpeg` package. `scripts\run.bat` prepends `.venv\Scripts` to `PATH`, so the
app uses that local binary without requiring a system-wide install.

To check the bundled binary after setup:

```powershell
.\.venv\Scripts\ffmpeg.exe -version
```

You can still use a system `ffmpeg` by changing `transcription.ffmpeg_path` in
`settings.json`.

## Quick Start

From the repository root:

```powershell
.\scripts\setup.bat
```

The script does all of this:

1. Creates `.venv` if it does not exist.
2. Updates `pip` inside `.venv`.
3. Installs every dependency from `requirements.txt`.
4. Installs `ffmpeg.exe` into `.venv\Scripts`.
5. Installs this project in editable mode with `pip install -e . --no-deps`.
6. Installs NVIDIA CUDA runtime packages only when a CUDA-capable NVIDIA GPU is detected
   and the required CUDA DLLs are missing.

Then launch the app:

```powershell
.\scripts\run.bat
```

Open the local URL printed by Streamlit, usually:

```text
http://localhost:8501
```

## Manual Setup

Use these commands if you do not want to run `scripts\setup.bat`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python scripts\install_ffmpeg.py
python -m pip install -e . --no-deps
python scripts\install_cuda_runtime_if_needed.py
```

Run the app:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

You can also run `python app.py`; the app redirects that direct Python launch into
Streamlit automatically.

## Using the App

1. Open the app with `scripts\run.bat`.
2. In the sidebar, choose the Whisper model:
   - Use a model name such as `tiny`, `base`, `small`, `medium`, or `large-v3`.
   - Or use a local CTranslate2 model path such as `models/whisper-large-v3-ct2`.
   - Models marked `downloaded` are already present in the Hugging Face cache or `models/`.
3. Optional: choose a **Tuning preset** in **Computer profile**:
   - **Fast** prioritizes throughput.
   - **Balanced** is the default recommendation for everyday local use.
   - **Quality** uses stronger/slower decoding settings.
4. Choose the device:
   - `cpu` works everywhere and is the safest first run.
   - `cuda` requires a correctly installed GPU PyTorch stack.
5. Choose the compute type:
   - `int8` is a good CPU default.
   - `float16` is commonly used on GPU.
6. Set the language code, for example `en` or `ru`. Leave it empty for auto-detection.
7. Optional: open **Quality controls** and add glossary terms, names, hotwords, or
   enable word timestamps.
8. Keep **Assign speakers** off for the fastest first transcription.
9. Upload one or more audio/video files.
10. Optional: open **Environment check** in the sidebar and click **Run checks**.
11. Optional: open **Benchmark** after one file has been processed to measure current
    settings on a short local WAV.
12. Click **Process queue**.
13. Select a completed transcript from the queue results.
14. Edit speaker labels or transcript text in the segment table while listening to the
    prepared audio player. Use **Edit tools** for find/replace, speaker renaming, and
    split/merge operations.
15. Optionally generate LLM notes from the edited transcript.
16. Download TXT, SRT, VTT, DOCX, JSON, Words CSV, or generated notes.

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
For video files, Transcripio extracts the audio track with `ffmpeg` first and runs the
models against the prepared WAV, not the original video container. The bundled Streamlit
configuration raises the upload ceiling to 102400 MB for local large-file workflows, and
the UI saves uploads to a temporary file as a stream instead of copying the whole file
through an in-memory buffer.

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

To explicitly prepare a local Whisper model, open **Download Whisper model** in the
sidebar, choose a faster-whisper repo, and download it into `models/`. After reload, the
model appears as a local option and can be used with **Use downloaded/local files only**.

## GPU Performance

Low GPU utilization is normal when transcribing one short file with a small model. Audio
preparation, VAD, and some decoding work still run on CPU, and a `small` model may not
fully load a modern GPU.

For faster CUDA transcription, start with:

```text
Device: cuda
Compute type: float16
Batched inference: enabled
Batch size: 8
Beam size: 1
Best of: 1
Assign speakers: off
```

Increase **Batch size** to `12`, `16`, or `24` if VRAM allows it. Use a larger downloaded
model such as `mobiuslabsgmbh/faster-whisper-large-v3-turbo` if you want the GPU to do
more work. Higher **Beam size** and **Best of** can improve quality, but they add decoding
work and may reduce throughput.

Using a local model path:

```text
models/whisper-large-v3-ct2
```

This is the preferred offline mode. Prepare or download the CTranslate2 model before
launching the app, then paste the local path into the sidebar.

The sidebar **Computer profile** panel detects CPU cores, RAM, free disk space, and
CUDA-capable NVIDIA GPUs when available. Its Fast, Balanced, and Quality presets choose
model, device, compute type, batching, and decoding settings based on the local machine
and downloaded model cache. Presets are starting points: you can still override every
setting before processing.

## Quality Controls

The sidebar **Quality controls** panel passes additional local options to
`faster-whisper`:

- **Glossary / initial prompt**: names, products, acronyms, and domain context.
- **Hotwords**: important words to bias recognition toward when supported by the model.
- **Word timestamps**: stores word-level timing and probability in JSON and Words CSV.
- **Use previous text as context**: improves continuity in long speech, but can carry a
  bad phrase forward.
- **No-speech threshold**, **language detection threshold**, and **hallucination silence
  guard**: tune silence handling and repeated-text hallucination behavior.

If you edit a segment's text manually, Transcripio clears that segment's word timings so
exports do not contain stale word-level timestamps.

## Benchmark

The sidebar **Benchmark** panel uses the current Whisper settings and the smallest
prepared WAV file under `data/output/`. It creates a short temporary clip and reports
elapsed time, detected language, segment count, and speed as `x realtime`. Select one or
more downloaded/local models in **Compare models** to run the same clip through each
model and view a side-by-side speed table. This is a quick way to compare presets or
models on the current computer without changing saved transcripts.

## Editing Tools

Completed transcripts include an **Edit tools** panel above the segment table:

- **Find** locates matching text or speaker labels and shows segment numbers/timestamps.
- **Replace all** updates matching transcript text and clears stale word timestamps for
  changed segments.
- **Rename speaker** applies one speaker name across all matching segments.
- **Split segment** divides a segment at a chosen timestamp, preserving word timings when
  available.
- **Merge segment** combines a segment with the following segment.

## Local pyannote Speaker Diarization

Speaker diarization is optional. If the diarization path is empty, Transcripio performs
regular transcription without speaker labels.

Diarization means "who spoke when". It assigns labels such as `SPEAKER_00` and
`SPEAKER_01`. After a result is created, use **Speaker names** above the segment table to
rename those labels to real names for export.

To use diarization locally:

1. Create a Hugging Face account.
2. Accept the terms for the selected diarization repository:
   - Recommended: `pyannote/speaker-diarization-community-1`
   - Legacy 3.1 additionally requires `pyannote/segmentation-3.0`
3. Create a Hugging Face access token.
   - If the token is fine-grained, grant read access to the required pyannote repositories
     or to public gated repositories you can access.
4. In the app sidebar, enable **Assign speakers**.
5. Open **Download diarization model**.
6. Add the token in the main **Hugging Face** sidebar section and click **Save token**.
7. Open **Download diarization model** and click **Check HF access**.
8. If required repositories show access granted, click **Download speaker model**.

The token is saved to local `.env`, which is ignored by git, and is not saved to
`settings.json`.

You can also download from the command line. Set the token in PowerShell:

```powershell
$env:HF_TOKEN="hf_your_token_here"
```

Then download the pipeline snapshot into `models/`:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_pyannote.py
```

The script prints a local `config.yaml` path. Paste that path into the app sidebar if it
is not selected automatically.

Default output path:

```text
models/pyannote-speaker-diarization/config.yaml
```

After the pipeline is downloaded, the app can use that local path without downloading it
again. Keep the `models/` directory local and do not commit it.

The sidebar automatically scans `models/` for local pyannote `config.yaml` files when
**Assign speakers** is enabled.

## LLM Notes

Transcripio can send an edited transcript to an OpenAI-compatible chat completions
provider for optional notes, summaries, action items, or a custom prompt. This step is
separate from transcription and runs only when you click **Generate** in **LLM notes**.

The default `settings.json` includes two provider examples:

```json
"llm": {
  "default_provider": "LM Studio",
  "providers": [
    {
      "name": "LM Studio",
      "base_url": "http://localhost:1234/v1",
      "model": "local-model",
      "api_key_env": "LM_STUDIO_API_KEY",
      "requires_api_key": false
    },
    {
      "name": "Yandex AI Studio",
      "base_url": "https://llm.api.cloud.yandex.net/v1",
      "model": "gpt://<folder_id>/yandexgpt/latest",
      "api_key_env": "YANDEX_API_KEY",
      "requires_api_key": true
    }
  ]
}
```

### LM Studio

1. Open LM Studio.
2. Download and load a chat model.
3. Start the local server in the Developer/API section, or run:

```powershell
lms server start
```

4. In `settings.json`, set the provider `model` to the model ID served by LM Studio.
5. Keep the base URL as:

```text
http://localhost:1234/v1
```

LM Studio usually does not require an API key for local use. If your local server is
configured to require one, set the variable named in `api_key_env` before launching:

```powershell
$env:LM_STUDIO_API_KEY="your-local-key"
```

### Yandex AI Studio and other providers

For a remote OpenAI-compatible provider, add or edit a provider entry with its `base_url`,
`model`, and `api_key_env`. Keep the real key out of `settings.json`:

```powershell
$env:YANDEX_API_KEY="your-api-key"
```

Then choose the provider in the sidebar and generate notes from a completed transcript.

## Offline Mode

For fully offline transcription:

1. Install dependencies while online with `.\scripts\setup.bat`.
2. Run `.\scripts\setup.bat` while online so the local `.venv` and `ffmpeg` binary are ready.
3. Download or prepare a local CTranslate2 Whisper model.
4. Optional: download a local pyannote diarization pipeline.
5. Optional: run LM Studio locally if you want offline LLM notes.
6. Disconnect from the network.
7. In the app sidebar, use only local model paths and local LLM providers.

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

## Environment Check

The sidebar **Environment check** panel validates the current run settings without
starting transcription. It checks that `ffmpeg` is callable, output and history folders
are writable, local model paths exist, and CUDA runtime DLLs are present when CUDA is
selected. Use it before a long queue or after changing model, storage, or GPU settings.

## Running Checks

Compile the Python files:

```powershell
.\.venv\Scripts\python.exe -m compileall app.py src tests
```

Run tests:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Run opt-in model-backed checks with local cached models and a local WAV file:

```powershell
$env:TRANSCRIPIO_RUN_MODEL_TESTS="1"
$env:TRANSCRIPIO_MODEL_TEST_AUDIO="data/output/example.prepared.wav"
$env:TRANSCRIPIO_WHISPER_MODEL="small"
$env:TRANSCRIPIO_DIARIZATION_MODEL="models/pyannote-speaker-diarization/config.yaml"
.\.venv\Scripts\python.exe -m pytest tests\test_model_backed_integration.py
```

Leave `TRANSCRIPIO_MODEL_TEST_AUDIO` unset to use the smallest WAV found under
`data/output/`. These checks are skipped during ordinary test runs because they require
local model weights and local media.

Check imports:

```powershell
.\.venv\Scripts\python.exe -c "import streamlit, faster_whisper, torchaudio; import transcripio; print('ok')"
```

Check the local ffmpeg binary:

```powershell
.\.venv\Scripts\ffmpeg.exe -version
```

## Troubleshooting

If `ffmpeg` is not found when launching through another shell, use `scripts\run.bat` or
activate `.venv` so `.venv\Scripts` is on `PATH`.

If CUDA fails, switch the app sidebar back to:

```text
Device: cpu
Compute type: int8
```

If you see `Library cublas64_12.dll is not found or cannot be loaded`, CUDA runtime
DLLs are missing. Transcripio falls back to CPU when it can, but CPU/int8 is the
recommended first transcription setting.

To use GPU mode on a computer with an NVIDIA GPU, open the CUDA install panel in the
sidebar or run:

```powershell
.\scripts\install_gpu_runtime.bat
```

This checks for a CUDA-capable NVIDIA GPU and then installs official NVIDIA Python
runtime packages into `.venv` only when the required DLLs are missing:

```text
nvidia-cublas-cu12
nvidia-cudnn-cu12==9.*
```

Transcripio automatically adds their DLL directories before loading the Whisper model.
If no CUDA-capable NVIDIA GPU is detected, setup and transcription skip this download and
use CPU-friendly defaults. If **Auto-install missing GPU runtime before transcription**
is enabled, the install happens automatically only when CUDA is selected, a compatible
GPU is detected, and those DLLs are missing.

If you see a Hugging Face unauthenticated request warning, either use a model marked
`downloaded`, enable **Use downloaded/local files only**, or set `HF_TOKEN` before launch.

If pyannote cannot load the diarization pipeline, confirm that:

- the model terms were accepted on Hugging Face;
- `HF_TOKEN` was set before running `scripts/prepare_pyannote.py`;
- the local `config.yaml` path exists;
- the path pasted into the app points to the local file, not to a web URL.

## License

MIT
