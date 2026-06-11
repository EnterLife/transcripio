from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from shutil import copyfileobj
from typing import NoReturn

import streamlit as st

UPLOAD_COPY_CHUNK_SIZE = 1024 * 1024
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transcripio.config import AppConfig, AppSettings, LlmProviderConfig, load_settings
from transcripio.cuda_runtime import (
    CUDA_RUNTIME_PACKAGES,
    configure_cuda_dll_paths,
    install_cuda_runtime_packages,
)
from transcripio.diarization_setup import (
    DIARIZATION_REPO_OPTIONS,
    check_huggingface_diarization_access,
    download_diarization_pipeline,
    required_access_repos,
)
from transcripio.formatters import to_docx, to_json, to_srt, to_txt, to_vtt
from transcripio.hf_token import (
    apply_saved_hf_token,
    clear_saved_hf_token,
    resolve_hf_token,
    save_hf_token,
)
from transcripio.model_catalog import (
    WhisperModelOption,
    list_diarization_model_options,
    list_whisper_model_options,
)
from transcripio.llm import LlmError, OpenAICompatibleLlm
from transcripio.models import TranscriptSegment, TranscriptionResult
from transcripio.pipeline import TranscriptionPipeline
from transcripio.storage import list_history, load_result, save_result


def _inject_status_spinner_css() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stStatusWidgetRunningIcon"] svg {
            display: none;
        }

        div[data-testid="stStatusWidgetRunningIcon"]::before {
            content: "";
            display: block;
            width: 1.2rem;
            height: 1.2rem;
            border: 2px solid rgba(250, 250, 250, 0.24);
            border-top-color: rgba(250, 250, 250, 0.82);
            border-radius: 50%;
            animation: transcripio-status-spin 0.8s linear infinite;
        }

        @keyframes transcripio-status-spin {
            to {
                transform: rotate(360deg);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _show_result_editor(
    result: TranscriptionResult,
    history_dir: Path,
    editor_key_prefix: str,
    llm_provider_config: LlmProviderConfig | None = None,
) -> None:
    st.subheader(result.source_name)
    if result.duration is not None:
        st.caption(f"Language: {result.language or 'unknown'} | Duration: {result.duration:.1f}s")
    else:
        st.caption(f"Language: {result.language or 'unknown'}")

    speaker_names = _speaker_name_overrides(result, editor_key_prefix)

    rows = [
        {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "speaker": speaker_names.get(segment.speaker or "", segment.speaker or ""),
            "text": segment.text,
        }
        for segment in result.segments
    ]
    edited_rows = st.data_editor(
        rows,
        key=f"{editor_key_prefix}-segments-{result.job_id}",
        width="stretch",
        hide_index=True,
        num_rows="fixed",
        column_config={
            "start": st.column_config.NumberColumn("Start", disabled=True, format="%.3f"),
            "end": st.column_config.NumberColumn("End", disabled=True, format="%.3f"),
            "speaker": st.column_config.TextColumn("Speaker"),
            "text": st.column_config.TextColumn("Text", width="large"),
        },
    )

    result.segments = [
        TranscriptSegment(
            start=float(row["start"]),
            end=float(row["end"]),
            speaker=(str(row["speaker"]).strip() or None),
            text=str(row["text"]).strip(),
        )
        for row in edited_rows
    ]
    save_result(result, history_dir)

    preview = to_txt(result.segments)
    st.text_area(
        "Plain text preview",
        value=preview,
        height=220,
        key=f"{editor_key_prefix}-plain-preview-{result.job_id}",
    )

    _show_llm_actions(result, editor_key_prefix, llm_provider_config)

    base_name = Path(result.source_name).stem or "transcript"
    txt = to_txt(result.segments)
    srt = to_srt(result.segments)
    vtt = to_vtt(result.segments)
    json_text = to_json(result)
    docx = to_docx(result)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.download_button(
        "TXT",
        txt,
        file_name=f"{base_name}.txt",
        mime="text/plain",
        key=f"{editor_key_prefix}-download-txt-{result.job_id}",
    )
    col2.download_button(
        "SRT",
        srt,
        file_name=f"{base_name}.srt",
        mime="application/x-subrip",
        key=f"{editor_key_prefix}-download-srt-{result.job_id}",
    )
    col3.download_button(
        "VTT",
        vtt,
        file_name=f"{base_name}.vtt",
        mime="text/vtt",
        key=f"{editor_key_prefix}-download-vtt-{result.job_id}",
    )
    col4.download_button(
        "DOCX",
        docx,
        file_name=f"{base_name}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        key=f"{editor_key_prefix}-download-docx-{result.job_id}",
    )
    col5.download_button(
        "JSON",
        json_text,
        file_name=f"{base_name}.json",
        mime="application/json",
        key=f"{editor_key_prefix}-download-json-{result.job_id}",
    )


def _speaker_name_overrides(
    result: TranscriptionResult,
    editor_key_prefix: str,
) -> dict[str, str]:
    speaker_labels = sorted({segment.speaker for segment in result.segments if segment.speaker})
    if not speaker_labels:
        return {}

    with st.expander("Speaker names"):
        overrides: dict[str, str] = {}
        for label in speaker_labels:
            name = st.text_input(
                label,
                value="",
                key=f"{editor_key_prefix}-speaker-name-{result.job_id}-{label}",
                placeholder=f"Name for {label}",
            ).strip()
            if name:
                overrides[label] = name
        return overrides


def _show_llm_actions(
    result: TranscriptionResult,
    editor_key_prefix: str,
    provider_config: LlmProviderConfig | None,
) -> None:
    if provider_config is None:
        return

    note_key = f"{editor_key_prefix}-llm-note-{result.job_id}-{provider_config.name}"
    prompt_options = {
        "Summary": "Summarize the transcript in concise bullet points.",
        "Action items": (
            "Extract action items, owners, and deadlines. Mark missing owners or dates as unknown."
        ),
        "Meeting notes": "Write structured meeting notes with decisions, risks, and next steps.",
        "Custom": "",
    }

    with st.expander("LLM notes"):
        st.caption(f"{provider_config.name}: {provider_config.model}")
        prompt_label = st.selectbox(
            "Prompt",
            options=list(prompt_options),
            key=f"{editor_key_prefix}-llm-prompt-{result.job_id}",
        )
        if prompt_label == "Custom":
            instruction = st.text_area(
                "Instruction",
                value="",
                height=100,
                key=f"{editor_key_prefix}-llm-custom-instruction-{result.job_id}",
            )
        else:
            instruction = prompt_options[prompt_label]

        if st.button(
            "Generate",
            key=f"{editor_key_prefix}-llm-generate-{result.job_id}",
            width="stretch",
        ):
            api_key = os.getenv(provider_config.api_key_env or "")
            try:
                with st.spinner(f"Calling {provider_config.name}"):
                    note = OpenAICompatibleLlm(
                        provider_config,
                        api_key=api_key,
                    ).generate_transcript_note(result.segments, instruction)
            except LlmError as exc:
                st.error(str(exc))
            else:
                st.session_state[note_key] = note

        if note_key in st.session_state:
            st.text_area(
                "Output",
                value=st.session_state[note_key],
                height=260,
                key=f"{editor_key_prefix}-llm-output-{result.job_id}",
            )
            st.download_button(
                "Download note",
                st.session_state[note_key],
                file_name=f"{Path(result.source_name).stem or 'transcript'}-llm-note.txt",
                mime="text/plain",
                key=f"{editor_key_prefix}-llm-download-{result.job_id}",
            )


def _select_llm_provider(settings: AppSettings) -> LlmProviderConfig | None:
    if not settings.llm_providers:
        st.caption("No LLM providers configured.")
        return None

    provider_names = [provider.name for provider in settings.llm_providers]
    selected_name = st.selectbox(
        "Provider",
        options=provider_names,
        index=_selected_index(provider_names, settings.default_llm_provider),
    )
    provider = settings.llm_providers[provider_names.index(selected_name)]
    st.caption(f"{provider.base_url} | {provider.model}")

    if provider.requires_api_key:
        if provider.api_key_env and os.getenv(provider.api_key_env):
            st.caption(f"{provider.api_key_env} is available.")
        elif provider.api_key_env:
            st.warning(f"Set {provider.api_key_env} before using this provider.")
        else:
            st.warning("This provider requires an API key.")
    elif provider.api_key_env and os.getenv(provider.api_key_env):
        st.caption(f"{provider.api_key_env} is available.")
    else:
        st.caption("API key is optional for this provider.")

    return provider


def _selected_index(options: list[str], value: str, default: int = 0) -> int:
    try:
        return options.index(value)
    except ValueError:
        return default


def _selected_model_option(
    options: list[WhisperModelOption],
    configured_model: str,
) -> WhisperModelOption | None:
    for option in options:
        if option.value == configured_model:
            return option
    return options[0] if options else None


def _has_streamlit_context() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


def _save_uploaded_media(uploaded_file, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    uploaded_file.seek(0)
    with destination.open("wb") as output_file:
        copyfileobj(uploaded_file, output_file, UPLOAD_COPY_CHUNK_SIZE)
    uploaded_file.seek(0)


def _launch_with_streamlit() -> NoReturn:
    from streamlit.web import cli as streamlit_cli

    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_SHOW_EMAIL_PROMPT", "false")
    sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
    raise SystemExit(streamlit_cli.main())


def _show_hf_token_controls() -> str | None:
    saved_or_env_token = resolve_hf_token()

    st.header("Hugging Face")
    if saved_or_env_token:
        st.caption("HF_TOKEN is available for model downloads.")
    else:
        st.caption("HF_TOKEN is not set. Downloaded/local models work without it.")

    with st.expander("HF token"):
        token_input = st.text_input(
            "HF token",
            value="",
            type="password",
            placeholder="Paste a token to save or use for this session",
            help="Saved to local .env, which is ignored by git.",
        )
        col1, col2 = st.columns(2)
        if col1.button("Save token", width="stretch"):
            try:
                save_hf_token(token_input)
            except ValueError as exc:
                st.error(str(exc))
            else:
                saved_or_env_token = resolve_hf_token()
                st.success("HF_TOKEN saved to .env")
        if col2.button("Clear token", width="stretch"):
            clear_saved_hf_token()
            saved_or_env_token = None
            st.success("HF_TOKEN removed from .env")

    return token_input.strip() or saved_or_env_token


def main() -> None:
    apply_saved_hf_token()
    settings = load_settings()
    default_config = settings.config
    model_options = list_whisper_model_options(settings.whisper_models)
    diarization_options = list_diarization_model_options()

    st.set_page_config(page_title=settings.page_title, page_icon=settings.page_icon, layout="wide")
    _inject_status_spinner_css()

    st.title(settings.page_title)
    st.caption(settings.caption)

    if "results" not in st.session_state:
        st.session_state.results = {}

    with st.sidebar:
        hf_token = _show_hf_token_controls()
        st.divider()
        st.header("Models")
        selected_default = _selected_model_option(model_options, default_config.whisper_model)
        model_labels = [option.label for option in model_options] + ["Custom name or path"]
        selected_model_label = st.selectbox(
            "Whisper model",
            options=model_labels,
            index=model_options.index(selected_default) if selected_default else 0,
        )
        if selected_model_label == "Custom name or path":
            selected_model_option = None
            model_name_or_path = st.text_input(
                "Model name or local path",
                value=default_config.whisper_model,
                help="Use a faster-whisper model name, HF repo ID, or local CTranslate2 directory.",
            )
        else:
            selected_model_option = model_options[model_labels.index(selected_model_label)]
            model_name_or_path = selected_model_option.value

        if selected_model_option:
            if selected_model_option.is_downloaded:
                st.caption("This model is available locally.")
            else:
                st.caption("This model can be downloaded by faster-whisper on first use.")

        local_files_default = default_config.local_files_only
        if selected_model_option and selected_model_option.is_downloaded:
            local_files_default = True
        local_files_only = st.checkbox(
            "Use downloaded/local files only",
            value=local_files_default,
            help="Avoids Hugging Face downloads and token warnings when the selected model is already local.",
        )

        device_options = ["cpu", "cuda"]
        device = st.selectbox(
            "Device",
            options=device_options,
            index=_selected_index(device_options, default_config.device),
        )
        if device == "cuda":
            cuda_status = configure_cuda_dll_paths()
            st.warning(
                "CUDA needs NVIDIA CUDA/cuBLAS DLLs. If they are missing, Transcripio will fall back to CPU."
            )
            if cuda_status.is_ready:
                st.caption("CUDA runtime DLLs were found.")
            else:
                st.caption(f"Missing CUDA DLLs: {', '.join(cuda_status.missing_dlls)}")
                with st.expander("Install official CUDA runtime packages"):
                    st.caption(
                        "Installs via pip: " + ", ".join(CUDA_RUNTIME_PACKAGES)
                    )
                    if st.button("Install GPU runtime", width="stretch"):
                        with st.spinner("Installing NVIDIA CUDA runtime packages"):
                            completed = install_cuda_runtime_packages()
                        if completed.returncode == 0:
                            configure_cuda_dll_paths()
                            st.success("GPU runtime packages installed. Try processing again.")
                        else:
                            details = completed.stderr.strip() or completed.stdout.strip()
                            st.error(f"Could not install GPU runtime packages: {details}")
            auto_install_cuda_runtime = st.checkbox(
                "Auto-install missing GPU runtime before transcription",
                value=default_config.auto_install_cuda_runtime,
                help="Downloads official NVIDIA pip packages only when CUDA is selected and DLLs are missing.",
            )
        else:
            auto_install_cuda_runtime = False
        compute_type_options = ["int8", "float16", "float32"]
        default_compute_type = default_config.compute_type
        if device == "cuda" and default_compute_type == "int8":
            default_compute_type = "float16"
        compute_type = st.selectbox(
            "Compute type",
            options=compute_type_options,
            index=_selected_index(compute_type_options, default_compute_type),
            help="Use int8 for most CPU runs and float16 for many GPU runs.",
        )
        use_batched_inference = False
        batch_size = default_config.batch_size
        beam_size = default_config.beam_size
        best_of = default_config.best_of
        cpu_threads = default_config.cpu_threads
        num_workers = default_config.num_workers
        vad_filter = default_config.vad_filter
        with st.expander("Performance"):
            use_batched_inference = st.checkbox(
                "Batched inference",
                value=device == "cuda" or default_config.use_batched_inference,
                help="Improves GPU utilization by decoding multiple chunks together.",
            )
            if use_batched_inference:
                batch_size = st.slider(
                    "Batch size",
                    min_value=1,
                    max_value=32,
                    value=default_config.batch_size,
                    step=1,
                    help="Higher values load the GPU more but require more VRAM.",
                )
            beam_size = st.slider(
                "Beam size",
                min_value=1,
                max_value=8,
                value=default_config.beam_size,
                step=1,
                help="1 is fastest. Higher values may improve quality but slow decoding.",
            )
            best_of = st.slider(
                "Best of",
                min_value=1,
                max_value=8,
                value=default_config.best_of,
                step=1,
                help="1 is fastest. Higher values do extra candidate decoding.",
            )
            vad_filter = st.checkbox(
                "Skip silence",
                value=default_config.vad_filter,
                help="Filters silence before transcription. Usually faster for recordings with pauses.",
            )
            cpu_threads = st.number_input(
                "CPU threads",
                min_value=0,
                max_value=64,
                value=default_config.cpu_threads,
                step=1,
                help="0 lets CTranslate2 choose automatically.",
            )
            num_workers = st.number_input(
                "Workers",
                min_value=1,
                max_value=8,
                value=default_config.num_workers,
                step=1,
                help="More workers can improve throughput for queued files, but uses more memory.",
            )
        language = st.text_input(
            "Language",
            value=default_config.language or "",
            help="Leave empty for auto-detection.",
        )
        use_diarization = st.checkbox(
            "Assign speakers",
            value=bool(default_config.diarization_model_path),
            help="Keep this off for the fastest first transcription.",
        )
        if use_diarization:
            with st.expander("Download diarization model"):
                st.caption(
                    "Accept the pyannote model terms on Hugging Face first. Downloads use the token from the Hugging Face section."
                )
                selected_diarization_repo = st.selectbox(
                    "Hugging Face repo",
                    options=list(DIARIZATION_REPO_OPTIONS) + ["Custom repo"],
                    index=_selected_index(
                        list(DIARIZATION_REPO_OPTIONS) + ["Custom repo"],
                        default_config.diarization_repo_id,
                    ),
                )
                if selected_diarization_repo == "Custom repo":
                    diarization_repo_id = st.text_input(
                        "Custom Hugging Face repo",
                        value=default_config.diarization_repo_id,
                    )
                else:
                    diarization_repo_id = selected_diarization_repo

                st.markdown(
                    "\n".join(
                        f"- Accept access for [{repo_id}](https://huggingface.co/{repo_id})"
                        for repo_id in required_access_repos(diarization_repo_id)
                    )
                )
                diarization_output_dir = st.text_input(
                    "Save to",
                    value=str(default_config.diarization_output_dir),
                    help="Local folder for the pyannote pipeline snapshot.",
                )
                if hf_token:
                    st.caption("Using HF_TOKEN from the Hugging Face section.")
                else:
                    st.caption("Add HF_TOKEN in the Hugging Face section before checking or downloading.")
                if st.button("Check HF access", width="stretch"):
                    try:
                        with st.spinner("Checking Hugging Face access"):
                            access_check = check_huggingface_diarization_access(
                                token=hf_token or "",
                                repo_id=diarization_repo_id,
                            )
                    except Exception as exc:  # noqa: BLE001 - Streamlit should show a clean error.
                        st.error(str(exc))
                    else:
                        if access_check.username:
                            st.caption(f"Token account: {access_check.username}")
                        for repo_access in access_check.repos:
                            if repo_access.has_access:
                                st.success(f"{repo_access.repo_id}: access granted")
                            else:
                                st.error(f"{repo_access.repo_id}: {repo_access.message}")
                if st.button("Download speaker model", width="stretch"):
                    try:
                        with st.spinner("Downloading local diarization pipeline"):
                            download_result = download_diarization_pipeline(
                                repo_id=diarization_repo_id,
                                output_dir=Path(diarization_output_dir),
                                token=hf_token or "",
                            )
                    except Exception as exc:  # noqa: BLE001 - Streamlit should show a clean error.
                        st.error(f"Could not download diarization model: {exc}")
                    else:
                        diarization_options = list_diarization_model_options()
                        default_config.diarization_model_path = str(download_result.config_path)
                        st.success(f"Downloaded: {download_result.config_path}")

            diarization_labels = [option.label for option in diarization_options] + [
                "Custom path"
            ]
            if not diarization_options:
                st.caption("No local diarization config.yaml found under models/.")
            default_diarization_label = "Custom path"
            for option in diarization_options:
                if option.value == default_config.diarization_model_path:
                    default_diarization_label = option.label
                    break

            selected_diarization_label = st.selectbox(
                "Diarization model",
                options=diarization_labels,
                index=_selected_index(diarization_labels, default_diarization_label),
            )
            if selected_diarization_label == "Custom path":
                diarization_model_path = st.text_input(
                    "Local diarization pipeline path",
                    value=default_config.diarization_model_path or "",
                    help="Example: models/pyannote-speaker-diarization/config.yaml",
                )
            else:
                selected_index = diarization_labels.index(selected_diarization_label)
                diarization_model_path = diarization_options[selected_index].value
        else:
            diarization_model_path = ""

        st.divider()
        st.header("LLM")
        llm_provider_config = _select_llm_provider(settings)

    uploaded_files = st.file_uploader(
        "Add audio or video files",
        type=list(settings.upload_types),
        accept_multiple_files=True,
    )

    queue_tab, history_tab = st.tabs(["Queue", "History"])

    with queue_tab:
        if uploaded_files:
            st.subheader("Queued files")
            st.table(
                [
                    {"File": uploaded_file.name, "Size MB": round(uploaded_file.size / 1024 / 1024, 2)}
                    for uploaded_file in uploaded_files
                ]
            )
        else:
            st.info("Upload one or more media files to create a transcription queue.")

        run = st.button(
            "Process queue",
            type="primary",
            disabled=not uploaded_files,
            width="stretch",
        )

        if run and uploaded_files:
            config = AppConfig(
                whisper_model=model_name_or_path.strip() or default_config.whisper_model,
                device=device,
                compute_type=compute_type,
                language=language.strip() or None,
                diarization_model_path=diarization_model_path.strip() or None,
                ffmpeg_path=default_config.ffmpeg_path,
                output_dir=default_config.output_dir,
                history_dir=default_config.history_dir,
                local_files_only=local_files_only,
                allow_cpu_fallback=default_config.allow_cpu_fallback,
                auto_install_cuda_runtime=auto_install_cuda_runtime,
                use_batched_inference=use_batched_inference,
                batch_size=batch_size,
                beam_size=beam_size,
                best_of=best_of,
                cpu_threads=cpu_threads,
                num_workers=num_workers,
                vad_filter=vad_filter,
            )
            pipeline = TranscriptionPipeline(config)
            overall_progress = st.progress(0, text="Starting queue")

            for index, uploaded_file in enumerate(uploaded_files, start=1):
                st.write(f"Processing `{uploaded_file.name}` ({index}/{len(uploaded_files)})")
                file_progress = st.progress(0, text="Preparing file")

                with tempfile.TemporaryDirectory(prefix="transcripio-ui-") as tmp_dir:
                    input_path = Path(tmp_dir) / Path(uploaded_file.name).name
                    _save_uploaded_media(uploaded_file, input_path)

                    def on_step(message: str, value: float) -> None:
                        clamped = min(max(value, 0.0), 1.0)
                        file_progress.progress(clamped, text=message)
                        overall = ((index - 1) + clamped) / len(uploaded_files)
                        overall_progress.progress(overall, text=f"Queue progress: {overall:.0%}")

                    try:
                        result = pipeline.run(input_path, on_step=on_step)
                    except Exception as exc:  # noqa: BLE001 - Streamlit should show a clean error.
                        st.error(f"{uploaded_file.name}: {exc}")
                        continue

                save_result(result, config.history_dir)
                st.session_state.results[result.job_id] = result
                st.success(f"Finished `{uploaded_file.name}`")

        if st.session_state.results:
            _show_result_editor(
                list(st.session_state.results.values())[-1],
                default_config.history_dir,
                "queue",
                llm_provider_config,
            )

    with history_tab:
        history_paths = list_history(default_config.history_dir)
        if not history_paths:
            st.info("No saved transcripts yet.")
        else:
            labels = [path.name for path in history_paths]
            selected_history = st.selectbox("Saved transcripts", labels)
            selected_path = history_paths[labels.index(selected_history)]
            loaded_result = load_result(selected_path)
            st.session_state.results[loaded_result.job_id] = loaded_result
            _show_result_editor(
                loaded_result,
                default_config.history_dir,
                "history",
                llm_provider_config,
            )


if __name__ == "__main__":
    if not _has_streamlit_context():
        _launch_with_streamlit()
    main()
