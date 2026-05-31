from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from transcripio.config import AppConfig
from transcripio.formatters import to_docx, to_json, to_srt, to_txt, to_vtt
from transcripio.models import TranscriptSegment, TranscriptionResult
from transcripio.pipeline import TranscriptionPipeline
from transcripio.storage import list_history, load_result, save_result


def _show_result_editor(result: TranscriptionResult) -> None:
    st.subheader(result.source_name)
    if result.duration is not None:
        st.caption(f"Language: {result.language or 'unknown'} | Duration: {result.duration:.1f}s")
    else:
        st.caption(f"Language: {result.language or 'unknown'}")

    rows = [
        {
            "start": round(segment.start, 3),
            "end": round(segment.end, 3),
            "speaker": segment.speaker or "",
            "text": segment.text,
        }
        for segment in result.segments
    ]
    edited_rows = st.data_editor(
        rows,
        key=f"segments-{result.job_id}",
        use_container_width=True,
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
    save_result(result, AppConfig().history_dir)

    preview = to_txt(result.segments)
    st.text_area("Plain text preview", value=preview, height=220)

    base_name = Path(result.source_name).stem or "transcript"
    txt = to_txt(result.segments)
    srt = to_srt(result.segments)
    vtt = to_vtt(result.segments)
    json_text = to_json(result)
    docx = to_docx(result)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.download_button("TXT", txt, file_name=f"{base_name}.txt", mime="text/plain")
    col2.download_button("SRT", srt, file_name=f"{base_name}.srt", mime="application/x-subrip")
    col3.download_button("VTT", vtt, file_name=f"{base_name}.vtt", mime="text/vtt")
    col4.download_button(
        "DOCX",
        docx,
        file_name=f"{base_name}.docx",
        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    col5.download_button("JSON", json_text, file_name=f"{base_name}.json", mime="application/json")


st.set_page_config(page_title="Transcripio", page_icon="T", layout="wide")

st.title("Transcripio")
st.caption("Local audio and video transcription with optional local speaker diarization.")

if "results" not in st.session_state:
    st.session_state.results = {}

with st.sidebar:
    st.header("Models")
    model_name_or_path = st.text_input(
        "Whisper model name or local path",
        value="small",
        help="Use tiny/base/small/medium/large-v3 or a local CTranslate2 model directory.",
    )
    device = st.selectbox("Device", options=["cpu", "cuda"], index=0)
    compute_type = st.selectbox(
        "Compute type",
        options=["int8", "float16", "float32"],
        index=0,
        help="Use int8 for most CPU runs and float16 for many GPU runs.",
    )
    language = st.text_input("Language", value="en", help="Leave empty for auto-detection.")
    diarization_model_path = st.text_input(
        "Local diarization pipeline path",
        value="",
        help="Example: models/pyannote-speaker-diarization/config.yaml",
    )
    st.divider()
    st.caption("Everything runs locally. Network access is only needed if a model name has to be downloaded.")

uploaded_files = st.file_uploader(
    "Add audio or video files",
    type=["mp3", "wav", "m4a", "flac", "ogg", "mp4", "mov", "mkv", "avi", "webm"],
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
        use_container_width=True,
    )

    if run and uploaded_files:
        config = AppConfig(
            whisper_model=model_name_or_path.strip() or "small",
            device=device,
            compute_type=compute_type,
            language=language.strip() or None,
            diarization_model_path=diarization_model_path.strip() or None,
        )
        pipeline = TranscriptionPipeline(config)
        overall_progress = st.progress(0, text="Starting queue")

        for index, uploaded_file in enumerate(uploaded_files, start=1):
            st.write(f"Processing `{uploaded_file.name}` ({index}/{len(uploaded_files)})")
            file_progress = st.progress(0, text="Preparing file")

            with tempfile.TemporaryDirectory(prefix="transcripio-ui-") as tmp_dir:
                input_path = Path(tmp_dir) / uploaded_file.name
                input_path.write_bytes(uploaded_file.getbuffer())

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
        _show_result_editor(list(st.session_state.results.values())[-1])

with history_tab:
    history_dir = AppConfig().history_dir
    history_paths = list_history(history_dir)
    if not history_paths:
        st.info("No saved transcripts yet.")
    else:
        labels = [path.name for path in history_paths]
        selected_history = st.selectbox("Saved transcripts", labels)
        selected_path = history_paths[labels.index(selected_history)]
        loaded_result = load_result(selected_path)
        st.session_state.results[loaded_result.job_id] = loaded_result
        _show_result_editor(loaded_result)
