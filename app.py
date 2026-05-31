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
from transcripio.formatters import to_json, to_srt, to_txt
from transcripio.pipeline import TranscriptionPipeline


st.set_page_config(page_title="Transcripio", page_icon="T", layout="wide")

st.title("Transcripio")
st.caption("Локальная транскрибация аудио и видео с опциональным распознаванием участников.")

with st.sidebar:
    st.header("Модели")
    model_name_or_path = st.text_input("Whisper модель или путь", value="small")
    device = st.selectbox("Устройство", options=["cpu", "cuda"], index=0)
    compute_type = st.selectbox(
        "Compute type",
        options=["int8", "float16", "float32"],
        index=0,
        help="Для CPU обычно удобен int8, для GPU часто float16.",
    )
    language = st.text_input("Язык", value="ru", help="Оставьте пустым для автоопределения.")
    diarization_model_path = st.text_input(
        "Путь к локальной модели участников",
        value="",
        help="Например models/pyannote-speaker-diarization/config.yaml",
    )

uploaded_file = st.file_uploader(
    "Выберите аудио или видео",
    type=["mp3", "wav", "m4a", "flac", "ogg", "mp4", "mov", "mkv", "avi", "webm"],
)

if uploaded_file:
    st.write(f"Файл: `{uploaded_file.name}`")

run = st.button("Транскрибировать", type="primary", disabled=uploaded_file is None)

if run and uploaded_file:
    config = AppConfig(
        whisper_model=model_name_or_path.strip() or "small",
        device=device,
        compute_type=compute_type,
        language=language.strip() or None,
        diarization_model_path=diarization_model_path.strip() or None,
    )

    with tempfile.TemporaryDirectory(prefix="transcripio-ui-") as tmp_dir:
        input_path = Path(tmp_dir) / uploaded_file.name
        input_path.write_bytes(uploaded_file.getbuffer())

        progress = st.progress(0, text="Подготовка файла")
        pipeline = TranscriptionPipeline(config)

        def on_step(message: str, value: float) -> None:
            progress.progress(min(max(value, 0.0), 1.0), text=message)

        try:
            result = pipeline.run(input_path, on_step=on_step)
        except Exception as exc:  # noqa: BLE001 - UI should show a clean error.
            st.error(str(exc))
            st.stop()

    st.success("Готово")

    txt = to_txt(result.segments)
    srt = to_srt(result.segments)
    json_text = to_json(result)

    st.subheader("Текст")
    st.text_area("Результат", value=txt, height=360)

    col1, col2, col3 = st.columns(3)
    col1.download_button("Скачать TXT", txt, file_name="transcript.txt")
    col2.download_button("Скачать SRT", srt, file_name="transcript.srt")
    col3.download_button("Скачать JSON", json_text, file_name="transcript.json")
