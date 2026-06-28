from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from transcripio.config import AppConfig, AppSettings, load_settings
from transcripio.health import run_environment_checks
from transcripio.media import is_supported_media
from transcripio.models import TranscriptSegment, TranscriptionResult
from transcripio.pipeline import TranscriptionPipeline
from transcripio.storage import StorageError, list_history, load_result, save_result
from transcripio_desktop.helpers import (
    EXPORT_FORMATS,
    build_desktop_config,
    export_result,
    format_file_size,
    result_metrics,
    result_title,
)


class TranscriptionWorker(QThread):
    file_started = Signal(str, int, int)
    progress_changed = Signal(str, int)
    file_finished = Signal(object)
    file_failed = Signal(str, str)
    queue_finished = Signal(bool)

    def __init__(self, files: list[Path], config: AppConfig) -> None:
        super().__init__()
        self._files = files
        self._config = config
        self._cancel_requested = False

    def cancel_after_current_file(self) -> None:
        self._cancel_requested = True

    def run(self) -> None:
        pipeline = TranscriptionPipeline(self._config)
        total = len(self._files)
        canceled = False
        for index, path in enumerate(self._files, start=1):
            if self._cancel_requested:
                canceled = True
                break
            self.file_started.emit(path.name, index, total)

            def on_step(message: str, value: float) -> None:
                percent = int(min(max(value, 0.0), 1.0) * 100)
                self.progress_changed.emit(message, percent)

            try:
                result = pipeline.run(path, on_step=on_step)
            except Exception as exc:  # noqa: BLE001 - desktop UI should show concise job errors.
                self.file_failed.emit(path.name, str(exc))
            else:
                self.file_finished.emit(result)
        self.queue_finished.emit(canceled)


class DesktopWindow(QMainWindow):
    def __init__(self, settings: AppSettings | None = None) -> None:
        super().__init__()
        self.settings = settings or load_settings()
        self.base_config = self.settings.config
        self.queue_files: list[Path] = []
        self.results: list[TranscriptionResult] = []
        self.current_result: TranscriptionResult | None = None
        self.worker: TranscriptionWorker | None = None
        self.active_run_config: AppConfig | None = None
        self._updating_segments = False

        self.setWindowTitle("Transcripio")
        self.resize(1320, 860)
        self.setMinimumSize(1060, 700)
        self.setAcceptDrops(True)

        self._build_ui()
        self._apply_style()
        self._refresh_history()

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        self._add_files(paths)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt method name.
        if self.worker is None:
            event.accept()
            return
        self._cancel_queue()
        QMessageBox.information(
            self,
            "Transcription is running",
            "The queue will stop after the current file finishes. "
            "Close Transcripio after the run returns to Ready.",
        )
        event.ignore()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._build_sidebar())
        root.addWidget(self._build_main_area(), 1)
        self.setCentralWidget(central)

    def _build_sidebar(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("sidebar")
        frame.setFixedWidth(360)
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(18, 20, 18, 20)

        title = QLabel("Transcripio")
        title.setObjectName("appTitle")
        caption = QLabel("Local transcription desktop")
        caption.setObjectName("muted")
        outer.addWidget(title)
        outer.addWidget(caption)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 16, 0, 0)

        layout.addWidget(self._build_model_group())
        layout.addWidget(self._build_runtime_group())
        layout.addWidget(self._build_diarization_group())
        layout.addWidget(self._build_storage_group())
        layout.addStretch(1)

        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        checks_button = QPushButton("Run Environment Checks")
        checks_button.clicked.connect(self._run_environment_checks)
        outer.addWidget(checks_button)
        return frame

    def _build_model_group(self) -> QGroupBox:
        group = QGroupBox("Model")
        form = QFormLayout(group)
        self.model_combo = QComboBox()
        self.model_combo.setEditable(True)
        self.model_combo.addItems(self.settings.whisper_models)
        self.model_combo.setCurrentText(self.base_config.whisper_model)
        self.local_only_checkbox = QCheckBox("Use downloaded/local files only")
        self.local_only_checkbox.setChecked(self.base_config.local_files_only)
        self.language_edit = QLineEdit(self.base_config.language or "")
        self.language_edit.setPlaceholderText("Auto")
        self.prompt_edit = QTextEdit()
        self.prompt_edit.setFixedHeight(72)
        self.prompt_edit.setPlaceholderText("Glossary / initial prompt")
        if self.base_config.initial_prompt:
            self.prompt_edit.setPlainText(self.base_config.initial_prompt)
        self.hotwords_edit = QTextEdit()
        self.hotwords_edit.setFixedHeight(58)
        self.hotwords_edit.setPlaceholderText("Important terms")
        if self.base_config.hotwords:
            self.hotwords_edit.setPlainText(self.base_config.hotwords)

        form.addRow("Whisper", self.model_combo)
        form.addRow("", self.local_only_checkbox)
        form.addRow("Language", self.language_edit)
        form.addRow("Prompt", self.prompt_edit)
        form.addRow("Hotwords", self.hotwords_edit)
        return group

    def _build_runtime_group(self) -> QGroupBox:
        group = QGroupBox("Runtime")
        form = QFormLayout(group)
        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "cuda"])
        self.device_combo.setCurrentText(self.base_config.device)
        self.compute_combo = QComboBox()
        self.compute_combo.addItems(["int8", "float16", "float32"])
        self.compute_combo.setCurrentText(self.base_config.compute_type)
        self.beam_spin = _spin_box(1, 8, self.base_config.beam_size)
        self.best_of_spin = _spin_box(1, 8, self.base_config.best_of)
        self.cpu_threads_spin = _spin_box(0, 64, self.base_config.cpu_threads)
        self.workers_spin = _spin_box(1, 8, self.base_config.num_workers)
        self.vad_checkbox = QCheckBox("Skip silence")
        self.vad_checkbox.setChecked(self.base_config.vad_filter)
        self.words_checkbox = QCheckBox("Word timestamps")
        self.words_checkbox.setChecked(self.base_config.word_timestamps)

        form.addRow("Device", self.device_combo)
        form.addRow("Compute", self.compute_combo)
        form.addRow("Beam size", self.beam_spin)
        form.addRow("Best of", self.best_of_spin)
        form.addRow("CPU threads", self.cpu_threads_spin)
        form.addRow("Workers", self.workers_spin)
        form.addRow("", self.vad_checkbox)
        form.addRow("", self.words_checkbox)
        return group

    def _build_diarization_group(self) -> QGroupBox:
        group = QGroupBox("Speakers")
        form = QFormLayout(group)
        self.diarization_checkbox = QCheckBox("Assign speakers")
        self.diarization_checkbox.setChecked(bool(self.base_config.diarization_model_path))
        self.diarization_path_edit = QLineEdit(self.base_config.diarization_model_path or "")
        diarization_browse = QPushButton("Browse")
        diarization_browse.clicked.connect(self._browse_diarization_path)
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.diarization_path_edit, 1)
        path_layout.addWidget(diarization_browse)
        form.addRow("", self.diarization_checkbox)
        form.addRow("Pipeline", path_row)
        return group

    def _build_storage_group(self) -> QGroupBox:
        group = QGroupBox("Storage")
        form = QFormLayout(group)
        self.ffmpeg_edit = QLineEdit(self.base_config.ffmpeg_path)
        self.output_dir_edit = QLineEdit(str(self.base_config.output_dir))
        self.history_dir_edit = QLineEdit(str(self.base_config.history_dir))
        form.addRow("ffmpeg", self.ffmpeg_edit)
        form.addRow("Output", self._path_row(self.output_dir_edit, self._browse_output_dir))
        form.addRow("History", self._path_row(self.history_dir_edit, self._browse_history_dir))
        return group

    def _build_main_area(self) -> QWidget:
        area = QWidget()
        layout = QVBoxLayout(area)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)

        header = QHBoxLayout()
        heading = QLabel("Desktop Workspace")
        heading.setObjectName("pageTitle")
        subheading = QLabel("Queue media, run local transcription, edit and export results.")
        subheading.setObjectName("muted")
        title_box = QVBoxLayout()
        title_box.addWidget(heading)
        title_box.addWidget(subheading)
        header.addLayout(title_box)
        header.addStretch(1)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusPill")
        header.addWidget(self.status_label)
        layout.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_queue_panel())
        splitter.addWidget(self._build_result_tabs())
        splitter.setSizes([430, 760])
        layout.addWidget(splitter, 1)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(96)
        self.log_box.setPlaceholderText("Run log")
        layout.addWidget(self.log_box)
        return area

    def _build_queue_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        title = QLabel("Queue")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        buttons = QHBoxLayout()
        add_button = QPushButton("Add Files")
        add_button.clicked.connect(self._choose_files)
        remove_button = QPushButton("Remove")
        remove_button.clicked.connect(self._remove_selected_files)
        clear_button = QPushButton("Clear")
        clear_button.clicked.connect(self._clear_queue)
        buttons.addWidget(add_button)
        buttons.addWidget(remove_button)
        buttons.addWidget(clear_button)
        layout.addLayout(buttons)

        self.queue_table = QTableWidget(0, 3)
        self.queue_table.setHorizontalHeaderLabels(["File", "Size", "Path"])
        self.queue_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.queue_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.queue_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.queue_table.verticalHeader().setVisible(False)
        self.queue_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.queue_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        layout.addWidget(self.queue_table, 1)

        self.process_button = QPushButton("Process Queue")
        self.process_button.setObjectName("primaryButton")
        self.process_button.clicked.connect(self._process_queue)
        layout.addWidget(self.process_button)
        self.cancel_button = QPushButton("Cancel Queue")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._cancel_queue)
        layout.addWidget(self.cancel_button)
        return panel

    def _build_result_tabs(self) -> QTabWidget:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_result_panel(), "Result")
        self.tabs.addTab(self._build_history_panel(), "History")
        return self.tabs

    def _build_result_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        self.result_title = QLabel("No transcript selected")
        self.result_title.setObjectName("sectionTitle")
        layout.addWidget(self.result_title)

        self.metrics_grid = QGridLayout()
        self.metric_labels: dict[str, QLabel] = {}
        for column, name in enumerate(["Language", "Duration", "Segments", "Speakers", "Created"]):
            label = QLabel(name)
            label.setObjectName("metricLabel")
            value = QLabel("-")
            value.setObjectName("metricValue")
            self.metric_labels[name] = value
            self.metrics_grid.addWidget(label, 0, column)
            self.metrics_grid.addWidget(value, 1, column)
        layout.addLayout(self.metrics_grid)

        self.segment_table = QTableWidget(0, 4)
        self.segment_table.setHorizontalHeaderLabels(["Start", "End", "Speaker", "Text"])
        self.segment_table.horizontalHeader().setSectionResizeMode(
            0,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.segment_table.horizontalHeader().setSectionResizeMode(
            1,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.segment_table.horizontalHeader().setSectionResizeMode(
            2,
            QHeaderView.ResizeMode.ResizeToContents,
        )
        self.segment_table.horizontalHeader().setSectionResizeMode(
            3,
            QHeaderView.ResizeMode.Stretch,
        )
        self.segment_table.verticalHeader().setVisible(False)
        self.segment_table.itemChanged.connect(self._segment_item_changed)
        layout.addWidget(self.segment_table, 1)

        self.preview_box = QTextEdit()
        self.preview_box.setReadOnly(True)
        self.preview_box.setFixedHeight(125)
        layout.addWidget(self.preview_box)

        actions = QHBoxLayout()
        save_edits_button = QPushButton("Save Edits")
        save_edits_button.clicked.connect(self._save_current_result)
        open_audio_button = QPushButton("Open WAV")
        open_audio_button.clicked.connect(self._open_current_audio)
        actions.addWidget(save_edits_button)
        actions.addWidget(open_audio_button)
        actions.addStretch(1)
        for format_name in EXPORT_FORMATS:
            button = QPushButton(format_name)
            button.clicked.connect(
                lambda _checked=False, name=format_name: self._export_current(name)
            )
            actions.addWidget(button)
        layout.addLayout(actions)
        return panel

    def _build_history_panel(self) -> QWidget:
        panel = QFrame()
        panel.setObjectName("panel")
        layout = QVBoxLayout(panel)
        header = QHBoxLayout()
        title = QLabel("Saved Transcripts")
        title.setObjectName("sectionTitle")
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_history)
        load = QPushButton("Load Selected")
        load.clicked.connect(self._load_selected_history)
        header.addWidget(title)
        header.addStretch(1)
        header.addWidget(refresh)
        header.addWidget(load)
        layout.addLayout(header)

        self.history_list = QListWidget()
        self.history_list.itemDoubleClicked.connect(lambda _item: self._load_selected_history())
        layout.addWidget(self.history_list, 1)
        return panel

    def _path_row(self, edit: QLineEdit, callback) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        browse = QPushButton("Browse")
        browse.clicked.connect(callback)
        layout.addWidget(edit, 1)
        layout.addWidget(browse)
        return row

    def _choose_files(self) -> None:
        filters = (
            "Media files ("
            + " ".join(f"*.{item}" for item in self.settings.upload_types)
            + ")"
        )
        paths, _selected_filter = QFileDialog.getOpenFileNames(self, "Add media files", "", filters)
        self._add_files([Path(path) for path in paths])

    def _add_files(self, paths: list[Path]) -> None:
        added = 0
        for path in paths:
            if not path.exists() or not path.is_file() or not is_supported_media(path):
                continue
            if path in self.queue_files:
                continue
            self.queue_files.append(path)
            added += 1
        if added:
            self._refresh_queue()
            self._log(f"Added {added} file(s).")

    def _refresh_queue(self) -> None:
        self.queue_table.setRowCount(len(self.queue_files))
        for row, path in enumerate(self.queue_files):
            size = format_file_size(path.stat().st_size) if path.exists() else "missing"
            for column, value in enumerate([path.name, size, str(path)]):
                self.queue_table.setItem(row, column, QTableWidgetItem(value))
        self.process_button.setEnabled(bool(self.queue_files) and self.worker is None)

    def _remove_selected_files(self) -> None:
        rows = sorted({index.row() for index in self.queue_table.selectedIndexes()}, reverse=True)
        for row in rows:
            if 0 <= row < len(self.queue_files):
                self.queue_files.pop(row)
        self._refresh_queue()

    def _clear_queue(self) -> None:
        self.queue_files.clear()
        self._refresh_queue()

    def _process_queue(self) -> None:
        if not self.queue_files:
            QMessageBox.information(self, "Queue is empty", "Add one or more media files first.")
            return

        config = self._current_config()
        self.active_run_config = config
        self.worker = TranscriptionWorker(list(self.queue_files), config)
        self.worker.file_started.connect(self._on_file_started)
        self.worker.progress_changed.connect(self._on_progress)
        self.worker.file_finished.connect(self._on_file_finished)
        self.worker.file_failed.connect(self._on_file_failed)
        self.worker.queue_finished.connect(self._on_queue_finished)
        self.process_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.status_label.setText("Running")
        self.progress_bar.setValue(0)
        self.worker.start()

    def _cancel_queue(self) -> None:
        if self.worker is None:
            return
        self.worker.cancel_after_current_file()
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Canceling after current file")
        self._log("Cancel requested. The current file will finish, then the queue will stop.")

    def _on_file_started(self, name: str, index: int, total: int) -> None:
        self._log(f"Processing {name} ({index}/{total})")

    def _on_progress(self, message: str, percent: int) -> None:
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def _on_file_finished(self, result: object) -> None:
        transcript = result
        if not isinstance(transcript, TranscriptionResult):
            self._log("Finished job returned an unsupported result.")
            return
        self.results.append(transcript)
        self._show_result(transcript)
        history_dir = (self.active_run_config or self._current_config()).history_dir
        try:
            save_result(transcript, history_dir)
        except StorageError as exc:
            self._log(f"History was not saved: {exc}")
        else:
            self._refresh_history()
        self._log(f"Finished {transcript.source_name}")

    def _on_file_failed(self, name: str, message: str) -> None:
        self._log(f"{name}: {message}")
        QMessageBox.warning(self, "Transcription failed", f"{name}: {message}")

    def _on_queue_finished(self, canceled: bool = False) -> None:
        self.worker = None
        self.active_run_config = None
        self.process_button.setEnabled(bool(self.queue_files))
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Ready")
        self.progress_bar.setValue(100)
        if canceled:
            self._log("Queue canceled after the current file.")
        else:
            self._log("Queue complete.")

    def _show_result(self, result: TranscriptionResult) -> None:
        self.current_result = result
        self.result_title.setText(result_title(result))
        for name, value in result_metrics(result).items():
            self.metric_labels[name].setText(value)

        self._updating_segments = True
        self.segment_table.setRowCount(len(result.segments))
        for row, segment in enumerate(result.segments):
            start_item = QTableWidgetItem(f"{segment.start:.3f}")
            end_item = QTableWidgetItem(f"{segment.end:.3f}")
            start_item.setFlags(start_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            end_item.setFlags(end_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.segment_table.setItem(row, 0, start_item)
            self.segment_table.setItem(row, 1, end_item)
            self.segment_table.setItem(row, 2, QTableWidgetItem(segment.speaker or ""))
            self.segment_table.setItem(row, 3, QTableWidgetItem(segment.text))
        self._updating_segments = False
        self._refresh_preview()
        self.tabs.setCurrentIndex(0)

    def _segment_item_changed(self, _item: QTableWidgetItem) -> None:
        if self._updating_segments or self.current_result is None:
            return
        self._apply_segment_table_to_result()
        self._refresh_preview()

    def _apply_segment_table_to_result(self) -> None:
        if self.current_result is None:
            return
        updated: list[TranscriptSegment] = []
        for row in range(self.segment_table.rowCount()):
            previous = (
                self.current_result.segments[row]
                if row < len(self.current_result.segments)
                else None
            )
            text = self.segment_table.item(row, 3).text().strip()
            speaker = self.segment_table.item(row, 2).text().strip() or None
            updated.append(
                TranscriptSegment(
                    start=float(self.segment_table.item(row, 0).text()),
                    end=float(self.segment_table.item(row, 1).text()),
                    speaker=speaker,
                    text=text,
                    words=previous.words if previous is not None and text == previous.text else [],
                )
            )
        self.current_result.segments = updated

    def _refresh_preview(self) -> None:
        if self.current_result is None:
            self.preview_box.clear()
            return
        from transcripio.formatters import to_txt

        self.preview_box.setPlainText(to_txt(self.current_result.segments))

    def _save_current_result(self) -> None:
        if self.current_result is None:
            return
        self._apply_segment_table_to_result()
        try:
            save_result(self.current_result, self._current_config().history_dir)
        except StorageError as exc:
            QMessageBox.warning(self, "History save failed", str(exc))
        else:
            self._refresh_history()
            self._log("Saved edits to history.")

    def _export_current(self, format_name: str) -> None:
        if self.current_result is None:
            QMessageBox.information(self, "No transcript", "Run or load a transcript first.")
            return
        self._apply_segment_table_to_result()
        try:
            artifact = export_result(self.current_result, format_name)
        except Exception as exc:  # noqa: BLE001 - export errors should be shown in the UI.
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        target, _filter = QFileDialog.getSaveFileName(self, "Export transcript", artifact.file_name)
        if not target:
            return
        try:
            Path(target).write_bytes(artifact.data)
        except OSError as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
        else:
            self._log(f"Exported {Path(target).name}")

    def _open_current_audio(self) -> None:
        if self.current_result is None or not self.current_result.audio_path.exists():
            QMessageBox.information(self, "Audio unavailable", "Prepared WAV is not available.")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.current_result.audio_path.resolve())))

    def _refresh_history(self) -> None:
        if not hasattr(self, "history_list"):
            return
        self.history_list.clear()
        for path in list_history(self._current_config().history_dir):
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            self.history_list.addItem(item)

    def _load_selected_history(self) -> None:
        item = self.history_list.currentItem()
        if item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        try:
            result = load_result(path)
        except StorageError as exc:
            QMessageBox.warning(self, "Could not load transcript", str(exc))
            return
        self.results.append(result)
        self._show_result(result)
        self._log(f"Loaded {path.name}")

    def _run_environment_checks(self) -> None:
        checks = run_environment_checks(self._current_config())
        self._log("Environment checks:")
        for check in checks:
            self._log(f"  {check.name}: {check.status} - {check.message}")

    def _browse_diarization_path(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Select diarization config.yaml",
            self.diarization_path_edit.text() or "models",
            "YAML files (*.yaml *.yml);;All files (*)",
        )
        if path:
            self.diarization_path_edit.setText(path)
            self.diarization_checkbox.setChecked(True)

    def _browse_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            self.output_dir_edit.text(),
        )
        if path:
            self.output_dir_edit.setText(path)

    def _browse_history_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(
            self,
            "Select history directory",
            self.history_dir_edit.text(),
        )
        if path:
            self.history_dir_edit.setText(path)
            self._refresh_history()

    def _current_config(self) -> AppConfig:
        diarization_path = (
            self.diarization_path_edit.text()
            if self.diarization_checkbox.isChecked()
            else ""
        )
        return build_desktop_config(
            self.base_config,
            whisper_model=self.model_combo.currentText(),
            device=self.device_combo.currentText(),
            compute_type=self.compute_combo.currentText(),
            language=self.language_edit.text(),
            diarization_model_path=diarization_path,
            ffmpeg_path=self.ffmpeg_edit.text(),
            output_dir=Path(self.output_dir_edit.text()),
            history_dir=Path(self.history_dir_edit.text()),
            local_files_only=self.local_only_checkbox.isChecked(),
            vad_filter=self.vad_checkbox.isChecked(),
            word_timestamps=self.words_checkbox.isChecked(),
            beam_size=self.beam_spin.value(),
            best_of=self.best_of_spin.value(),
            cpu_threads=self.cpu_threads_spin.value(),
            num_workers=self.workers_spin.value(),
            initial_prompt=self.prompt_edit.toPlainText(),
            hotwords=self.hotwords_edit.toPlainText(),
        )

    def _log(self, message: str) -> None:
        self.log_box.append(message)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0d0f12;
                color: #f4f4f5;
                font-size: 13px;
            }
            #sidebar {
                background: #101216;
                border-right: 1px solid #303640;
            }
            #appTitle {
                color: #f4f4f5;
                font-size: 30px;
                font-weight: 800;
            }
            #pageTitle {
                font-size: 26px;
                font-weight: 750;
            }
            #sectionTitle {
                font-size: 18px;
                font-weight: 700;
            }
            #muted, .QLabel#muted {
                color: #a8afbd;
            }
            #statusPill {
                background: #1f242c;
                border: 1px solid #303640;
                border-radius: 8px;
                padding: 7px 12px;
                color: #ffb86b;
            }
            QGroupBox, #panel {
                background: #171a20;
                border: 1px solid #303640;
                border-radius: 8px;
                margin-top: 10px;
                padding: 12px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
                color: #ffb86b;
                font-weight: 700;
            }
            QLineEdit, QTextEdit, QComboBox, QSpinBox, QTableWidget, QListWidget {
                background: #1f242c;
                border: 1px solid #303640;
                border-radius: 6px;
                color: #f4f4f5;
                selection-background-color: #f97316;
                selection-color: #111111;
            }
            QLineEdit, QComboBox, QSpinBox {
                min-height: 30px;
                padding: 3px 7px;
            }
            QPushButton {
                background: #1f242c;
                border: 1px solid #303640;
                border-radius: 7px;
                color: #f4f4f5;
                padding: 8px 11px;
                font-weight: 600;
            }
            QPushButton:hover {
                border-color: #f97316;
                color: #ffb86b;
            }
            QPushButton#primaryButton {
                background: #f97316;
                border-color: #f97316;
                color: #111111;
            }
            QPushButton#primaryButton:hover {
                background: #fb923c;
                border-color: #fb923c;
            }
            QHeaderView::section {
                background: #171a20;
                border: 0;
                border-bottom: 1px solid #303640;
                color: #a8afbd;
                padding: 8px;
                font-weight: 700;
            }
            QTabWidget::pane {
                border: 0;
            }
            QTabBar::tab {
                background: #171a20;
                border: 1px solid #303640;
                border-bottom: 0;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
                padding: 8px 14px;
                color: #a8afbd;
            }
            QTabBar::tab:selected {
                color: #ffb86b;
                border-bottom: 2px solid #f97316;
            }
            QProgressBar {
                background: #1f242c;
                border: 1px solid #303640;
                border-radius: 6px;
                height: 12px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #f97316;
                border-radius: 5px;
            }
            #metricLabel {
                color: #a8afbd;
                font-size: 12px;
            }
            #metricValue {
                color: #f4f4f5;
                font-size: 17px;
                font-weight: 700;
            }
            """
        )


def _spin_box(minimum: int, maximum: int, value: int) -> QSpinBox:
    spin = QSpinBox()
    spin.setRange(minimum, maximum)
    spin.setValue(min(max(value, minimum), maximum))
    return spin


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("Transcripio")
    window = DesktopWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
