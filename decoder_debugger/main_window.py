from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from industrial_code_reader.core.engine import CodeReaderEngine
from industrial_code_reader.core.types import CodeResult, DecodeOptions, Roi
from industrial_code_reader.datamatrix.decoder import DataMatrixDecoder
from industrial_code_reader.datamatrix.roi import DataMatrixRoiGenerator
from stereo_aruco_gui.app.image_view import ImageView

from decoder_debugger.image_io import list_image_files, read_image_color
from decoder_debugger.overlay import draw_debug_overlay
from decoder_debugger.preprocess import PreprocessConfig, PreprocessResult, apply_preprocess


class DecoderDebuggerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DataMatrix Decoder Debugger")
        self.resize(1400, 850)
        self.current_path: Path | None = None
        self.current_image: np.ndarray | None = None
        self.current_folder_images: list[Path] = []
        self.current_rois: tuple[Roi, ...] = ()
        self.current_results: list[CodeResult] = []
        self.preprocess_config = PreprocessConfig()
        self.preprocess_result: PreprocessResult | None = None
        self.selected_roi_id: int | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.addTab(self._preprocess_tab(), "Preprocess")
        self.tabs.addTab(self._find_decode_tab(), "Find / Decode")
        self.tabs.addTab(self._logs_tab(), "Batch / Logs")
        self.setCentralWidget(self.tabs)

    def _preprocess_tab(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)

        controls = QVBoxLayout()
        open_image = QPushButton("Open Image")
        open_image.clicked.connect(self.open_image)
        open_folder = QPushButton("Open Folder")
        open_folder.clicked.connect(self.open_folder)
        self.preview_view_mode = QComboBox()
        self.preview_view_mode.addItems(("Original", "Preprocessed"))
        self.preview_view_mode.currentTextChanged.connect(lambda _text: self._refresh_preprocess_preview())
        self.path_label = QLabel("No image loaded")
        self.path_label.setWordWrap(True)
        controls.addWidget(open_image)
        controls.addWidget(open_folder)
        controls.addWidget(QLabel("Preview View"))
        controls.addWidget(self.preview_view_mode)
        controls.addWidget(self._operator_group())
        controls.addWidget(self.path_label)
        controls.addStretch(1)

        self.image_view = ImageView("Preprocess preview")
        self.image_view.set_zoom_enabled(True)

        layout.addLayout(controls, stretch=0)
        layout.addWidget(self.image_view, stretch=1)
        return root

    def _operator_group(self) -> QGroupBox:
        group = QGroupBox("Preprocess Operators")
        form = QFormLayout(group)
        self.channel_select = QComboBox()
        self.channel_select.addItems(("Original", "Gray", "B", "G", "R", "HSV Saturation", "HSV Value"))
        self.channel_select.currentTextChanged.connect(lambda _text: self._preprocess_controls_changed())
        self.threshold_mode = QComboBox()
        self.threshold_mode.addItems(("None", "Otsu", "Adaptive", "Manual"))
        self.threshold_mode.currentTextChanged.connect(lambda _text: self._preprocess_controls_changed())
        self.manual_threshold = QSpinBox()
        self.manual_threshold.setRange(0, 255)
        self.manual_threshold.setValue(128)
        self.manual_threshold.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        form.addRow("Channel", self.channel_select)
        form.addRow("Threshold", self.threshold_mode)
        form.addRow("Manual value", self.manual_threshold)
        return group

    def _find_decode_tab(self) -> QWidget:
        root = QWidget()
        layout = QHBoxLayout(root)
        controls = QVBoxLayout()
        self.find_input_view = QComboBox()
        self.find_input_view.addItems(("Original", "Preprocessed"))
        find_button = QPushButton("Find Code")
        find_button.clicked.connect(self.find_code)
        decode_button = QPushButton("Decode Selected ROI")
        decode_button.clicked.connect(self.decode_selected_roi)
        self.find_decode_view = ImageView("Find / Decode")
        self.find_decode_view.set_zoom_enabled(True)
        self.timing_label = QLabel("Timing: --")
        self.find_decode_result_box = QTextEdit()
        self.find_decode_result_box.setReadOnly(True)
        self.result_box = self.find_decode_result_box
        controls.addWidget(QLabel("Input View"))
        controls.addWidget(self.find_input_view)
        controls.addWidget(find_button)
        controls.addWidget(decode_button)
        controls.addStretch(1)
        layout.addLayout(controls, stretch=0)
        layout.addWidget(self.find_decode_view, stretch=1)
        inspector = QVBoxLayout()
        inspector.addWidget(QLabel("Decode Inspector"))
        inspector.addWidget(self.timing_label)
        inspector.addWidget(self.find_decode_result_box, stretch=1)
        inspector_widget = QWidget()
        inspector_widget.setLayout(inspector)
        layout.addWidget(inspector_widget, stretch=0)
        return root

    def _logs_tab(self) -> QWidget:
        root = QWidget()
        layout = QVBoxLayout(root)
        self.batch_log_box = QTextEdit()
        self.batch_log_box.setReadOnly(True)
        self.batch_log_box.setPlainText("Batch / Logs")
        layout.addWidget(self.batch_log_box)
        return root

    def _legacy_inspector_layout(self) -> QVBoxLayout:
        inspector = QVBoxLayout()
        self.timing_label = QLabel("Timing: --")
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        inspector.addWidget(QLabel("Decode Inspector"))
        inspector.addWidget(self.timing_label)
        inspector.addWidget(self.result_box, stretch=1)
        return inspector

    def _preprocess_controls_changed(self) -> None:
        self.preprocess_config = PreprocessConfig(
            view=self.preview_view_mode.currentText(),
            channel=self.channel_select.currentText(),
            threshold_mode=self.threshold_mode.currentText(),
            manual_threshold=self.manual_threshold.value(),
        )
        self._refresh_preprocess_preview()

    def _refresh_preprocess_preview(self) -> None:
        if self.current_image is None:
            return
        self.preprocess_result = apply_preprocess(self.current_image, self.preprocess_config)
        if self.preview_view_mode.currentText() == "Preprocessed":
            self.image_view.set_frame(_displayable_image(self.preprocess_result.output))
        else:
            self.image_view.set_frame(self.current_image)

    def _selected_input_image(self) -> np.ndarray | None:
        if self.current_image is None:
            return None
        if self.find_input_view.currentText() == "Preprocessed":
            if self.preprocess_result is None:
                self.preprocess_result = apply_preprocess(self.current_image, self.preprocess_config)
            return _displayable_image(self.preprocess_result.output)
        return self.current_image

    def find_code(self) -> None:
        self.find_decode_result_box.setPlainText("Find Code is not implemented yet.")

    def decode_selected_roi(self) -> None:
        self.find_decode_result_box.setPlainText("Decode Selected ROI is not implemented yet.")

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open DataMatrix Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.load_image_path(Path(path))

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Image Folder", "")
        if not folder:
            return
        self.current_folder_images = list_image_files(folder)
        if self.current_folder_images:
            self.load_image_path(self.current_folder_images[0])

    def load_image_path(self, path: str | Path) -> None:
        image_path = Path(path)
        image = read_image_color(image_path)
        if image is None:
            QMessageBox.warning(self, "Error", f"Failed to read image: {image_path}")
            return
        self.current_path = image_path
        self.current_image = image
        self.current_rois = ()
        self.current_results = []
        self.path_label.setText(str(image_path))
        self.result_box.setPlainText("Image loaded. Click Run Decode.")
        self.timing_label.setText("Timing: --")
        self.image_view.set_frame(image)
        self.preprocess_result = apply_preprocess(image, self.preprocess_config)
        self.find_decode_view.set_frame(image)

    def run_decode(self) -> None:
        if self.current_image is None:
            self.result_box.setPlainText("Load an image first.")
            return
        start = perf_counter()
        roi_start = perf_counter()
        self.current_rois = DataMatrixRoiGenerator(max_rois=32).generate(self.current_image)
        roi_elapsed_ms = (perf_counter() - roi_start) * 1000.0
        decode_start = perf_counter()
        engine = CodeReaderEngine([DataMatrixDecoder()])
        self.current_results = engine.decode(
            self.current_image,
            DecodeOptions(rois=self.current_rois[:16], max_results=16, max_rois=16, return_failures=True),
        )
        decode_elapsed_ms = (perf_counter() - decode_start) * 1000.0
        total_elapsed_ms = (perf_counter() - start) * 1000.0
        overlay = draw_debug_overlay(self.current_image, self.current_rois, self.current_results)
        self.image_view.set_frame(overlay)
        self.timing_label.setText(
            f"Total {total_elapsed_ms:.1f} ms | ROI {roi_elapsed_ms:.1f} ms | Decode {decode_elapsed_ms:.1f} ms"
        )
        self.result_box.setPlainText(self._result_text())

    def _result_text(self) -> str:
        lines = [f"Candidates: {len(self.current_rois)}", f"Results: {len(self.current_results)}"]
        for result in self.current_results:
            if result.failure_reason:
                lines.append(f"ROI {result.roi_id}: failed - {result.failure_reason}")
            else:
                lines.append(f"ROI {result.roi_id}: {result.text} ({result.symbology}, {result.preprocessing})")
        return "\n".join(lines)


def _displayable_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
