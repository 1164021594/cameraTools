from __future__ import annotations

from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
from decoder_debugger.preprocess import (
    PreprocessConfig,
    PreprocessResult,
    apply_preprocess,
    config_from_json,
    config_to_json,
)


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
        self.preview_view_mode.currentTextChanged.connect(lambda _text: self._preprocess_controls_changed())
        self.path_label = QLabel("No image loaded")
        self.path_label.setWordWrap(True)
        controls.addWidget(open_image)
        controls.addWidget(open_folder)
        controls.addWidget(QLabel("Preview View"))
        controls.addWidget(self.preview_view_mode)
        operator_scroll = QScrollArea()
        operator_scroll.setWidgetResizable(True)
        operator_scroll.setWidget(self._operator_group())
        controls.addWidget(operator_scroll, stretch=1)
        save_preset = QPushButton("Save Preset")
        save_preset.clicked.connect(self._save_preprocess_preset_dialog)
        load_preset = QPushButton("Load Preset")
        load_preset.clicked.connect(self._load_preprocess_preset_dialog)
        controls.addWidget(save_preset)
        controls.addWidget(load_preset)
        controls.addWidget(self.path_label)

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
        self.threshold_invert = QCheckBox()
        self.threshold_invert.stateChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.adaptive_block_size = QSpinBox()
        self.adaptive_block_size.setRange(3, 201)
        self.adaptive_block_size.setSingleStep(2)
        self.adaptive_block_size.setValue(21)
        self.adaptive_block_size.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.adaptive_c = QSpinBox()
        self.adaptive_c.setRange(-30, 30)
        self.adaptive_c.setValue(4)
        self.adaptive_c.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.clahe_enabled = QCheckBox()
        self.clahe_enabled.stateChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.clahe_clip_limit = QDoubleSpinBox()
        self.clahe_clip_limit.setRange(0.1, 20.0)
        self.clahe_clip_limit.setSingleStep(0.1)
        self.clahe_clip_limit.setValue(2.0)
        self.clahe_clip_limit.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.clahe_tile_size = QSpinBox()
        self.clahe_tile_size.setRange(1, 64)
        self.clahe_tile_size.setValue(8)
        self.clahe_tile_size.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.blur_mode = QComboBox()
        self.blur_mode.addItems(("None", "Gaussian", "Median"))
        self.blur_mode.currentTextChanged.connect(lambda _text: self._preprocess_controls_changed())
        self.blur_kernel = QSpinBox()
        self.blur_kernel.setRange(1, 31)
        self.blur_kernel.setSingleStep(2)
        self.blur_kernel.setValue(3)
        self.blur_kernel.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.sharpen_enabled = QCheckBox()
        self.sharpen_enabled.stateChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.sharpen_strength = QDoubleSpinBox()
        self.sharpen_strength.setRange(1.0, 5.0)
        self.sharpen_strength.setSingleStep(0.1)
        self.sharpen_strength.setValue(1.5)
        self.sharpen_strength.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.sharpen_radius = QDoubleSpinBox()
        self.sharpen_radius.setRange(0.1, 10.0)
        self.sharpen_radius.setSingleStep(0.1)
        self.sharpen_radius.setValue(1.0)
        self.sharpen_radius.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.morphology = QComboBox()
        self.morphology.addItems(("None", "Erode", "Dilate", "Open", "Close"))
        self.morphology.currentTextChanged.connect(lambda _text: self._preprocess_controls_changed())
        self.morphology_kernel = QSpinBox()
        self.morphology_kernel.setRange(1, 31)
        self.morphology_kernel.setValue(3)
        self.morphology_kernel.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.morphology_iterations = QSpinBox()
        self.morphology_iterations.setRange(1, 10)
        self.morphology_iterations.setValue(1)
        self.morphology_iterations.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.scale_factor = QDoubleSpinBox()
        self.scale_factor.setRange(0.25, 8.0)
        self.scale_factor.setSingleStep(0.25)
        self.scale_factor.setValue(1.0)
        self.scale_factor.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        self.quiet_zone_padding = QSpinBox()
        self.quiet_zone_padding.setRange(0, 512)
        self.quiet_zone_padding.setValue(0)
        self.quiet_zone_padding.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
        form.addRow("Channel", self.channel_select)
        form.addRow("CLAHE", self.clahe_enabled)
        form.addRow("CLAHE clip", self.clahe_clip_limit)
        form.addRow("CLAHE tile", self.clahe_tile_size)
        form.addRow("Blur", self.blur_mode)
        form.addRow("Blur kernel", self.blur_kernel)
        form.addRow("Sharpen", self.sharpen_enabled)
        form.addRow("Sharpen strength", self.sharpen_strength)
        form.addRow("Sharpen radius", self.sharpen_radius)
        form.addRow("Threshold", self.threshold_mode)
        form.addRow("Manual value", self.manual_threshold)
        form.addRow("Adaptive block", self.adaptive_block_size)
        form.addRow("Adaptive C", self.adaptive_c)
        form.addRow("Invert threshold", self.threshold_invert)
        form.addRow("Morphology", self.morphology)
        form.addRow("Morph kernel", self.morphology_kernel)
        form.addRow("Morph iterations", self.morphology_iterations)
        form.addRow("Scale", self.scale_factor)
        form.addRow("Quiet zone padding", self.quiet_zone_padding)
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
            clahe_enabled=self.clahe_enabled.isChecked(),
            clahe_clip_limit=self.clahe_clip_limit.value(),
            clahe_tile_size=self.clahe_tile_size.value(),
            blur_mode=self.blur_mode.currentText(),
            blur_kernel=self.blur_kernel.value(),
            sharpen_enabled=self.sharpen_enabled.isChecked(),
            sharpen_strength=self.sharpen_strength.value(),
            sharpen_radius=self.sharpen_radius.value(),
            threshold_mode=self.threshold_mode.currentText(),
            adaptive_block_size=self.adaptive_block_size.value(),
            adaptive_c=self.adaptive_c.value(),
            manual_threshold=self.manual_threshold.value(),
            threshold_invert=self.threshold_invert.isChecked(),
            morphology=self.morphology.currentText(),
            morphology_kernel=self.morphology_kernel.value(),
            morphology_iterations=self.morphology_iterations.value(),
            scale_factor=self.scale_factor.value(),
            quiet_zone_padding=self.quiet_zone_padding.value(),
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
        image = self._selected_input_image()
        if image is None:
            self.find_decode_result_box.setPlainText("Load an image first.")
            return
        start = perf_counter()
        self.current_rois = DataMatrixRoiGenerator(max_rois=32).generate(image)
        self.selected_roi_id = self.current_rois[0].id if self.current_rois else None
        overlay = draw_debug_overlay(image, self.current_rois, [], selected_roi_id=self.selected_roi_id)
        self.find_decode_view.set_frame(overlay)
        elapsed_ms = (perf_counter() - start) * 1000.0
        self.timing_label.setText(f"Find Code: {elapsed_ms:.1f} ms")
        self.find_decode_result_box.setPlainText(f"Candidates: {len(self.current_rois)}\nFind Code: {elapsed_ms:.1f} ms")

    def decode_selected_roi(self) -> None:
        image = self._selected_input_image()
        if image is None:
            self.find_decode_result_box.setPlainText("Load an image first.")
            return
        if not self.current_rois:
            self.find_decode_result_box.setPlainText("Run Find Code first.")
            return
        selected = self._selected_roi()
        start = perf_counter()
        engine = CodeReaderEngine([DataMatrixDecoder()])
        self.current_results = engine.decode(
            image,
            DecodeOptions(rois=(selected,), max_results=16, max_rois=16, return_failures=True),
        )
        elapsed_ms = (perf_counter() - start) * 1000.0
        overlay = draw_debug_overlay(image, self.current_rois, self.current_results, selected_roi_id=selected.id)
        self.find_decode_view.set_frame(overlay)
        self.timing_label.setText(f"Total {elapsed_ms:.1f} ms | Decode Selected ROI {elapsed_ms:.1f} ms")
        self.find_decode_result_box.setPlainText(self._result_text() + f"\nDecode Selected ROI: {elapsed_ms:.1f} ms")

    def _selected_roi(self) -> Roi:
        if self.selected_roi_id is not None:
            for roi in self.current_rois:
                if roi.id == self.selected_roi_id:
                    return roi
        return self.current_rois[0]

    def save_preprocess_preset(self, path: str | Path) -> None:
        self._preprocess_controls_changed()
        Path(path).write_text(config_to_json(self.preprocess_config), encoding="utf-8")

    def load_preprocess_preset(self, path: str | Path) -> None:
        config = config_from_json(Path(path).read_text(encoding="utf-8"))
        self.preprocess_config = config
        self.preview_view_mode.setCurrentText(config.view)
        self.channel_select.setCurrentText(config.channel)
        self.clahe_enabled.setChecked(config.clahe_enabled)
        self.clahe_clip_limit.setValue(config.clahe_clip_limit)
        self.clahe_tile_size.setValue(config.clahe_tile_size)
        self.blur_mode.setCurrentText(config.blur_mode)
        self.blur_kernel.setValue(config.blur_kernel)
        self.sharpen_enabled.setChecked(config.sharpen_enabled)
        self.sharpen_strength.setValue(config.sharpen_strength)
        self.sharpen_radius.setValue(config.sharpen_radius)
        self.threshold_mode.setCurrentText(config.threshold_mode)
        self.adaptive_block_size.setValue(config.adaptive_block_size)
        self.adaptive_c.setValue(config.adaptive_c)
        self.manual_threshold.setValue(config.manual_threshold)
        self.threshold_invert.setChecked(config.threshold_invert)
        self.morphology.setCurrentText(config.morphology)
        self.morphology_kernel.setValue(config.morphology_kernel)
        self.morphology_iterations.setValue(config.morphology_iterations)
        self.scale_factor.setValue(config.scale_factor)
        self.quiet_zone_padding.setValue(config.quiet_zone_padding)
        self._refresh_preprocess_preview()

    def _save_preprocess_preset_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Preprocess Preset", "", "JSON (*.json)")
        if path:
            self.save_preprocess_preset(path)

    def _load_preprocess_preset_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Preprocess Preset", "", "JSON (*.json)")
        if path:
            self.load_preprocess_preset(path)

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
        self.find_code()
        self.decode_selected_roi()

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
