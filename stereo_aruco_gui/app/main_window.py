from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from stereo_aruco_gui.app.aruco_board import ARUCO_DICTIONARIES, detect_markers, draw_detection
from stereo_aruco_gui.app.barcode import (
    SUPPORTED_BARCODE_LABELS,
    BarcodeConfirmation,
    BarcodeDetection,
    decode_barcodes,
    draw_barcode_detections,
)
from stereo_aruco_gui.app.calibration import (
    CalibrationDatasetReport,
    CalibrationPairPreview,
    MonoCalibrationResult,
    StereoCalibrationResult,
    analyze_calibration_pairs,
    calibrate_mono_from_images,
    calibrate_from_pairs,
    calibration_quality_label,
)
from stereo_aruco_gui.app.camera_probe import scan_camera_indices
from stereo_aruco_gui.app.camera_worker import BACKEND_MAP, PIXEL_FORMATS, SingleCameraWorker, preview_copy
from stereo_aruco_gui.app.config import AppConfig, ArucoConfig, CameraConfig, save_config
from stereo_aruco_gui.app.image_view import ImageView
from stereo_aruco_gui.app.measurement_2d import plane_distance_between_pixels
from stereo_aruco_gui.app.original_image_dialog import OriginalImageDialog
from stereo_aruco_gui.app.rectification import (
    DisparityConfig,
    Rectifier,
    compute_disparity,
    disparity_preview,
    draw_depth_roi,
    distance_at,
    filtered_disparity_preview,
    draw_horizontal_guides,
)
from stereo_aruco_gui.app.storage import (
    delete_last_image_pair,
    list_image_pairs,
    load_calibration_npz,
    load_mono_calibration_npz,
    save_image_pair,
    save_single_image,
)
from stereo_aruco_gui.app.ui_state import camera_selection_warning, pair_count_text, view_mode_enabled


RESOLUTION_OPTIONS = (
    (320, 240),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 960),
    (1920, 1080),
    (2592, 1944),
)
DEFAULT_RESOLUTION = (1280, 960)

VIEW_MODES = (
    "Live Preview",
    "ArUco Detection",
    "Rectified Preview",
    "Depth / Distance",
    "2D Measurement",
    "Barcode Detection",
)
BACKEND_OPTIONS = ("AUTO", *BACKEND_MAP.keys())
DEPTH_PRESETS = {
    "Balanced": DisparityConfig(),
    "Near": DisparityConfig(
        num_disparities=256,
        block_size=5,
        uniqueness_ratio=5,
        speckle_window_size=0,
        speckle_range=2,
        disp12_max_diff=-1,
    ),
    "Far / Fast": DisparityConfig(
        num_disparities=128,
        block_size=5,
        uniqueness_ratio=10,
        speckle_window_size=100,
        speckle_range=2,
        disp12_max_diff=1,
    ),
}


def resolution_label(width: int, height: int) -> str:
    return f"{width} x {height}"


def parse_resolution_label(label: str) -> tuple[int, int]:
    width_text, height_text = label.lower().replace("*", "x").split("x", maxsplit=1)
    return int(width_text.strip()), int(height_text.strip())


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.left_worker: SingleCameraWorker | None = None
        self.right_worker: SingleCameraWorker | None = None
        self.latest_left: np.ndarray | None = None
        self.latest_right: np.ndarray | None = None
        self.frozen_left: np.ndarray | None = None
        self.frozen_right: np.ndarray | None = None
        self._freeze_enabled = False
        self.rectifier: Rectifier | None = None
        self.mono_calibration: dict[str, np.ndarray] | None = None
        self.measurement_2d_points: list[tuple[int, int]] = []
        self.barcode_confirmation = BarcodeConfirmation(required_count=3)
        self.latest_barcode_detections: list[BarcodeDetection] = []
        self._last_confirmed_barcode_key: tuple[str, str] | None = None
        self.latest_disparity: np.ndarray | None = None
        self._opening_labels: set[str] = set()
        self._original_dialogs: list[OriginalImageDialog] = []
        self._last_calibration_preview: CalibrationPairPreview | None = None
        self._calibration_review_previews: list[CalibrationPairPreview] = []
        self._updating_calibration_pair_select = False
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self._refresh_preview)

        self.setWindowTitle("Stereo ArUco Calibration")
        self.resize(1400, 820)
        self._build_ui()
        self._refresh_pair_count()

    def _any_camera_worker_running(self) -> bool:
        return (self.left_worker is not None and self.left_worker.isRunning()) or (
            self.right_worker is not None and self.right_worker.isRunning()
        )

    def closeEvent(self, event) -> None:  # noqa: ANN001
        self._stop_cameras("window close")
        super().closeEvent(event)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QGridLayout(root)
        layout.setColumnStretch(0, 0)
        layout.setColumnStretch(1, 1)
        layout.setRowStretch(0, 1)
        layout.setRowStretch(1, 0)

        sidebar = QWidget()
        sidebar.setMinimumWidth(330)
        sidebar.setMaximumWidth(390)
        controls = QVBoxLayout(sidebar)
        controls.addWidget(self._camera_group())
        controls.addWidget(self._capture_group())
        self.calibration_group = self._calibration_group()
        self.depth_group = self._depth_group()
        self.measurement_2d_group = self._measurement_2d_group()
        self.barcode_group = self._barcode_group()
        self.depth_group.hide()
        self.measurement_2d_group.hide()
        self.barcode_group.hide()
        controls.addWidget(self.calibration_group)
        controls.addWidget(self.depth_group)
        controls.addWidget(self.measurement_2d_group)
        controls.addWidget(self.barcode_group)
        controls.addWidget(self._ranging_group())
        controls.addWidget(self._diagnostics_group())
        controls.addStretch(1)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setWidget(sidebar)
        layout.addWidget(sidebar_scroll, 0, 0, 2, 1)

        preview_panel = QWidget()
        preview_panel_layout = QVBoxLayout(preview_panel)
        preview_header = QHBoxLayout()
        self.view_mode = QComboBox()
        self.view_mode.addItems(VIEW_MODES)
        self.view_mode.currentTextChanged.connect(self._view_mode_changed)
        preview_header.addWidget(QLabel("View"))
        preview_header.addWidget(self.view_mode)
        self.prev_calibration_pair_button = QPushButton("Prev Pair")
        self.prev_calibration_pair_button.clicked.connect(self._show_previous_calibration_pair)
        self.next_calibration_pair_button = QPushButton("Next Pair")
        self.next_calibration_pair_button.clicked.connect(self._show_next_calibration_pair)
        self.calibration_pair_select = QComboBox()
        self.calibration_pair_select.setMinimumWidth(190)
        self.calibration_pair_select.currentIndexChanged.connect(self._calibration_pair_selected)
        preview_header.addWidget(QLabel("Calibration Pair"))
        preview_header.addWidget(self.prev_calibration_pair_button)
        preview_header.addWidget(self.calibration_pair_select)
        preview_header.addWidget(self.next_calibration_pair_button)
        self._set_calibration_review_enabled(False)
        preview_header.addStretch(1)
        self.preview_info = QLabel("Live Preview")
        preview_header.addWidget(self.preview_info)
        preview_panel_layout.addLayout(preview_header)

        preview_layout = QGridLayout()
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setHorizontalSpacing(8)
        preview_layout.setVerticalSpacing(1)
        preview_layout.setColumnStretch(0, 1)
        preview_layout.setColumnStretch(1, 1)
        preview_layout.setRowStretch(0, 0)
        preview_layout.setRowStretch(1, 1)
        self.left_view = ImageView("Left Camera")
        self.right_view = ImageView("Right Camera")
        self.left_view.image_clicked.connect(self._handle_image_click)
        self.right_view.image_clicked.connect(self._handle_image_click)
        left_title = QLabel("Left Camera")
        right_title = QLabel("Right Camera")
        left_title.setMaximumHeight(20)
        right_title.setMaximumHeight(20)
        preview_layout.addWidget(left_title, 0, 0)
        preview_layout.addWidget(right_title, 0, 1)
        preview_layout.addWidget(self.left_view, 1, 0)
        preview_layout.addWidget(self.right_view, 1, 1)
        preview_panel_layout.addLayout(preview_layout, stretch=1)
        layout.addWidget(preview_panel, 0, 1)

        bottom_panel = QWidget()
        bottom_layout = QGridLayout(bottom_panel)
        self.left_state = QLabel("Left: closed")
        self.right_state = QLabel("Right: closed")
        self.fps_state = QLabel("FPS: --")
        self.pair_state = QLabel(pair_count_text(0))
        self.calibration_state = QLabel("Calibration: not loaded")
        self.baseline_state = QLabel("Baseline: --")
        self.distance_state = QLabel("Distance: --")
        for column, widget in enumerate(
            (
                self.left_state,
                self.right_state,
                self.fps_state,
                self.pair_state,
                self.calibration_state,
                self.baseline_state,
                self.distance_state,
            )
        ):
            widget.setMinimumWidth(120)
            bottom_layout.addWidget(widget, 0, column)
        self.status_box = QTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setMaximumHeight(96)
        bottom_layout.addWidget(self.status_box, 1, 0, 1, 7)
        layout.addWidget(bottom_panel, 1, 1)

        self.setCentralWidget(root)
        self._refresh_diagnostics()

    def _camera_group(self) -> QGroupBox:
        group = QGroupBox("1. Camera Connection")
        form = QFormLayout(group)
        cam = self.config.camera
        self.left_index = self._camera_combo(cam.left_index)
        self.right_index = self._camera_combo(cam.right_index)
        self.left_index.currentIndexChanged.connect(self._refresh_diagnostics)
        self.right_index.currentIndexChanged.connect(self._refresh_diagnostics)
        self.resolution = self._resolution_combo(cam.width, cam.height)
        self.fps = self._spin(1, 120, cam.fps)
        self.backend = QComboBox()
        self.backend.addItems(list(BACKEND_OPTIONS))
        self.backend.setCurrentText(cam.backend if cam.backend in BACKEND_OPTIONS else "AUTO")
        self.pixel_format = QComboBox()
        self.pixel_format.addItems(list(PIXEL_FORMATS))
        self.pixel_format.setCurrentText(cam.pixel_format)
        self.single_mode = QCheckBox("Single camera mode")
        self.single_mode.setChecked(cam.single_mode)
        self.single_mode.stateChanged.connect(self._refresh_diagnostics)
        self.camera_stats = QLabel("left fps: -- | right fps: --")
        self.open_button = QPushButton("Open Cameras")
        self.open_button.clicked.connect(self._toggle_cameras)
        self.scan_button = QPushButton("Scan Cameras")
        self.scan_button.clicked.connect(self._scan_cameras)
        self.swap_button = QPushButton("Swap Left/Right")
        self.swap_button.clicked.connect(self._swap_cameras)
        save_btn = QPushButton("Save Config")
        save_btn.clicked.connect(self._save_current_config)
        form.addRow("Left camera", self.left_index)
        form.addRow("Right camera", self.right_index)
        form.addRow("Resolution", self.resolution)
        form.addRow("FPS", self.fps)
        form.addRow("Backend", self.backend)
        form.addRow("Format", self.pixel_format)
        form.addRow("", self.single_mode)
        form.addRow("Stats", self.camera_stats)
        form.addRow(self.scan_button, self.open_button)
        form.addRow(self.swap_button, save_btn)
        return group

    def _capture_group(self) -> QGroupBox:
        group = QGroupBox("2. Capture")
        layout = QVBoxLayout(group)
        self.capture_button = QPushButton("Capture Pair")
        self.capture_button.clicked.connect(self._capture_pair)
        self.delete_pair_button = QPushButton("Delete Last Pair")
        self.delete_pair_button.clicked.connect(self._delete_last_pair)
        self.open_capture_dir_button = QPushButton("Open Capture Directory")
        self.open_capture_dir_button.clicked.connect(self._open_capture_dir)
        self.pair_count = QLabel(pair_count_text(0))
        self.last_pair_label = QLabel("Last pair: --")
        layout.addWidget(self.capture_button)
        layout.addWidget(self.delete_pair_button)
        layout.addWidget(self.open_capture_dir_button)
        layout.addWidget(self.pair_count)
        layout.addWidget(self.last_pair_label)
        return group

    def _calibration_group(self) -> QGroupBox:
        group = QGroupBox("3. Calibration")
        form = QFormLayout(group)
        aruco = self.config.aruco
        self.dictionary = QComboBox()
        self.dictionary.addItems(sorted(ARUCO_DICTIONARIES))
        self.dictionary.setCurrentText(aruco.dictionary)
        self.markers_x = self._spin(1, 30, aruco.markers_x)
        self.markers_y = self._spin(1, 30, aruco.markers_y)
        self.marker_length = self._double_spin(0.001, 1.0, aruco.marker_length_m)
        self.marker_separation = self._double_spin(0.0, 1.0, aruco.marker_separation_m)
        self.board_mirror_x = QCheckBox("Mirror board X")
        self.board_mirror_x.setChecked(aruco.board_mirror_x)
        form.addRow("Dictionary", self.dictionary)
        form.addRow("Markers X", self.markers_x)
        form.addRow("Markers Y", self.markers_y)
        form.addRow("Marker length m", self.marker_length)
        form.addRow("Separation m", self.marker_separation)
        form.addRow("", self.board_mirror_x)
        self.calibrate_button = QPushButton("Start Calibration")
        self.calibrate_button.clicked.connect(self._calibrate)
        self.load_button = QPushButton("Load Calibration")
        self.load_button.clicked.connect(self._load_calibration)
        self.calibration_metrics = QLabel("Errors: --")
        self.calibration_metrics.setWordWrap(True)
        self.output_path_label = QLabel(f"Output: {self.config.output.dir}")
        self.output_path_label.setWordWrap(True)
        form.addRow(self.calibrate_button, self.load_button)
        form.addRow("Result", self.calibration_metrics)
        form.addRow("Output", self.output_path_label)
        return group

    def _depth_group(self) -> QGroupBox:
        group = QGroupBox("3. Depth Parameters")
        form = QFormLayout(group)
        self.depth_preset = QComboBox()
        self.depth_preset.addItems(list(DEPTH_PRESETS))
        self.depth_preset.setCurrentText("Balanced")
        self.depth_num_disparities = self._value_combo((128, 192, 256, 320), 192)
        self.depth_block_size = self._value_combo((3, 5, 7, 9, 11), 7)
        self.depth_uniqueness_ratio = self._value_combo((5, 10, 15), 10)
        self.depth_speckle_window_size = self._value_combo((0, 50, 100, 200), 100)
        self.depth_speckle_range = self._value_combo((1, 2, 4), 2)
        self.depth_disp12_max_diff = self._value_combo((-1, 1, 2, 5), 1)
        self.depth_display_mode = QComboBox()
        self.depth_display_mode.addItems(("Raw", "Filtered"))
        self.depth_preset.currentTextChanged.connect(self._apply_depth_preset)
        form.addRow("Preset", self.depth_preset)
        form.addRow("Display", self.depth_display_mode)
        form.addRow("numDisparities", self.depth_num_disparities)
        form.addRow("blockSize", self.depth_block_size)
        form.addRow("uniquenessRatio", self.depth_uniqueness_ratio)
        form.addRow("speckleWindowSize", self.depth_speckle_window_size)
        form.addRow("speckleRange", self.depth_speckle_range)
        form.addRow("disp12MaxDiff", self.depth_disp12_max_diff)
        hint = QLabel("Changes apply to the next depth frame.")
        hint.setWordWrap(True)
        form.addRow("", hint)
        return group

    def _measurement_2d_group(self) -> QGroupBox:
        group = QGroupBox("3. 2D Measurement")
        form = QFormLayout(group)
        self.plane_distance_mm = QDoubleSpinBox()
        self.plane_distance_mm.setRange(1.0, 100000.0)
        self.plane_distance_mm.setDecimals(2)
        self.plane_distance_mm.setSingleStep(10.0)
        self.plane_distance_mm.setValue(500.0)
        self.load_mono_button = QPushButton("Load Mono Calibration")
        self.load_mono_button.clicked.connect(self._load_mono_calibration)
        self.clear_measurement_2d_button = QPushButton("Clear 2D Points")
        self.clear_measurement_2d_button.clicked.connect(self._clear_2d_measurement)
        self.measurement_2d_info = QLabel("Load mono calibration, enter plane distance, then click two points.")
        self.measurement_2d_info.setWordWrap(True)
        form.addRow("Plane distance mm", self.plane_distance_mm)
        form.addRow(self.load_mono_button, self.clear_measurement_2d_button)
        form.addRow("Result", self.measurement_2d_info)
        return group

    def _barcode_group(self) -> QGroupBox:
        group = QGroupBox("3. Barcode Detection")
        form = QFormLayout(group)
        self.barcode_format = QComboBox()
        self.barcode_format.addItems(("All", *SUPPORTED_BARCODE_LABELS))
        self.barcode_format.setCurrentText("All")
        self.barcode_confirm_frames = self._spin(1, 20, 3)
        self.barcode_info = QLabel("Select barcode type and show a barcode in the left/current camera view.")
        self.barcode_info.setWordWrap(True)
        self.clear_barcode_button = QPushButton("Clear Barcode Result")
        self.clear_barcode_button.clicked.connect(self._clear_barcode_result)
        form.addRow("Type", self.barcode_format)
        form.addRow("Confirm frames", self.barcode_confirm_frames)
        form.addRow("", self.clear_barcode_button)
        form.addRow("Result", self.barcode_info)
        return group

    def _ranging_group(self) -> QGroupBox:
        group = QGroupBox("4. Result And Ranging")
        layout = QVBoxLayout(group)
        self.detect_button = QPushButton("Detect Current")
        self.detect_button.clicked.connect(self._detect_current)
        self.freeze_button = QPushButton("Freeze Frame")
        self.freeze_button.clicked.connect(self._toggle_freeze)
        self.view_left_original_button = QPushButton("View Left Original")
        self.view_left_original_button.clicked.connect(lambda: self._open_original_view("left"))
        self.view_right_original_button = QPushButton("View Right Original")
        self.view_right_original_button.clicked.connect(lambda: self._open_original_view("right"))
        self.guides_check = QCheckBox("Horizontal guides")
        self.guides_check.setChecked(True)
        self.ranging_hint = QLabel("Depth and rectified views require calibration.")
        layout.addWidget(self.detect_button)
        layout.addWidget(self.freeze_button)
        layout.addWidget(self.view_left_original_button)
        layout.addWidget(self.view_right_original_button)
        layout.addWidget(self.guides_check)
        layout.addWidget(self.ranging_hint)
        return group

    def _diagnostics_group(self) -> QGroupBox:
        group = QGroupBox("5. Light Diagnostics")
        layout = QVBoxLayout(group)
        self.diagnostics_label = QLabel("Ready")
        self.diagnostics_label.setWordWrap(True)
        layout.addWidget(self.diagnostics_label)
        return group

    def _spin(self, minimum: int, maximum: int, value: int) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _value_combo(self, values: tuple[int, ...], current: int) -> QComboBox:
        combo = QComboBox()
        for value in values:
            combo.addItem(str(value), value)
        combo.setCurrentText(str(current if current in values else values[0]))
        return combo

    def _camera_combo(self, current_index: int) -> QComboBox:
        combo = QComboBox()
        combo.setEditable(True)
        for index in range(21):
            combo.addItem(f"index {index}", index)
        combo.setCurrentIndex(max(0, min(current_index, 20)))
        return combo

    def _resolution_combo(self, width: int, height: int) -> QComboBox:
        combo = QComboBox()
        options = list(RESOLUTION_OPTIONS)
        current = (width, height)
        for option_width, option_height in options:
            combo.addItem(resolution_label(option_width, option_height))
        selected = current if current in options else DEFAULT_RESOLUTION
        combo.setCurrentText(resolution_label(*selected))
        return combo

    def _double_spin(self, minimum: float, maximum: float, value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setDecimals(4)
        spin.setSingleStep(0.001)
        spin.setValue(value)
        return spin

    def _current_camera_config(self) -> CameraConfig:
        width, height = parse_resolution_label(self.resolution.currentText())
        return CameraConfig(
            left_index=self._selected_camera_index(self.left_index),
            right_index=self._selected_camera_index(self.right_index),
            width=width,
            height=height,
            fps=self.fps.value(),
            single_mode=self.single_mode.isChecked(),
            backend=self.backend.currentText(),
            pixel_format=self.pixel_format.currentText(),
        )

    def _current_disparity_config(self) -> DisparityConfig:
        return DisparityConfig(
            num_disparities=int(self.depth_num_disparities.currentText()),
            block_size=int(self.depth_block_size.currentText()),
            uniqueness_ratio=int(self.depth_uniqueness_ratio.currentText()),
            speckle_window_size=int(self.depth_speckle_window_size.currentText()),
            speckle_range=int(self.depth_speckle_range.currentText()),
            disp12_max_diff=int(self.depth_disp12_max_diff.currentText()),
        ).normalized()

    def _set_combo_value(self, combo: QComboBox, value: int) -> None:
        text = str(value)
        if combo.findText(text) < 0:
            combo.addItem(text, value)
        combo.setCurrentText(text)

    def _apply_depth_preset(self, preset_name: str) -> None:
        config = DEPTH_PRESETS.get(preset_name)
        if config is None:
            return
        self._set_combo_value(self.depth_num_disparities, config.num_disparities)
        self._set_combo_value(self.depth_block_size, config.block_size)
        self._set_combo_value(self.depth_uniqueness_ratio, config.uniqueness_ratio)
        self._set_combo_value(self.depth_speckle_window_size, config.speckle_window_size)
        self._set_combo_value(self.depth_speckle_range, config.speckle_range)
        self._set_combo_value(self.depth_disp12_max_diff, config.disp12_max_diff)

    def _selected_camera_index(self, combo: QComboBox) -> int:
        data = combo.currentData()
        if isinstance(data, int):
            return data
        text = combo.currentText().strip()
        if text.startswith("index "):
            text = text.split()[1]
        return int(text)

    def _current_aruco_config(self) -> ArucoConfig:
        return ArucoConfig(
            dictionary=self.dictionary.currentText(),
            markers_x=self.markers_x.value(),
            markers_y=self.markers_y.value(),
            marker_length_m=self.marker_length.value(),
            marker_separation_m=self.marker_separation.value(),
            board_mirror_x=self.board_mirror_x.isChecked(),
        )

    def _toggle_cameras(self) -> None:
        if self.open_button.text() == "Stopping Cameras":
            return
        if self._any_camera_worker_running():
            self._stop_cameras("Open Cameras button toggled to close")
            return
        camera = self._current_camera_config()
        warning = camera_selection_warning(camera.left_index, camera.right_index, camera.single_mode)
        if warning is not None:
            self._set_diagnostic(warning)
            self._log(warning)
            return
        self.left_worker = SingleCameraWorker(
            camera.left_index,
            camera.width,
            camera.height,
            camera.fps,
            "left",
            camera.backend,
            camera.pixel_format,
        )
        self.left_worker.status.connect(self._handle_camera_status)
        self.left_worker.finished.connect(self._camera_worker_finished)
        self._opening_labels = {"left"}
        self.left_worker.start()
        if not camera.single_mode:
            self.right_worker = SingleCameraWorker(
                camera.right_index,
                camera.width,
                camera.height,
                camera.fps,
                "right",
                camera.backend,
                camera.pixel_format,
            )
            self.right_worker.status.connect(self._handle_camera_status)
            self.right_worker.finished.connect(self._camera_worker_finished)
            self._opening_labels.add("right")
            self.right_worker.start()
        self.preview_timer.start()
        self.open_button.setText("Opening Cameras")
        self.open_button.setEnabled(False)
        self.left_state.setText(f"Left: opening index {camera.left_index}")
        self.right_state.setText("Right: single mode" if camera.single_mode else f"Right: opening index {camera.right_index}")
        self._log("Opening cameras. Close Windows Camera or any other program using the cameras if no frames appear.")

    def _handle_camera_status(self, message: str) -> None:
        self._log(message)
        is_opened = ": opened index " in message
        is_open_failure = ": failed to open camera " in message or ": camera thread error:" in message
        if is_opened or is_open_failure:
            label = message.split(":", maxsplit=1)[0]
            self._opening_labels.discard(label)
        if is_open_failure and self.open_button.text() == "Opening Cameras":
            self._stop_cameras(f"{message.split(':', maxsplit=1)[0]} failed during stereo open")
            self.left_worker = None
            self.right_worker = None
            self.preview_timer.stop()
            self._opening_labels.clear()
            self.open_button.setText("Open Cameras")
            self.open_button.setEnabled(True)
            self.left_state.setText("Left: closed")
            self.right_state.setText("Right: closed")
            self.fps_state.setText("FPS: --")
            return
        if not self._opening_labels and self.open_button.text() == "Opening Cameras":
            self.open_button.setText("Close Cameras")
            self.open_button.setEnabled(True)

    def _stop_cameras(self, reason: str = "unspecified") -> None:
        self._log(f"Stopping cameras: {reason}")
        still_running = False
        if self.left_worker:
            if self.left_worker.stop():
                self.left_worker = None
            else:
                still_running = True
        if self.right_worker:
            if self.right_worker.stop():
                self.right_worker = None
            else:
                still_running = True
        self.preview_timer.stop()
        self._opening_labels.clear()
        if still_running:
            self.open_button.setText("Stopping Cameras")
            self.open_button.setEnabled(False)
            self._log("Camera backend is still stopping; waiting for thread cleanup")
        else:
            self.open_button.setText("Open Cameras")
            self.open_button.setEnabled(True)
        self.left_state.setText("Left: closed")
        self.right_state.setText("Right: closed")
        self.fps_state.setText("FPS: --")

    def _camera_worker_finished(self) -> None:
        left_running = self.left_worker is not None and self.left_worker.isRunning()
        right_running = self.right_worker is not None and self.right_worker.isRunning()
        if left_running or right_running:
            return
        self.left_worker = None
        self.right_worker = None
        self._opening_labels.clear()
        self.preview_timer.stop()
        self.open_button.setText("Open Cameras")
        self.open_button.setEnabled(True)
        self.left_state.setText("Left: closed")
        self.right_state.setText("Right: closed")
        self.fps_state.setText("FPS: --")

    def _refresh_preview(self) -> None:
        if self._freeze_enabled:
            if self.frozen_left is not None:
                frozen_right = self.frozen_right if self.right_worker is not None else None
                self._update_frames(self.frozen_left, frozen_right, freeze_preview=True)
                self._update_camera_stats()
            return
        if self.left_worker is None:
            return
        left = self.left_worker.get_latest()
        right = None if self.right_worker is None else self.right_worker.get_latest()
        if left is None:
            if self.right_worker is not None and right is not None:
                self.latest_left = None
                self.latest_right = right
                self.left_view.setText("Waiting for left camera")
                self.right_view.set_frame(preview_copy(right))
                self._update_camera_stats()
            return
        if self.right_worker is not None and right is None:
            self.left_view.set_frame(preview_copy(left))
            self.right_view.setText("Waiting for right camera")
            self._update_camera_stats()
            return
        self._update_frames(left, right)
        self._update_camera_stats()

    def _update_frames(self, left: np.ndarray, right: np.ndarray | None, freeze_preview: bool = False) -> None:
        if not freeze_preview:
            self.latest_left = left
            self.latest_right = right
        freeze_preview = freeze_preview or self._freeze_enabled
        show_left, show_right = left, right
        mode = self.view_mode.currentText()
        if right is None:
            if mode == "2D Measurement":
                show_left = self._draw_2d_measurement_overlay(show_left)
            elif mode == "Barcode Detection":
                show_left = self._process_barcode_frame(show_left)
            left_preview = preview_copy(show_left)
            self.left_view.set_frame(left_preview)
            self.left_view.set_source_size(show_left.shape[1], show_left.shape[0])
            self.right_view.setText("Single camera mode")
            single_preview_mode = mode if mode in ("2D Measurement", "Barcode Detection") else "Live Preview"
            self.preview_info.setText("Frozen Preview" if freeze_preview else single_preview_mode)
            return
        if not view_mode_enabled(mode, self.rectifier is not None):
            self.preview_info.setText("Calibration required")
            self.latest_disparity = None
            left_preview = preview_copy(left)
            right_preview = preview_copy(right)
            self.left_view.set_frame(left_preview)
            self.left_view.set_source_size(left.shape[1], left.shape[0])
            self.right_view.set_frame(right_preview)
            self.right_view.set_source_size(right.shape[1], right.shape[0])
            return
        self.preview_info.setText("Frozen Preview" if freeze_preview else mode)
        if mode == "ArUco Detection":
            try:
                aruco = self._current_aruco_config()
                left_detection = detect_markers(left, aruco)
                right_detection = detect_markers(right, aruco)
                show_left = draw_detection(left, left_detection)
                show_right = draw_detection(right, right_detection)
                self.calibration_metrics.setText(f"Markers: L {left_detection.count} / R {right_detection.count}")
            except Exception as exc:  # noqa: BLE001
                self._set_diagnostic(str(exc))
        elif mode == "Rectified Preview" and self.rectifier is not None:
            show_left, show_right = self.rectifier.rectify(left, right)
            if self.guides_check.isChecked():
                show_left = draw_horizontal_guides(show_left)
                show_right = draw_horizontal_guides(show_right)
            self.latest_disparity = None
        elif mode == "Depth / Distance" and self.rectifier is not None:
            left_rect, right_rect = self.rectifier.rectify(left, right)
            disparity_config = self._current_disparity_config()
            self.latest_disparity = compute_disparity(left_rect, right_rect, disparity_config)
            valid_x = min(disparity_config.num_disparities, max(self.latest_disparity.shape[1] - 1, 0))
            preview_fn = filtered_disparity_preview if self.depth_display_mode.currentText() == "Filtered" else disparity_preview
            show_left = preview_fn(self.latest_disparity)[:, valid_x:]
            show_right = draw_depth_roi(right_rect, valid_x)
            if self.guides_check.isChecked():
                show_right = draw_horizontal_guides(show_right)
        elif mode == "2D Measurement":
            show_left = self._draw_2d_measurement_overlay(show_left)
            self.latest_disparity = None
        elif mode == "Barcode Detection":
            show_left = self._process_barcode_frame(show_left)
            self.latest_disparity = None
        else:
            self.latest_disparity = None
        left_preview = preview_copy(show_left)
        right_preview = preview_copy(show_right)
        self.left_view.set_frame(left_preview)
        self.left_view.set_source_size(show_left.shape[1], show_left.shape[0])
        self.right_view.set_frame(right_preview)
        self.right_view.set_source_size(show_right.shape[1], show_right.shape[0])
        if mode == "Depth / Distance" and self.latest_disparity is not None:
            self.left_view.set_source_rect(valid_x, 0, show_left.shape[1], show_left.shape[0])

    def _detect_current(self) -> None:
        if self.latest_left is None:
            self._log("No camera frames available")
            return
        try:
            aruco = self._current_aruco_config()
            left = detect_markers(self.latest_left, aruco)
            if self.latest_right is None:
                self.left_view.set_frame(draw_detection(self.latest_left, left))
                self._log(f"Detected markers: left={left.count}")
                return
            right = detect_markers(self.latest_right, aruco)
            self.left_view.set_frame(draw_detection(self.latest_left, left))
            self.right_view.set_frame(draw_detection(self.latest_right, right))
            self._log(f"Detected markers: left={left.count}, right={right.count}")
            self.view_mode.setCurrentText("ArUco Detection")
        except Exception as exc:  # noqa: BLE001
            self._error(str(exc))

    def _scan_cameras(self) -> None:
        if self.left_worker and self.left_worker.isRunning():
            self._stop_cameras("Scan Cameras requested")
        self._log("Scanning camera indexes 0-5 at 640x480...")
        try:
            results = scan_camera_indices(width=640, height=480, fps=30)
            readable_by_index = {}
            for result in results:
                if result.read_ok:
                    readable_by_index.setdefault(result.index, result)
                    self._log(
                        f"{result.backend_name} index {result.index}: OK "
                        f"{result.actual_width:.0f}x{result.actual_height:.0f}@{result.actual_fps:.1f} "
                        f"shape={result.shape}"
                    )
            self._populate_camera_combos(readable_by_index)
            if not any(result.read_ok for result in results):
                self._set_diagnostic("No readable camera found. Check Windows privacy permission and close other camera apps.")
                self._log("No readable camera found. Check Windows privacy permission and close other camera apps.")
        except Exception as exc:  # noqa: BLE001
            self._error(str(exc))

    def _swap_cameras(self) -> None:
        left = self.left_index.currentIndex()
        right = self.right_index.currentIndex()
        self.left_index.setCurrentIndex(right)
        self.right_index.setCurrentIndex(left)
        self._log("Swapped left/right camera selectors")

    def _populate_camera_combos(self, readable_by_index) -> None:  # noqa: ANN001
        if not readable_by_index:
            return

        current_left = self._selected_camera_index(self.left_index)
        current_right = self._selected_camera_index(self.right_index)
        for combo, current in ((self.left_index, current_left), (self.right_index, current_right)):
            combo.clear()
            for index, result in sorted(readable_by_index.items()):
                combo.addItem(
                    f"index {index} ({result.backend_name}, {result.actual_width:.0f}x{result.actual_height:.0f})",
                    index,
                )
            match = combo.findData(current)
            combo.setCurrentIndex(match if match >= 0 else 0)

        if self.left_index.count() >= 1:
            self.left_index.setCurrentIndex(0)
        if self.right_index.count() >= 2:
            self.right_index.setCurrentIndex(1)
        self._log("Camera selectors updated from scan results")

    def _capture_pair(self) -> None:
        if self.single_mode.isChecked():
            if self.latest_left is None:
                self._log("No camera frames available")
                return
            try:
                image = save_single_image(self.config.capture.output_dir, self.latest_left)
                self.last_pair_label.setText(f"Last single: {image.path.name}")
                self._log(f"Saved single image {image.index:04d}")
            except Exception as exc:  # noqa: BLE001
                self._error(str(exc))
            return
        if self.latest_left is None or self.latest_right is None:
            self._log("No camera frames available")
            return
        try:
            pair = save_image_pair(self.config.capture.output_dir, self.latest_left, self.latest_right)
            self._refresh_pair_count()
            self.last_pair_label.setText(f"Last pair: {pair.left_path.name}")
            self._log(f"Saved pair {pair.index:04d}")
        except Exception as exc:  # noqa: BLE001
            self._error(str(exc))

    def _delete_last_pair(self) -> None:
        pair = delete_last_image_pair(self.config.capture.output_dir)
        self._refresh_pair_count()
        if pair is None:
            self._log("No captured pair to delete")
            return
        self.last_pair_label.setText(f"Deleted pair: {pair.left_path.name}")
        self._log(f"Deleted pair {pair.index:04d}")

    def _open_capture_dir(self) -> None:
        path = Path(self.config.capture.output_dir).resolve()
        path.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        self._log(f"Opened capture directory: {path}")

    def _calibrate(self) -> None:
        if self.single_mode.isChecked():
            self._calibrate_mono()
            return
        was_preview_running = self.preview_timer.isActive()
        try:
            aruco = self._current_aruco_config()
            self.calibrate_button.setEnabled(False)
            self.load_button.setEnabled(False)
            review_previews: list[CalibrationPairPreview] = []
            self._set_calibration_review_items([])
            if was_preview_running:
                self.preview_timer.stop()
            self._log("Calibration started")
            self._log(
                "Board: "
                f"{aruco.dictionary}, {aruco.markers_x}x{aruco.markers_y}, "
                f"tag={aruco.marker_length_m:.3f} m, spacing={aruco.marker_separation_m:.3f} m, "
                f"mirror_x={aruco.board_mirror_x}"
            )
            total_pairs = len(list_image_pairs(self.config.capture.output_dir))
            report = analyze_calibration_pairs(
                self.config.capture.output_dir,
                aruco,
                on_pair=lambda preview: (review_previews.append(preview), self._show_calibration_progress(preview, total_pairs)),
            )
            self._show_calibration_report(report)
            self._set_calibration_review_items(review_previews)
            result = calibrate_from_pairs(
                self.config.capture.output_dir,
                self.config.output.dir,
                aruco,
            )
            self._set_rectifier_from_result(result)
            baseline = float(np.linalg.norm(result.T))
            quality = calibration_quality_label(result.stereo_error)
            self._show_calibration_result(result, baseline, quality)
            self.calibration_metrics.setText(
                f"Pairs {report.valid_pairs}/{report.total_pairs} valid | "
                f"L {result.left_error:.4f} / R {result.right_error:.4f} / stereo {result.stereo_error:.4f} | {quality}"
            )
            self.calibration_state.setText(f"Calibration: stereo error {result.stereo_error:.4f}")
            self.baseline_state.setText(f"Baseline: {baseline:.4f} m")
            self.output_path_label.setText(f"{Path(self.config.output.dir) / 'stereo_calib.npz'}")
            self._log(
                "Calibration done\n"
                f"valid pairs: {report.valid_pairs}/{report.total_pairs}\n"
                f"left error: {result.left_error:.4f}\n"
                f"right error: {result.right_error:.4f}\n"
                f"stereo error: {result.stereo_error:.4f}\n"
                f"baseline: {baseline:.4f} m\n"
                f"quality: {quality}"
            )
        except Exception as exc:  # noqa: BLE001
            self._error(str(exc))
        finally:
            self.calibrate_button.setEnabled(True)
            self.load_button.setEnabled(True)
            if was_preview_running and self._any_camera_worker_running():
                self.preview_timer.start()

    def _calibrate_mono(self) -> None:
        was_preview_running = self.preview_timer.isActive()
        try:
            aruco = self._current_aruco_config()
            self.calibrate_button.setEnabled(False)
            self.load_button.setEnabled(False)
            if was_preview_running:
                self.preview_timer.stop()
            self._log("Mono calibration started")
            self._log(
                "Board: "
                f"{aruco.dictionary}, {aruco.markers_x}x{aruco.markers_y}, "
                f"tag={aruco.marker_length_m:.3f} m, spacing={aruco.marker_separation_m:.3f} m, "
                f"mirror_x={aruco.board_mirror_x}"
            )
            result = calibrate_mono_from_images(
                self.config.capture.output_dir,
                self.config.output.dir,
                aruco,
            )
            self._show_mono_calibration_result(result)
        except Exception as exc:  # noqa: BLE001
            self._error(str(exc))
        finally:
            self.calibrate_button.setEnabled(True)
            self.load_button.setEnabled(True)
            if was_preview_running and self._any_camera_worker_running():
                self.preview_timer.start()

    def _show_mono_calibration_result(self, result: MonoCalibrationResult) -> None:
        self.rectifier = None
        self.latest_disparity = None
        self.mono_calibration = result.as_dict()
        self.calibration_metrics.setText(f"Mono images {result.valid_images} | error {result.error:.4f}")
        self.calibration_state.setText(f"Calibration: mono error {result.error:.4f}")
        self.baseline_state.setText("Baseline: mono N/A")
        self.output_path_label.setText(f"{Path(self.config.output.dir) / 'mono_calib.npz'}")
        self._log(
            "Mono calibration done\n"
            f"valid images: {result.valid_images}\n"
            f"error: {result.error:.4f}\n"
            f"output: {Path(self.config.output.dir) / 'mono_calib.npz'}"
        )

    def _show_calibration_progress(self, preview: CalibrationPairPreview, total_pairs: int) -> None:
        self._last_calibration_preview = preview
        stats = preview.stats
        self._display_calibration_preview(preview, total_pairs, "checking")
        QApplication.processEvents()

    def _set_calibration_review_items(self, previews: list[CalibrationPairPreview]) -> None:
        self._calibration_review_previews = list(previews)
        self._updating_calibration_pair_select = True
        self.calibration_pair_select.clear()
        for preview in self._calibration_review_previews:
            stats = preview.stats
            self.calibration_pair_select.addItem(
                f"{stats.index:04d} {stats.status} | L {stats.left_markers} R {stats.right_markers} common {stats.common_points}"
            )
        self._updating_calibration_pair_select = False
        self._set_calibration_review_enabled(bool(self._calibration_review_previews))
        if self._calibration_review_previews:
            self.calibration_pair_select.setCurrentIndex(0)
            self._show_calibration_review_pair(0)

    def _set_calibration_review_enabled(self, enabled: bool) -> None:
        self.calibration_pair_select.setEnabled(enabled)
        self.prev_calibration_pair_button.setEnabled(False)
        self.next_calibration_pair_button.setEnabled(enabled and len(self._calibration_review_previews) > 1)

    def _calibration_pair_selected(self, index: int) -> None:
        if self._updating_calibration_pair_select:
            return
        self._show_calibration_review_pair(index)

    def _show_previous_calibration_pair(self) -> None:
        self._show_calibration_review_pair(self.calibration_pair_select.currentIndex() - 1)

    def _show_next_calibration_pair(self) -> None:
        self._show_calibration_review_pair(self.calibration_pair_select.currentIndex() + 1)

    def _show_calibration_review_pair(self, index: int) -> None:
        if not self._calibration_review_previews:
            self._set_calibration_review_enabled(False)
            return
        index = max(0, min(index, len(self._calibration_review_previews) - 1))
        if self.calibration_pair_select.currentIndex() != index:
            self._updating_calibration_pair_select = True
            self.calibration_pair_select.setCurrentIndex(index)
            self._updating_calibration_pair_select = False
        self.prev_calibration_pair_button.setEnabled(index > 0)
        self.next_calibration_pair_button.setEnabled(index < len(self._calibration_review_previews) - 1)
        preview = self._calibration_review_previews[index]
        self._last_calibration_preview = preview
        self._display_calibration_preview(preview, len(self._calibration_review_previews), "review")

    def _display_calibration_preview(self, preview: CalibrationPairPreview, total_pairs: int, mode: str) -> None:
        stats = preview.stats
        line = (
            f"Pair {stats.index:04d}/{total_pairs} {stats.status} | "
            f"L {stats.left_markers} R {stats.right_markers} common {stats.common_points}"
        )
        self.calibration_metrics.setText(line)
        self.calibration_state.setText(f"Calibration: {mode} {stats.index:04d}/{total_pairs}")
        if stats.status != "VALID":
            self._set_diagnostic(f"{stats.filename}: {stats.reason}")
        else:
            self._set_diagnostic(line)

        left_image = self._draw_calibration_preview(preview.left_image, preview.left_detection, line, stats.reason)
        right_image = self._draw_calibration_preview(preview.right_image, preview.right_detection, line, stats.reason)
        if left_image is not None:
            self.left_view.set_frame(preview_copy(left_image))
            self.left_view.set_source_size(left_image.shape[1], left_image.shape[0])
        if right_image is not None:
            self.right_view.set_frame(preview_copy(right_image))
            self.right_view.set_source_size(right_image.shape[1], right_image.shape[0])
        if mode == "review":
            self.frozen_left = None if left_image is None else left_image.copy()
            self.frozen_right = None if right_image is None else right_image.copy()
            self._freeze_enabled = True
            self.freeze_button.setText("Unfreeze")
            self.preview_info.setText("Calibration Review")

    def _show_calibration_report(self, report: CalibrationDatasetReport) -> None:
        if report.total_pairs == 0:
            raise RuntimeError(f"No image pairs found in {self.config.capture.output_dir}")
        size_text = "--" if report.image_size is None else f"{report.image_size[0]}x{report.image_size[1]}"
        summary = (
            f"Calibration input: total={report.total_pairs}, readable={report.readable_pairs}, "
            f"valid={report.valid_pairs}, weak={report.weak_pairs}, invalid={report.invalid_pairs}, image={size_text}, "
            f"avg common={report.average_common_points:.1f}"
        )
        self._log(summary)
        if report.weak_filenames:
            self._log("Weak pairs skipped: " + ", ".join(report.weak_filenames[:30]))
        if report.invalid_filenames:
            self._log("Invalid pairs: " + ", ".join(report.invalid_filenames[:30]))

    def _show_calibration_result(self, result: StereoCalibrationResult, baseline: float, quality: str) -> None:
        if self._last_calibration_preview is None:
            return
        text = (
            f"RESULT {quality}\n"
            f"L err {result.left_error:.4f}  R err {result.right_error:.4f}\n"
            f"stereo {result.stereo_error:.4f}  baseline {baseline:.4f} m"
        )
        preview = self._last_calibration_preview
        left_image = self._draw_calibration_preview(preview.left_image, preview.left_detection, text, "")
        right_image = self._draw_calibration_preview(preview.right_image, preview.right_detection, text, "")
        if left_image is not None:
            self.left_view.set_frame(preview_copy(left_image))
            self.left_view.set_source_size(left_image.shape[1], left_image.shape[0])
        if right_image is not None:
            self.right_view.set_frame(preview_copy(right_image))
            self.right_view.set_source_size(right_image.shape[1], right_image.shape[0])
        self.frozen_left = None if left_image is None else left_image.copy()
        self.frozen_right = None if right_image is None else right_image.copy()
        self._freeze_enabled = True
        self.freeze_button.setText("Unfreeze")
        self.preview_info.setText("Calibration Result")
        QApplication.processEvents()

    def _draw_calibration_preview(
        self,
        image: np.ndarray | None,
        detection,
        title: str,
        detail: str,
    ) -> np.ndarray | None:
        if image is None:
            return None
        output = image.copy()
        if detection is not None:
            output = draw_detection(output, detection)
        lines = title.splitlines()
        if detail and detail != "OK":
            lines.extend(detail.splitlines())
        return self._draw_text_panel(output, lines)

    def _draw_text_panel(self, image: np.ndarray, lines: list[str]) -> np.ndarray:
        output = image.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(output.shape[1], output.shape[0]) / 900)
        thickness = max(1, int(round(scale * 2)))
        padding = 8
        line_gap = 8
        sizes = [cv2.getTextSize(line, font, scale, thickness)[0] for line in lines]
        panel_width = max((width for width, _ in sizes), default=1) + padding * 2
        panel_height = sum(height for _, height in sizes) + line_gap * max(len(lines) - 1, 0) + padding * 2
        panel_width = min(panel_width, output.shape[1])
        panel_height = min(panel_height, output.shape[0])
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (panel_width, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.62, output, 0.38, 0, output)
        y = padding
        for line, (_, height) in zip(lines, sizes):
            y += height
            cv2.putText(output, line, (padding, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
            y += line_gap
        return output

    def _draw_2d_measurement_overlay(self, image: np.ndarray) -> np.ndarray:
        if not self.measurement_2d_points:
            return image
        output = image.copy()
        color = (0, 255, 0)
        points = list(self.measurement_2d_points[:2])
        if len(points) == 2:
            cv2.line(output, points[0], points[1], color, 2, cv2.LINE_AA)
        for index, point in enumerate(points, start=1):
            cv2.circle(output, point, 5, color, -1, cv2.LINE_AA)
            cv2.putText(
                output,
                f"P{index}",
                (point[0] + 8, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return output

    def _process_barcode_frame(self, image: np.ndarray) -> np.ndarray:
        try:
            detections = decode_barcodes(image, self._enabled_barcode_labels())
        except Exception as exc:  # noqa: BLE001
            self.latest_barcode_detections = []
            self.barcode_info.setText(str(exc))
            self._set_diagnostic(str(exc))
            return image
        self.latest_barcode_detections = detections
        self.barcode_confirmation.required_count = max(1, self.barcode_confirm_frames.value())
        confirmed = self.barcode_confirmation.update(detections)
        if confirmed is None:
            self.barcode_info.setText(self.barcode_confirmation.status_text())
        else:
            key = (confirmed.text, confirmed.format)
            self.barcode_info.setText(f"{confirmed.text} ({confirmed.format})")
            self.distance_state.setText(f"Barcode: {confirmed.text}")
            if key != self._last_confirmed_barcode_key:
                self._log(f"Barcode confirmed: {confirmed.text} ({confirmed.format})")
                self._last_confirmed_barcode_key = key
        return draw_barcode_detections(image, detections)

    def _enabled_barcode_labels(self) -> list[str]:
        current = self.barcode_format.currentText()
        if current == "All":
            return list(SUPPORTED_BARCODE_LABELS)
        return [current]

    def _load_calibration(self) -> None:
        try:
            data = load_calibration_npz(self.config.output.dir)
            self.rectifier = Rectifier.from_calibration(data)
            self.calibration_state.setText("Calibration: loaded")
            self.baseline_state.setText("Baseline: loaded")
            self._log(f"Loaded calibration from {Path(self.config.output.dir) / 'stereo_calib.npz'}")
        except Exception as exc:  # noqa: BLE001
            self._error(str(exc))

    def _load_mono_calibration(self) -> None:
        try:
            data = load_mono_calibration_npz(self.config.output.dir)
            if "K" not in data or "D" not in data:
                raise ValueError("mono_calib.npz must contain K and D")
            self.rectifier = None
            self.latest_disparity = None
            self.mono_calibration = data
            self.calibration_state.setText("Calibration: mono loaded")
            self.baseline_state.setText("Baseline: mono N/A")
            self.output_path_label.setText(f"{Path(self.config.output.dir) / 'mono_calib.npz'}")
            self._log(f"Loaded mono calibration from {Path(self.config.output.dir) / 'mono_calib.npz'}")
        except Exception as exc:  # noqa: BLE001
            self._error(str(exc))

    def _set_rectifier_from_result(self, result: StereoCalibrationResult) -> None:
        self.rectifier = Rectifier.from_calibration({k: v for k, v in result.as_dict().items() if isinstance(v, np.ndarray) or k == "image_size"})

    def _handle_image_click(self, x: int, y: int) -> None:
        self._handle_preview_click("preview", x, y)

    def _handle_preview_click(self, side: str, x: int, y: int) -> None:
        if self.view_mode.currentText() == "2D Measurement":
            self._handle_2d_measurement_click(side, x, y)
            return
        if self.view_mode.currentText() != "Depth / Distance" or self.rectifier is None or self.latest_disparity is None:
            self._log(f"Clicked {side} image point x={x}, y={y}")
            self.distance_state.setText(f"Point: {side} {x}, {y}")
            return
        distance = distance_at(self.latest_disparity, self.rectifier.Q, x, y, window_size=5)
        if distance is None:
            self._log(f"x={x}, y={y}: no valid depth")
            self.distance_state.setText(f"Distance: invalid at {x}, {y}")
        else:
            self._log(f"x={x}, y={y}: distance={distance:.3f} m")
            self.distance_state.setText(f"Distance: {distance:.3f} m")

    def _handle_2d_measurement_click(self, side: str, x: int, y: int) -> None:
        if self.mono_calibration is None:
            self._log("Load mono calibration before 2D measurement")
            self.distance_state.setText("2D: load mono calibration")
            self.measurement_2d_info.setText("Load mono calibration first.")
            return
        if len(self.measurement_2d_points) >= 2:
            self.measurement_2d_points.clear()
        point = (int(x), int(y))
        self.measurement_2d_points.append(point)
        if len(self.measurement_2d_points) == 1:
            self.distance_state.setText(f"2D P1: {point[0]}, {point[1]}")
            self.measurement_2d_info.setText(f"P1 {point}; click second point.")
            self._log(f"2D point 1 on {side}: x={point[0]}, y={point[1]}")
            return
        p1, p2 = self.measurement_2d_points
        distance_mm = plane_distance_between_pixels(
            p1,
            p2,
            self.mono_calibration["K"],
            self.mono_calibration["D"],
            self.plane_distance_mm.value(),
        )
        self.distance_state.setText(f"2D: {distance_mm:.2f} mm")
        self.measurement_2d_info.setText(
            f"P1 {p1}, P2 {p2}, Z={self.plane_distance_mm.value():.2f} mm, distance={distance_mm:.2f} mm"
        )
        self._log(f"P1={p1}, P2={p2}, Z={self.plane_distance_mm.value():.2f} mm: 2D distance={distance_mm:.2f} mm")

    def _clear_2d_measurement(self) -> None:
        self.measurement_2d_points.clear()
        self.distance_state.setText("2D: --")
        self.measurement_2d_info.setText("Click two points on the same plane.")
        self._log("Cleared 2D measurement points")

    def _clear_barcode_result(self) -> None:
        self.barcode_confirmation.reset()
        self.latest_barcode_detections = []
        self._last_confirmed_barcode_key = None
        self.distance_state.setText("Barcode: --")
        self.barcode_info.setText("No barcode")
        self._log("Cleared barcode result")

    def _save_current_config(self) -> None:
        self.config.camera = self._current_camera_config()
        self.config.aruco = self._current_aruco_config()
        save_config(self.config)
        self._log("Config saved")

    def _refresh_pair_count(self) -> None:
        count = len(list_image_pairs(self.config.capture.output_dir))
        text = pair_count_text(count)
        self.pair_count.setText(text)
        self.pair_state.setText(text)

    def _update_camera_stats(self) -> None:
        summary_parts = []
        for side, worker, view in (
            ("L", self.left_worker, self.left_view),
            ("R", self.right_worker, self.right_view),
        ):
            if worker is None:
                view.set_overlay_text("")
                continue
            stats = worker.snapshot_stats()
            age = stats["age"]
            age_text = "--" if age < 0 else f"{age:.2f}s"
            summary_parts.append(f"{side} {stats['fps']:.1f} fps")
            view.set_overlay_text(
                f"idx {stats['index']}\n"
                f"{stats['fps']:.1f} fps\n"
                f"frames {stats['frames']}\n"
                f"age {age_text}"
            )
        summary_text = " | ".join(summary_parts) if summary_parts else "L -- fps | R -- fps"
        self.camera_stats.setText(summary_text)
        self.fps_state.setText(f"Camera: {summary_text}")
        if self.left_worker is not None:
            self.left_state.setText(f"Left: online index {self.left_worker.index}")
        if self.right_worker is not None:
            self.right_state.setText(f"Right: online index {self.right_worker.index}")

    def _log(self, message: str) -> None:
        self.status_box.append(message)
        lowered = message.lower()
        if any(token in lowered for token in ("failed", "black frame", "no readable", "waiting", "error")):
            self._set_diagnostic(message)

    def _error(self, message: str) -> None:
        self._log(f"Error: {message}")
        QMessageBox.warning(self, "Error", message)

    def _view_mode_changed(self, mode: str) -> None:
        if not view_mode_enabled(mode, self.rectifier is not None):
            self._set_diagnostic(f"{mode} requires calibration.")
            self._log(f"{mode} requires calibration.")
            self.view_mode.setCurrentText("Live Preview")
            return
        depth_mode = mode == "Depth / Distance"
        measurement_mode = mode == "2D Measurement"
        barcode_mode = mode == "Barcode Detection"
        self.calibration_group.setVisible(not depth_mode and not measurement_mode and not barcode_mode)
        self.depth_group.setVisible(depth_mode)
        self.measurement_2d_group.setVisible(measurement_mode)
        self.barcode_group.setVisible(barcode_mode)
        self.preview_info.setText(mode)

    def _refresh_diagnostics(self) -> None:
        try:
            warning = camera_selection_warning(
                self._selected_camera_index(self.left_index),
                self._selected_camera_index(self.right_index),
                self.single_mode.isChecked(),
            )
        except Exception:  # noqa: BLE001
            warning = "Camera index must be an integer."
        self._set_diagnostic(warning or "Ready")

    def _set_diagnostic(self, message: str) -> None:
        self.diagnostics_label.setText(message)

    def _toggle_freeze(self) -> None:
        if not self._freeze_enabled:
            if self.latest_left is None:
                self._log("No camera frames available to freeze")
                return
            self.frozen_left = self.latest_left.copy()
            self.frozen_right = None if self.latest_right is None else self.latest_right.copy()
            self._freeze_enabled = True
            self.freeze_button.setText("Unfreeze")
            self.preview_info.setText("Frozen Preview")
            return
        self._freeze_enabled = False
        self.frozen_left = None
        self.frozen_right = None
        self.freeze_button.setText("Freeze Frame")
        self.preview_info.setText(self.view_mode.currentText())

    def _frame_for_original_view(self, side: str) -> np.ndarray | None:
        if side == "left":
            return self.frozen_left if self._freeze_enabled and self.frozen_left is not None else self.latest_left
        return self.frozen_right if self._freeze_enabled and self.frozen_right is not None else self.latest_right

    def _open_original_view(self, side: str) -> None:
        frame = self._frame_for_original_view(side)
        if frame is None:
            self._log(f"No {side} frame available")
            return
        dialog = OriginalImageDialog(f"{side.title()} Original", frame, self)
        dialog.point_selected.connect(lambda x, y, side_name=side: self._handle_preview_click(side_name, x, y))
        dialog.show()
        self._original_dialogs.append(dialog)
