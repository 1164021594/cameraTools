from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
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
from stereo_aruco_gui.app.calibration import StereoCalibrationResult, calibrate_from_pairs
from stereo_aruco_gui.app.camera_probe import scan_camera_indices
from stereo_aruco_gui.app.camera_worker import BACKEND_MAP, PIXEL_FORMATS, SingleCameraWorker, preview_copy
from stereo_aruco_gui.app.config import AppConfig, ArucoConfig, CameraConfig, save_config
from stereo_aruco_gui.app.image_view import ImageView
from stereo_aruco_gui.app.rectification import Rectifier, compute_disparity, disparity_preview, distance_at, draw_horizontal_guides
from stereo_aruco_gui.app.storage import delete_last_image_pair, list_image_pairs, load_calibration_npz, save_image_pair
from stereo_aruco_gui.app.ui_state import camera_selection_warning, pair_count_text, view_mode_enabled


RESOLUTION_OPTIONS = (
    (320, 240),
    (640, 480),
    (800, 600),
    (1024, 768),
    (1280, 720),
    (1280, 960),
    (1920, 1080),
    (2592, 1944),
)

VIEW_MODES = ("Live Preview", "ArUco Detection", "Rectified Preview", "Depth / Distance")


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
        self.rectifier: Rectifier | None = None
        self.latest_disparity: np.ndarray | None = None
        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(100)
        self.preview_timer.timeout.connect(self._refresh_preview)

        self.setWindowTitle("Stereo ArUco Calibration")
        self.resize(1400, 820)
        self._build_ui()
        self._refresh_pair_count()

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
        controls.addWidget(self._calibration_group())
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
        preview_header.addStretch(1)
        self.preview_info = QLabel("Live Preview")
        preview_header.addWidget(self.preview_info)
        preview_panel_layout.addLayout(preview_header)

        preview_layout = QGridLayout()
        self.left_view = ImageView("Left Camera")
        self.right_view = ImageView("Right Camera")
        self.left_view.image_clicked.connect(self._handle_image_click)
        self.right_view.image_clicked.connect(self._handle_image_click)
        preview_layout.addWidget(QLabel("Left Camera"), 0, 0)
        preview_layout.addWidget(QLabel("Right Camera"), 0, 1)
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
        self.backend.addItems(["AUTO", *BACKEND_MAP.keys()])
        self.backend.setCurrentText(cam.backend)
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
        form.addRow("Dictionary", self.dictionary)
        form.addRow("Markers X", self.markers_x)
        form.addRow("Markers Y", self.markers_y)
        form.addRow("Marker length m", self.marker_length)
        form.addRow("Separation m", self.marker_separation)
        self.calibrate_button = QPushButton("Start Calibration")
        self.calibrate_button.clicked.connect(self._calibrate)
        self.load_button = QPushButton("Load Calibration")
        self.load_button.clicked.connect(self._load_calibration)
        self.calibration_metrics = QLabel("Errors: --")
        self.output_path_label = QLabel(f"Output: {self.config.output.dir}")
        form.addRow(self.calibrate_button, self.load_button)
        form.addRow("Result", self.calibration_metrics)
        form.addRow("Output", self.output_path_label)
        return group

    def _ranging_group(self) -> QGroupBox:
        group = QGroupBox("4. Result And Ranging")
        layout = QVBoxLayout(group)
        self.detect_button = QPushButton("Detect Current")
        self.detect_button.clicked.connect(self._detect_current)
        self.guides_check = QCheckBox("Horizontal guides")
        self.guides_check.setChecked(True)
        self.ranging_hint = QLabel("Depth and rectified views require calibration.")
        layout.addWidget(self.detect_button)
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
        if current not in options:
            options.insert(0, current)
        for option_width, option_height in options:
            combo.addItem(resolution_label(option_width, option_height))
        combo.setCurrentText(resolution_label(width, height))
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
        )

    def _toggle_cameras(self) -> None:
        if self.left_worker and self.left_worker.isRunning():
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
        self.left_worker.status.connect(self._log)
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
            self.right_worker.status.connect(self._log)
            self.right_worker.start()
        self.preview_timer.start()
        self.open_button.setText("Close Cameras")
        self.left_state.setText(f"Left: opening index {camera.left_index}")
        self.right_state.setText("Right: single mode" if camera.single_mode else f"Right: opening index {camera.right_index}")
        self._log("Opening cameras. Close Windows Camera or any other program using the cameras if no frames appear.")

    def _stop_cameras(self, reason: str = "unspecified") -> None:
        self._log(f"Stopping cameras: {reason}")
        if self.left_worker:
            self.left_worker.stop()
            self.left_worker = None
        if self.right_worker:
            self.right_worker.stop()
            self.right_worker = None
        self.preview_timer.stop()
        self.open_button.setText("Open Cameras")
        self.left_state.setText("Left: closed")
        self.right_state.setText("Right: closed")
        self.fps_state.setText("FPS: --")

    def _refresh_preview(self) -> None:
        if self.left_worker is None:
            return
        left = self.left_worker.get_latest()
        if left is None:
            return
        right = None if self.right_worker is None else self.right_worker.get_latest()
        if self.right_worker is not None and right is None:
            self.left_view.set_frame(preview_copy(left))
            self.right_view.setText("Waiting for right camera")
            return
        self._update_frames(left, right)
        self._update_camera_stats()

    def _update_frames(self, left: np.ndarray, right: np.ndarray | None) -> None:
        self.latest_left = left
        self.latest_right = right
        show_left, show_right = left, right
        if right is None:
            self.left_view.set_frame(preview_copy(show_left))
            self.right_view.setText("Single camera mode")
            return
        mode = self.view_mode.currentText()
        if not view_mode_enabled(mode, self.rectifier is not None):
            self.preview_info.setText("Calibration required")
            self.latest_disparity = None
            self.left_view.set_frame(preview_copy(left))
            self.right_view.set_frame(preview_copy(right))
            return
        self.preview_info.setText(mode)
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
            self.latest_disparity = compute_disparity(left_rect, right_rect)
            show_left = disparity_preview(self.latest_disparity)
            show_right = draw_horizontal_guides(right_rect) if self.guides_check.isChecked() else right_rect
        else:
            self.latest_disparity = None
        self.left_view.set_frame(preview_copy(show_left))
        self.right_view.set_frame(preview_copy(show_right))

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
            self._log("Single camera mode is for preview testing only; pair capture is disabled")
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
            self._log("Disable single camera mode before stereo calibration")
            return
        try:
            self._log("Calibration started")
            result = calibrate_from_pairs(
                self.config.capture.output_dir,
                self.config.output.dir,
                self._current_aruco_config(),
            )
            self._set_rectifier_from_result(result)
            baseline = float(np.linalg.norm(result.T))
            self.calibration_metrics.setText(
                f"L {result.left_error:.4f} / R {result.right_error:.4f} / stereo {result.stereo_error:.4f}"
            )
            self.calibration_state.setText(f"Calibration: stereo error {result.stereo_error:.4f}")
            self.baseline_state.setText(f"Baseline: {baseline:.4f} m")
            self.output_path_label.setText(f"{Path(self.config.output.dir) / 'stereo_calib.npz'}")
            self._log(
                "Calibration done\n"
                f"left error: {result.left_error:.4f}\n"
                f"right error: {result.right_error:.4f}\n"
                f"stereo error: {result.stereo_error:.4f}\n"
                f"baseline: {baseline:.4f} m"
            )
        except Exception as exc:  # noqa: BLE001
            self._error(str(exc))

    def _load_calibration(self) -> None:
        try:
            data = load_calibration_npz(self.config.output.dir)
            self.rectifier = Rectifier.from_calibration(data)
            self.calibration_state.setText("Calibration: loaded")
            self.baseline_state.setText("Baseline: loaded")
            self._log(f"Loaded calibration from {Path(self.config.output.dir) / 'stereo_calib.npz'}")
        except Exception as exc:  # noqa: BLE001
            self._error(str(exc))

    def _set_rectifier_from_result(self, result: StereoCalibrationResult) -> None:
        self.rectifier = Rectifier.from_calibration({k: v for k, v in result.as_dict().items() if isinstance(v, np.ndarray) or k == "image_size"})

    def _handle_image_click(self, x: int, y: int) -> None:
        if self.view_mode.currentText() != "Depth / Distance" or self.rectifier is None or self.latest_disparity is None:
            self._log(f"Clicked image point x={x}, y={y}")
            self.distance_state.setText(f"Point: {x}, {y}")
            return
        distance = distance_at(self.latest_disparity, self.rectifier.Q, x, y)
        if distance is None:
            self._log(f"x={x}, y={y}: no valid depth")
            self.distance_state.setText(f"Distance: invalid at {x}, {y}")
        else:
            self._log(f"x={x}, y={y}: distance={distance:.3f} m")
            self.distance_state.setText(f"Distance: {distance:.3f} m")

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
        parts = []
        for worker in (self.left_worker, self.right_worker):
            if worker is None:
                continue
            stats = worker.snapshot_stats()
            age = stats["age"]
            age_text = "--" if age < 0 else f"{age:.2f}s"
            parts.append(
                f"{stats['label']} idx {stats['index']}: "
                f"{stats['fps']:.1f} fps, frames {stats['frames']}, age {age_text}"
            )
        stats_text = " | ".join(parts) if parts else "left fps: -- | right fps: --"
        self.camera_stats.setText(stats_text)
        self.fps_state.setText(f"FPS: {stats_text}")
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
