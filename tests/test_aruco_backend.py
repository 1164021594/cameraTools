from __future__ import annotations

import os

import numpy as np
from PySide6.QtWidgets import QApplication

import stereo_aruco_gui.app.barcode as barcode_module
import stereo_aruco_gui.app.camera_worker as camera_worker_module
import stereo_aruco_gui.app.main_window as main_window_module
import stereo_aruco_gui.app.measurement_2d as measurement_2d_module
import stereo_aruco_gui.app.rectification as rectification_module
from stereo_aruco_gui.app.aruco_board import board_object_map, create_board, create_detector, dictionary_id
from stereo_aruco_gui.app.barcode import (
    BarcodeConfirmation,
    BarcodeDetection,
    barcode_formats_for_labels,
    decode_barcodes,
    draw_barcode_detections,
)
from stereo_aruco_gui.app.calibration import (
    CalibrationPairPreview,
    MonoCalibrationResult,
    PairDetectionStats,
    StereoCalibrationResult,
    analyze_calibration_pairs,
    calibrate_mono_from_images,
    calibration_quality_label,
    pair_quality_status,
)
from stereo_aruco_gui.app.camera_probe import DEFAULT_SCAN_BACKENDS, scan_camera_indices
from stereo_aruco_gui.app.camera_worker import (
    CameraWorker,
    OpenAttempt,
    SingleCameraWorker,
    accept_frame_for_preview,
    configure_capture,
    open_failure_summary,
    preview_copy,
    read_stereo_pair,
    should_decode_frame,
    usable_frame,
)
from stereo_aruco_gui.app.config import CameraConfig
from stereo_aruco_gui.app.config import AppConfig
from stereo_aruco_gui.app.config import ArucoConfig, load_config
from stereo_aruco_gui.app.image_view import ImageView
from stereo_aruco_gui.app.main_window import MainWindow, parse_resolution_label
from stereo_aruco_gui.app.measurement_2d import plane_distance_between_pixels
from stereo_aruco_gui.app.rectification import DisparityConfig, compute_disparity, distance_at, filtered_disparity_preview
from stereo_aruco_gui.app.storage import (
    delete_last_image_pair,
    list_image_pairs,
    list_single_images,
    save_image_pair,
    save_single_image,
)
from stereo_aruco_gui.app.ui_state import camera_selection_warning, pair_count_text, view_mode_enabled

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


class FakeStatsWorker:
    def __init__(self, label: str, index: int, fps: float, frames: int, age: float) -> None:
        self.label = label
        self.index = index
        self._stats = {"label": label, "index": index, "fps": fps, "frames": frames, "age": age}

    def snapshot_stats(self) -> dict[str, float | int | str]:
        return self._stats


class FakeRunningWorker:
    def __init__(self) -> None:
        self.stop_called = False

    def stop(self) -> bool:
        self.stop_called = True
        return False

    def isRunning(self) -> bool:  # noqa: N802
        return True


class FakeStoppedWorker:
    def __init__(self) -> None:
        self.stop_called = False

    def stop(self) -> bool:
        self.stop_called = True
        return True

    def isRunning(self) -> bool:  # noqa: N802
        return False


class FakeFrameWorker:
    def __init__(self, frame: np.ndarray | None) -> None:
        self.frame = frame
        self.index = 1

    def get_latest(self) -> np.ndarray | None:
        return self.frame

    def snapshot_stats(self) -> dict[str, float | int | str]:
        return {"label": "fake", "index": self.index, "fps": 1.0, "frames": 1 if self.frame is not None else 0, "age": 0.0}


class FakeRectifier:
    Q = np.eye(4)

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return left, right


def make_calibration_preview(
    index: int,
    *,
    valid: bool = True,
    status: str | None = None,
    common_points: int | None = None,
    left_markers: int = 12,
    right_markers: int = 12,
) -> CalibrationPairPreview:
    value = 80 if valid else 20
    frame = np.full((240, 320, 3), value, dtype=np.uint8)
    common_points = common_points if common_points is not None else (48 if valid else 0)
    status = status or ("VALID" if valid else "INVALID")
    stats = PairDetectionStats(
        index=index,
        filename=f"{index:04d}.png",
        readable=True,
        left_markers=left_markers,
        right_markers=right_markers,
        common_points=common_points,
        valid=valid,
        status=status,
        reason="OK"
        if valid
        else (
            f"Weak pair: need 40 common corners for calibration, found {common_points}"
            if status == "WEAK"
            else f"Need 16 common corners, found {common_points}"
        ),
    )
    return CalibrationPairPreview(
        stats=stats,
        left_image=frame,
        right_image=frame,
        left_detection=None,
        right_detection=None,
    )


def test_load_config_defaults_when_missing(tmp_path):
    config = load_config(tmp_path / "missing.yaml")

    assert config.camera.left_index == 0
    assert config.camera.right_index == 1
    assert config.camera.single_mode is False
    assert config.camera.backend == "AUTO"
    assert config.camera.pixel_format == "MJPG"
    assert config.aruco.dictionary == "DICT_4X4_50"


def test_parse_resolution_label_returns_width_and_height():
    assert parse_resolution_label("1280 x 960") == (1280, 960)


def test_resolution_dropdown_uses_4_3_modes_for_this_camera_module():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())

    options = [window.resolution.itemText(index) for index in range(window.resolution.count())]

    assert app is not None
    assert "1280 x 960" in options
    assert "1280 x 720" not in options


def test_resolution_dropdown_replaces_legacy_16_9_config_with_4_3_default():
    app = QApplication.instance() or QApplication([])
    config = AppConfig(camera=CameraConfig(width=1280, height=720))
    window = MainWindow(config)

    options = [window.resolution.itemText(index) for index in range(window.resolution.count())]

    assert app is not None
    assert window.resolution.currentText() == "1280 x 960"
    assert "1280 x 720" not in options


def test_backend_dropdown_keeps_all_manual_backend_choices():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())

    options = [window.backend.itemText(index) for index in range(window.backend.count())]

    assert app is not None
    assert options == ["AUTO", "DSHOW", "MSMF", "DEFAULT"]


def test_backend_dropdown_preserves_legacy_msmf_config_for_manual_testing():
    app = QApplication.instance() or QApplication([])
    config = AppConfig(camera=CameraConfig(backend="MSMF"))
    window = MainWindow(config)

    assert app is not None
    assert window.backend.currentText() == "MSMF"


def test_preview_title_row_is_compact():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())

    preview_panel_layout = window.left_view.parentWidget().layout()
    preview_grid = preview_panel_layout.itemAt(1).layout()
    top_margin = preview_grid.contentsMargins().top()
    bottom_margin = preview_grid.contentsMargins().bottom()
    left_title = preview_grid.itemAtPosition(0, 0).widget()
    right_title = preview_grid.itemAtPosition(0, 1).widget()

    assert app is not None
    assert preview_grid.verticalSpacing() <= 2
    assert top_margin <= 2
    assert bottom_margin <= 2
    assert preview_grid.rowStretch(0) == 0
    assert preview_grid.rowStretch(1) == 1
    assert left_title.maximumHeight() <= 22
    assert right_title.maximumHeight() <= 22



def test_barcode_formats_for_labels_maps_supported_types():
    formats = barcode_formats_for_labels(["Code 39", "Code 128", "Codabar", "EAN", "ITF25", "Code 93", "QR Code", "DataMatrix"])

    assert formats is not None


def test_decode_barcodes_reads_generated_qrcode():
    zxingcpp = barcode_module.require_zxingcpp()
    barcode = zxingcpp.create_barcode("CAMERA-OK", zxingcpp.BarcodeFormat.QRCode)
    image = np.array(zxingcpp.write_barcode_to_image(barcode, scale=4))

    detections = decode_barcodes(image, enabled_labels=["QR Code"])

    assert len(detections) == 1
    assert detections[0].text == "CAMERA-OK"
    assert detections[0].format == "QR Code"
    assert detections[0].points is not None


def test_barcode_confirmation_requires_repeated_same_result():
    confirmation = BarcodeConfirmation(required_count=3)
    detection = BarcodeDetection(text="ABC123", format="Code 128", points=None)

    assert confirmation.update([detection]) is None
    assert confirmation.update([detection]) is None

    confirmed = confirmation.update([detection])

    assert confirmed == detection
    assert confirmation.status_text() == "ABC123 (Code 128) x3"


def test_barcode_confirmation_resets_on_different_result():
    confirmation = BarcodeConfirmation(required_count=2)
    first = BarcodeDetection(text="ABC123", format="Code 128", points=None)
    second = BarcodeDetection(text="XYZ999", format="Code 128", points=None)

    assert confirmation.update([first]) is None
    assert confirmation.update([second]) is None

    confirmed = confirmation.update([second])

    assert confirmed == second
    assert confirmation.status_text() == "XYZ999 (Code 128) x2"


def test_barcode_detection_view_shows_controls_without_requiring_calibration():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.show()
    app.processEvents()

    window.view_mode.setCurrentText("Barcode Detection")

    assert app is not None
    assert window.calibration_group.isVisible() is False
    assert window.depth_group.isVisible() is False
    assert window.measurement_2d_group.isVisible() is False
    assert window.barcode_group.isVisible() is True


def test_barcode_detection_updates_preview_and_confirmed_result(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.view_mode.setCurrentText("Barcode Detection")
    window.barcode_confirm_frames.setValue(1)
    detection = BarcodeDetection(text="ABC123", format="Code 128", points=((2, 2), (8, 2), (8, 8), (2, 8)))

    monkeypatch.setattr(main_window_module, "decode_barcodes", lambda frame, enabled_labels: [detection])

    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    window._update_frames(frame, None)

    assert app is not None
    assert window.distance_state.text() == "Barcode: ABC123"
    assert "Barcode confirmed: ABC123 (Code 128)" in window.status_box.toPlainText()
    assert window.left_view._last_frame is not None
    assert np.array_equal(window.left_view._last_frame[2, 2], np.array([0, 255, 0], dtype=np.uint8))


def test_view_mode_enabled_allows_barcode_detection_without_calibration():
    assert view_mode_enabled("Barcode Detection", has_calibration=False) is True

def test_depth_view_replaces_calibration_controls_with_depth_parameters():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.rectifier = FakeRectifier()
    window.show()
    app.processEvents()

    assert app is not None
    assert window.calibration_group.isVisible() is True
    assert window.depth_group.isVisible() is False

    window.view_mode.setCurrentText("Depth / Distance")

    assert window.calibration_group.isVisible() is False
    assert window.depth_group.isVisible() is True

    window.view_mode.setCurrentText("Live Preview")

    assert window.calibration_group.isVisible() is True
    assert window.depth_group.isVisible() is False


def test_2d_measurement_view_shows_controls_without_requiring_stereo_calibration():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.show()
    app.processEvents()

    window.view_mode.setCurrentText("2D Measurement")

    assert app is not None
    assert window.calibration_group.isVisible() is False
    assert window.depth_group.isVisible() is False
    assert window.measurement_2d_group.isVisible() is True


def test_2d_measurement_clicks_two_points_and_reports_distance(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    config = load_config()
    config.output.dir = str(tmp_path / "output")
    window = MainWindow(config)
    window.view_mode.setCurrentText("2D Measurement")
    window.plane_distance_mm.setValue(500.0)
    window.mono_calibration = {"K": np.eye(3), "D": np.zeros((1, 5))}

    def fake_distance(p1, p2, K, D, plane_distance_mm):  # noqa: ANN001
        assert p1 == (10, 20)
        assert p2 == (110, 20)
        assert plane_distance_mm == 500.0
        return 42.5

    monkeypatch.setattr(main_window_module, "plane_distance_between_pixels", fake_distance)

    window._handle_preview_click("left", 10, 20)
    window._handle_preview_click("left", 110, 20)

    assert app is not None
    assert window.distance_state.text() == "2D: 42.50 mm"
    assert "2D distance=42.50 mm" in window.status_box.toPlainText()


def test_load_mono_calibration_loads_mono_file_without_stereo_rectifier(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    config = load_config()
    config.output.dir = str(tmp_path / "output")
    window = MainWindow(config)

    monkeypatch.setattr(
        main_window_module,
        "load_mono_calibration_npz",
        lambda output_dir: {"calibration_type": np.array("mono"), "K": np.eye(3), "D": np.zeros((1, 5))},
    )

    window._load_mono_calibration()

    assert app is not None
    assert window.rectifier is None
    assert window.mono_calibration is not None
    assert window.calibration_state.text() == "Calibration: mono loaded"
    assert "Loaded mono calibration" in window.status_box.toPlainText()


def test_depth_parameters_are_read_dynamically_for_each_disparity_frame(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.rectifier = FakeRectifier()
    window.view_mode.setCurrentText("Depth / Distance")
    captured: list[DisparityConfig] = []

    def fake_compute_disparity(left_rect, right_rect, config):  # noqa: ANN001
        captured.append(config)
        return np.zeros(left_rect.shape[:2], dtype=np.float32)

    monkeypatch.setattr(main_window_module, "compute_disparity", fake_compute_disparity)
    left = np.zeros((24, 32, 3), dtype=np.uint8)
    right = np.zeros((24, 32, 3), dtype=np.uint8)

    window.depth_num_disparities.setCurrentText("256")
    window.depth_block_size.setCurrentText("9")
    window.depth_uniqueness_ratio.setCurrentText("5")
    window.depth_speckle_window_size.setCurrentText("0")
    window.depth_speckle_range.setCurrentText("4")
    window.depth_disp12_max_diff.setCurrentText("-1")
    window._update_frames(left, right)

    window.depth_num_disparities.setCurrentText("128")
    window._update_frames(left, right)

    assert app is not None
    assert captured[0] == DisparityConfig(
        num_disparities=256,
        block_size=9,
        uniqueness_ratio=5,
        speckle_window_size=0,
        speckle_range=4,
        disp12_max_diff=-1,
    )
    assert captured[1].num_disparities == 128


def test_depth_display_mode_filtered_uses_filtered_preview_without_replacing_raw_disparity(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.rectifier = FakeRectifier()
    window.view_mode.setCurrentText("Depth / Distance")
    raw_disparity = np.zeros((48, 160), dtype=np.float32)
    raw_disparity[:, 128:] = 8.0
    calls = {"filtered": 0, "raw": 0}

    def fake_compute_disparity(left_rect, right_rect, config):  # noqa: ANN001
        return raw_disparity

    def fake_filtered_preview(disparity):
        calls["filtered"] += 1
        return np.full((*disparity.shape, 3), 90, dtype=np.uint8)

    def fake_raw_preview(disparity):
        calls["raw"] += 1
        return np.full((*disparity.shape, 3), 30, dtype=np.uint8)

    monkeypatch.setattr(main_window_module, "compute_disparity", fake_compute_disparity)
    monkeypatch.setattr(main_window_module, "filtered_disparity_preview", fake_filtered_preview)
    monkeypatch.setattr(main_window_module, "disparity_preview", fake_raw_preview)

    window.depth_display_mode.setCurrentText("Filtered")
    window.depth_num_disparities.setCurrentText("128")
    window._update_frames(np.zeros((48, 160, 3), dtype=np.uint8), np.zeros((48, 160, 3), dtype=np.uint8))

    assert app is not None
    assert calls == {"filtered": 1, "raw": 0}
    assert window.latest_disparity is raw_disparity
    assert window.left_view._last_frame is not None
    assert int(window.left_view._last_frame[0, 0, 0]) == 90


def test_depth_view_crops_left_preview_and_marks_right_roi(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.rectifier = FakeRectifier()
    window.view_mode.setCurrentText("Depth / Distance")

    def fake_compute_disparity(left_rect, right_rect, config):  # noqa: ANN001
        disparity = np.zeros(left_rect.shape[:2], dtype=np.float32)
        disparity[:, config.num_disparities :] = 12.0
        return disparity

    monkeypatch.setattr(main_window_module, "compute_disparity", fake_compute_disparity)
    left = np.zeros((48, 160, 3), dtype=np.uint8)
    right = np.full((48, 160, 3), 20, dtype=np.uint8)
    window.depth_num_disparities.setCurrentText("128")

    window._update_frames(left, right)

    assert app is not None
    assert window.left_view._source_rect == (128, 0, 32, 48)
    assert window.right_view._source_rect == (0, 0, 160, 48)
    assert window.left_view._frame_shape == (32, 48)
    assert window.right_view._frame_shape == (160, 48)
    right_preview = window.right_view._last_frame
    assert right_preview is not None
    assert np.array_equal(right_preview[1, 128], np.array([0, 255, 0], dtype=np.uint8))
    assert np.array_equal(right_preview[46, 158], np.array([0, 255, 0], dtype=np.uint8))
    assert np.array_equal(right_preview[24, 64], np.array([20, 20, 20], dtype=np.uint8))


def test_depth_preview_click_uses_raw_local_median_distance(monkeypatch):
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.rectifier = FakeRectifier()
    window.view_mode.setCurrentText("Depth / Distance")
    window.latest_disparity = np.zeros((9, 9), dtype=np.float32)
    window.latest_disparity[2:7, 2:7] = 10.0
    window.latest_disparity[4, 4] = 0.0

    def fake_distance_at(disparity, q, x, y, window_size=1):  # noqa: ANN001
        assert disparity is window.latest_disparity
        assert window_size == 5
        return 1.25

    monkeypatch.setattr(main_window_module, "distance_at", fake_distance_at)

    window._handle_preview_click("preview", 4, 4)

    assert app is not None
    assert window.distance_state.text() == "Distance: 1.250 m"
    assert "distance=1.250 m" in window.status_box.toPlainText()


def test_depth_preset_updates_disparity_controls():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())

    window.depth_preset.setCurrentText("Near")

    assert app is not None
    assert window._current_disparity_config() == DisparityConfig(
        num_disparities=256,
        block_size=5,
        uniqueness_ratio=5,
        speckle_window_size=0,
        speckle_range=2,
        disp12_max_diff=-1,
    )


def test_compute_disparity_uses_supplied_sgbm_parameters(monkeypatch):
    captured = {}

    class FakeMatcher:
        def compute(self, left_gray, right_gray):  # noqa: ANN001
            return np.zeros(left_gray.shape, dtype=np.int16)

    def fake_create(**kwargs):  # noqa: ANN003
        captured.update(kwargs)
        return FakeMatcher()

    monkeypatch.setattr(rectification_module.cv2, "StereoSGBM_create", fake_create)

    disparity = compute_disparity(
        np.zeros((12, 16), dtype=np.uint8),
        np.zeros((12, 16), dtype=np.uint8),
        DisparityConfig(
            num_disparities=256,
            block_size=9,
            uniqueness_ratio=5,
            speckle_window_size=0,
            speckle_range=4,
            disp12_max_diff=-1,
        ),
    )

    assert disparity.shape == (12, 16)
    assert captured == {
        "minDisparity": 0,
        "numDisparities": 256,
        "blockSize": 9,
        "P1": 8 * 9 * 9,
        "P2": 32 * 9 * 9,
        "uniquenessRatio": 5,
        "speckleWindowSize": 0,
        "speckleRange": 4,
        "disp12MaxDiff": -1,
    }


def test_filtered_disparity_preview_fills_small_holes_for_display_only():
    disparity = np.tile(np.linspace(4.0, 12.0, 9, dtype=np.float32), (9, 1))
    disparity[4, 4] = 0.0

    raw_preview = rectification_module.disparity_preview(disparity)
    preview = filtered_disparity_preview(disparity)

    assert preview.shape == (9, 9, 3)
    assert np.any(preview[4, 4] != raw_preview[4, 4])


def test_distance_at_uses_local_median_when_center_pixel_is_invalid():
    disparity = np.zeros((9, 9), dtype=np.float32)
    disparity[2:7, 2:7] = 10.0
    disparity[4, 4] = 0.0
    q = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 20.0],
            [0.0, 0.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    assert distance_at(disparity, q, 4, 4, window_size=5) == 1.0


def test_plane_distance_between_pixels_uses_mono_intrinsics_and_plane_z():
    K = np.array(
        [
            [500.0, 0.0, 320.0],
            [0.0, 500.0, 240.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    D = np.zeros((1, 5), dtype=np.float32)

    distance = plane_distance_between_pixels((320, 240), (420, 240), K, D, plane_distance_mm=500.0)

    assert distance == 100.0


def test_calibration_review_controls_start_disabled():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())

    assert app is not None
    assert window.calibration_pair_select.count() == 0
    assert window.calibration_pair_select.isEnabled() is False
    assert window.prev_calibration_pair_button.isEnabled() is False
    assert window.next_calibration_pair_button.isEnabled() is False


def test_image_view_supports_lower_right_overlay_text():
    app = QApplication.instance() or QApplication([])
    view = ImageView("Preview")
    view.resize(640, 480)
    view.show()

    view.set_overlay_text("idx 0\n29.8 fps\nframes 120\nage 0.03s")
    app.processEvents()

    assert app is not None
    assert view.overlay_text() == "idx 0\n29.8 fps\nframes 120\nage 0.03s"
    overlay = view.overlay_label
    assert overlay.parentWidget() is view
    assert overlay.isVisible()
    assert overlay.x() + overlay.width() <= view.width()
    assert overlay.y() + overlay.height() <= view.height()
    assert overlay.x() > view.width() // 2
    assert overlay.y() > view.height() // 2


def test_image_view_maps_clicks_using_source_size_override():
    app = QApplication.instance() or QApplication([])
    view = ImageView("Preview")
    view.resize(640, 480)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    view.set_frame(frame)
    view.set_source_size(800, 600)

    assert app is not None
    assert view.map_label_point_to_image(view.rect().center()) == (399, 299)


def test_image_view_maps_clicks_with_source_offset_for_cropped_depth_view():
    app = QApplication.instance() or QApplication([])
    view = ImageView("Preview")
    view.resize(640, 480)
    frame = np.zeros((480, 448, 3), dtype=np.uint8)
    view.set_frame(frame)
    view.set_source_rect(192, 0, 448, 480)

    assert app is not None
    assert view.map_label_point_to_image(view.rect().center()) == (415, 239)


def test_camera_stats_use_preview_overlays_and_short_bottom_summary():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.left_worker = FakeStatsWorker("left", 0, 29.84, 120, 0.03)
    window.right_worker = FakeStatsWorker("right", 1, 28.16, 118, 0.05)

    window._update_camera_stats()

    assert app is not None
    assert window.left_view.overlay_text() == "idx 0\n29.8 fps\nframes 120\nage 0.03s"
    assert window.right_view.overlay_text() == "idx 1\n28.2 fps\nframes 118\nage 0.05s"
    assert window.camera_stats.text() == "L 29.8 fps | R 28.2 fps"
    assert window.fps_state.text() == "Camera: L 29.8 fps | R 28.2 fps"
    assert "frames" not in window.fps_state.text()
    assert "age" not in window.fps_state.text()


def test_camera_stats_handle_single_camera_preview_overlay():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.left_worker = FakeStatsWorker("left", 0, 29.84, 120, 0.03)
    window.right_worker = None

    window._update_camera_stats()

    assert app is not None
    assert window.left_view.overlay_text() == "idx 0\n29.8 fps\nframes 120\nage 0.03s"
    assert window.right_view.overlay_text() == ""
    assert window.camera_stats.text() == "L 29.8 fps"
    assert window.fps_state.text() == "Camera: L 29.8 fps"


def test_aruco_dictionary_and_board_creation():
    config = ArucoConfig(
        dictionary="DICT_4X4_50",
        markers_x=5,
        markers_y=7,
        marker_length_m=0.03,
        marker_separation_m=0.01,
    )

    assert isinstance(dictionary_id("DICT_4X4_50"), int)
    board = create_board(config)
    assert len(board.getIds()) == 35


def test_apriltag_36h11_detector_uses_two_border_bits_for_this_board():
    config = ArucoConfig(
        dictionary="DICT_APRILTAG_36h11",
        markers_x=5,
        markers_y=7,
        marker_length_m=0.08,
        marker_separation_m=0.024,
    )

    detector = create_detector(config)

    assert detector.getDetectorParameters().markerBorderBits == 2


def test_t36h11_5x7_printed_board_uses_7x5_horizontal_mirrored_object_map():
    config = ArucoConfig(
        dictionary="DICT_APRILTAG_36h11",
        markers_x=7,
        markers_y=5,
        marker_length_m=0.08,
        marker_separation_m=0.024,
        board_mirror_x=True,
    )

    mapping = board_object_map(config)

    assert len(mapping) == 35
    np.testing.assert_allclose(mapping[0][0], [0.624, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(mapping[6][0], [0.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(mapping[34][0], [0.0, 0.416, 0.0], atol=1e-6)


def test_analyze_calibration_pairs_reports_blank_pair_as_invalid(tmp_path):
    config = ArucoConfig(
        dictionary="DICT_APRILTAG_36h11",
        markers_x=5,
        markers_y=7,
        marker_length_m=0.08,
        marker_separation_m=0.024,
    )
    blank = np.zeros((24, 32, 3), dtype=np.uint8)
    save_image_pair(tmp_path, blank, blank)

    report = analyze_calibration_pairs(tmp_path, config)

    assert report.total_pairs == 1
    assert report.readable_pairs == 1
    assert report.valid_pairs == 0
    assert report.invalid_pairs == 1
    assert report.image_size == (32, 24)
    assert report.pair_stats[0].left_markers == 0
    assert report.pair_stats[0].right_markers == 0
    assert report.pair_stats[0].common_points == 0
    assert report.pair_stats[0].valid is False
    assert report.pair_stats[0].status == "INVALID"
    assert "Need 16 common corners" in report.pair_stats[0].reason


def test_calibration_quality_label_marks_large_error_as_poor():
    assert calibration_quality_label(0.8) == "GOOD"
    assert calibration_quality_label(2.5) == "CHECK"
    assert calibration_quality_label(12.0) == "POOR"


def test_calibrate_mono_from_images_saves_mono_files_separate_from_stereo(monkeypatch, tmp_path):
    frame = np.full((32, 40, 3), 80, dtype=np.uint8)
    save_single_image(tmp_path / "images", frame)
    obj = np.zeros((40, 3), dtype=np.float32)
    pts = np.zeros((40, 2), dtype=np.float32)

    monkeypatch.setattr(
        "stereo_aruco_gui.app.calibration.collect_mono_calibration_points",
        lambda image_root, aruco_config: ([obj], [pts], (40, 32), 1),
    )

    def fake_calibrate(obj_points, image_points, image_size, camera_matrix, dist_coeffs):  # noqa: ANN001
        return 0.75, np.eye(3), np.zeros((1, 5)), [], []

    monkeypatch.setattr("stereo_aruco_gui.app.calibration.cv2.calibrateCamera", fake_calibrate)

    result = calibrate_mono_from_images(tmp_path / "images", tmp_path / "output", ArucoConfig(), min_valid_images=1)

    assert result.error == 0.75
    assert result.image_size == (40, 32)
    assert result.valid_images == 1
    np.testing.assert_allclose(result.K, np.eye(3))
    np.testing.assert_allclose(result.D, np.zeros((1, 5)))
    assert (tmp_path / "output" / "mono_calib.npz").exists()
    assert (tmp_path / "output" / "mono_calib.yaml").exists()
    assert not (tmp_path / "output" / "stereo_calib.npz").exists()
    with np.load(tmp_path / "output" / "mono_calib.npz") as data:
        assert data["calibration_type"] == "mono"


def test_pair_quality_status_marks_weak_pairs_but_excludes_them_from_calibration():
    assert pair_quality_status(8) == ("INVALID", False, "Need 16 common corners, found 8")
    assert pair_quality_status(28) == ("WEAK", False, "Weak pair: need 40 common corners for calibration, found 28")
    assert pair_quality_status(40) == ("VALID", True, "OK")


def test_main_window_shows_calibration_pair_progress_on_preview():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    preview = make_calibration_preview(1, valid=False, left_markers=0, right_markers=0)

    window._show_calibration_progress(preview, total_pairs=3)

    assert app is not None
    assert "Pair 0001/3" in window.calibration_metrics.text()
    assert "INVALID" in window.calibration_metrics.text()
    assert "Need 16 common corners" in window.diagnostics_label.text()
    assert window.left_view.pixmap() is not None
    assert window.right_view.pixmap() is not None


def test_calibration_review_controls_switch_saved_previews():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    previews = [
        make_calibration_preview(1, valid=True),
        make_calibration_preview(
            2,
            valid=False,
            status="WEAK",
            common_points=28,
            left_markers=8,
            right_markers=7,
        ),
    ]

    window._set_calibration_review_items(previews)

    assert app is not None
    assert window.calibration_pair_select.count() == 2
    assert window.calibration_pair_select.isEnabled() is True
    assert window.prev_calibration_pair_button.isEnabled() is False
    assert window.next_calibration_pair_button.isEnabled() is True
    assert window.calibration_pair_select.itemText(1).startswith("0002 WEAK")

    window._show_next_calibration_pair()

    assert window.calibration_pair_select.currentIndex() == 1
    assert "Pair 0002/2 WEAK" in window.calibration_metrics.text()
    assert "Weak pair" in window.diagnostics_label.text()
    assert window.prev_calibration_pair_button.isEnabled() is True
    assert window.next_calibration_pair_button.isEnabled() is False
    assert window._freeze_enabled is True
    assert window.preview_info.text() == "Calibration Review"
    assert window.frozen_left is not None
    assert int(window.frozen_left[-1, -1, 0]) == 20

    window._show_previous_calibration_pair()

    assert window.calibration_pair_select.currentIndex() == 0
    assert "Pair 0001/2 VALID" in window.calibration_metrics.text()
    assert window.frozen_left is not None
    assert int(window.frozen_left[-1, -1, 0]) == 80


def test_main_window_freezes_final_calibration_result_preview():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    preview = make_calibration_preview(1, valid=True)
    result = StereoCalibrationResult(
        left_error=1.0,
        right_error=1.2,
        stereo_error=2.0,
        image_size=(32, 24),
        K1=np.eye(3),
        D1=np.zeros((1, 5)),
        K2=np.eye(3),
        D2=np.zeros((1, 5)),
        R=np.eye(3),
        T=np.array([[0.1], [0.0], [0.0]]),
        E=np.eye(3),
        F=np.eye(3),
        R1=np.eye(3),
        R2=np.eye(3),
        P1=np.zeros((3, 4)),
        P2=np.zeros((3, 4)),
        Q=np.eye(4),
    )
    window._last_calibration_preview = preview

    window._show_calibration_result(result, baseline=0.1, quality="CHECK")

    assert app is not None
    assert window._freeze_enabled is True
    assert window.freeze_button.text() == "Unfreeze"
    assert window.frozen_left is not None
    assert window.frozen_right is not None
    assert window.preview_info.text() == "Calibration Result"


def test_calibration_preview_text_splits_multiline_overlay():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    frame = np.zeros((120, 180, 3), dtype=np.uint8)

    output = window._draw_calibration_preview(frame, None, "RESULT CHECK\nL err 1.0000", "stereo 2.0000")

    assert app is not None
    assert output is not None
    assert output.shape == frame.shape
    assert np.any(output != frame)


def test_save_and_list_image_pair(tmp_path):
    left = np.zeros((24, 32, 3), dtype=np.uint8)
    right = np.full((24, 32, 3), 255, dtype=np.uint8)

    pair = save_image_pair(tmp_path, left, right)
    pairs = list_image_pairs(tmp_path)

    assert pair.index == 1
    assert pair.left_path.exists()
    assert pair.right_path.exists()
    assert len(pairs) == 1


def test_save_and_list_single_images_use_separate_directory(tmp_path):
    frame = np.full((24, 32, 3), 80, dtype=np.uint8)

    image = save_single_image(tmp_path, frame)
    images = list_single_images(tmp_path)

    assert image.index == 1
    assert image.path == tmp_path / "single" / "0001.png"
    assert image.path.exists()
    assert images == [image]
    assert list_image_pairs(tmp_path) == []


def test_delete_last_image_pair_removes_highest_index_pair(tmp_path):
    left = np.zeros((24, 32, 3), dtype=np.uint8)
    right = np.full((24, 32, 3), 255, dtype=np.uint8)
    save_image_pair(tmp_path, left, right)
    save_image_pair(tmp_path, left, right)

    deleted = delete_last_image_pair(tmp_path)

    assert deleted is not None
    assert deleted.index == 2
    assert [pair.index for pair in list_image_pairs(tmp_path)] == [1]


def test_camera_selection_warning_blocks_duplicate_stereo_indexes():
    assert camera_selection_warning(0, 0, single_mode=False) == "Left and right cameras must be different."
    assert camera_selection_warning(0, 0, single_mode=True) is None


def test_pair_count_text_includes_recommended_target():
    assert pair_count_text(24) == "Pairs: 24 / recommended 30+"


def test_view_mode_enabled_requires_calibration_for_rectified_and_depth():
    assert view_mode_enabled("Live Preview", has_calibration=False) is True
    assert view_mode_enabled("ArUco Detection", has_calibration=False) is True
    assert view_mode_enabled("Rectified Preview", has_calibration=False) is False
    assert view_mode_enabled("Depth / Distance", has_calibration=False) is False
    assert view_mode_enabled("Depth / Distance", has_calibration=True) is True


def test_scan_camera_indices_uses_injected_probe():
    def fake_probe(index, backend, width, height, fps):
        return index == 1, (height, width, 3) if index == 1 else None, width, height, fps

    results = scan_camera_indices(
        indexes=[0, 1],
        backends=[("TEST", 123)],
        width=640,
        height=480,
        fps=30,
        probe=fake_probe,
    )

    assert len(results) == 2
    assert results[0].index == 0
    assert results[0].read_ok is False
    assert results[1].index == 1
    assert results[1].read_ok is True
    assert results[1].shape == (480, 640, 3)


def test_default_camera_scan_avoids_msmf_backend():
    names = [name for name, _ in DEFAULT_SCAN_BACKENDS]

    assert names == ["DSHOW"]


def test_usable_frame_rejects_black_frame():
    black = np.zeros((20, 20, 3), dtype=np.uint8)
    visible = np.full((20, 20, 3), 80, dtype=np.uint8)

    assert usable_frame(black) is False
    assert usable_frame(visible) is True


def test_black_frame_is_still_accepted_for_preview():
    black = np.zeros((20, 20, 3), dtype=np.uint8)

    assert accept_frame_for_preview(black) is True


def test_preview_copy_downscales_large_frame():
    frame = np.zeros((960, 1280, 3), dtype=np.uint8)

    preview = preview_copy(frame, max_width=640)

    assert preview.shape == (480, 640, 3)


def test_read_stereo_pair_falls_back_to_read_when_retrieve_fails():
    frame = np.full((20, 20, 3), 80, dtype=np.uint8)

    class FakeCapture:
        def grab(self):
            return True

        def retrieve(self):
            return False, None

        def read(self):
            return True, frame

    ok, left, right = read_stereo_pair(FakeCapture(), FakeCapture())

    assert ok is True
    assert left is frame
    assert right is frame


def test_receive_frame_pair_stores_latest_and_counts_frames():
    worker = CameraWorker(CameraConfig())
    left1 = np.full((4, 4, 3), 10, dtype=np.uint8)
    right1 = np.full((4, 4, 3), 20, dtype=np.uint8)
    left2 = np.full((4, 4, 3), 30, dtype=np.uint8)
    right2 = np.full((4, 4, 3), 40, dtype=np.uint8)

    worker.receive_frame_pair(left1, right1)
    worker.receive_frame_pair(left2, right2)
    latest = worker.get_latest()

    assert worker.received_frame_count == 2
    assert latest is not None
    latest_left, latest_right = latest
    assert np.array_equal(latest_left, left2)
    assert np.array_equal(latest_right, right2)


def test_receive_single_frame_stores_only_left_frame():
    worker = CameraWorker(CameraConfig(single_mode=True))
    frame = np.full((4, 4, 3), 20, dtype=np.uint8)

    worker.receive_frame_pair(frame, None)
    latest = worker.get_latest()

    assert worker.received_frame_count == 1
    assert latest is not None
    latest_left, latest_right = latest
    assert np.array_equal(latest_left, frame)
    assert latest_right is None


def test_single_camera_worker_stores_latest_frame():
    worker = SingleCameraWorker(index=0, width=640, height=480, fps=15, label="test")
    frame1 = np.full((4, 4, 3), 10, dtype=np.uint8)
    frame2 = np.full((4, 4, 3), 30, dtype=np.uint8)

    worker.receive_frame(frame1)
    worker.receive_frame(frame2)
    latest = worker.get_latest()

    assert worker.received_frame_count == 2
    assert latest is not None
    assert np.array_equal(latest, frame2)
    assert worker.snapshot_stats()["frames"] == 2


def test_single_camera_worker_stores_black_frames_for_preview():
    worker = SingleCameraWorker(index=0, width=640, height=480, fps=15, label="test")
    black = np.zeros((4, 4, 3), dtype=np.uint8)

    worker.receive_frame(black)
    latest = worker.get_latest()

    assert latest is not None
    assert np.array_equal(latest, black)
    assert worker.snapshot_stats()["frames"] == 1


def test_should_decode_frame_respects_interval():
    assert should_decode_frame(now=1.0, last_decode=0.0, interval=0.5) is True
    assert should_decode_frame(now=1.2, last_decode=1.0, interval=0.5) is False


def test_single_camera_capture_loop_reads_every_frame_without_grab_throttling(monkeypatch):
    frame = np.full((4, 4, 3), 80, dtype=np.uint8)

    class DirectReadCapture:
        def __init__(self) -> None:
            self.reads = 0
            self.grabs = 0
            self.retrieves = 0

        def read(self):
            self.reads += 1
            return True, frame

        def grab(self):
            self.grabs += 1
            return True

        def retrieve(self):
            self.retrieves += 1
            return True, frame

        def release(self) -> None:
            pass

    capture = DirectReadCapture()
    worker = SingleCameraWorker(index=0, width=640, height=480, fps=120, label="left")
    worker._capture = capture
    worker._running = True

    def stop_after_three_reads(frame_arg):  # noqa: ANN001
        SingleCameraWorker.receive_frame(worker, frame_arg)
        if worker.received_frame_count >= 3:
            worker._running = False

    monkeypatch.setattr(worker, "_open", lambda: "MSMF")
    monkeypatch.setattr(worker, "receive_frame", stop_after_three_reads)

    worker._run_capture_loop()

    assert worker.received_frame_count == 3
    assert capture.reads == 3
    assert capture.grabs == 0
    assert capture.retrieves == 0


def test_single_camera_worker_auto_backend_avoids_msmf_fallback():
    worker = SingleCameraWorker(index=0, width=640, height=480, fps=15, label="test", backend="AUTO")

    assert worker._next_backend_names() == ["DSHOW"]
    worker._mark_backend_failed("DSHOW")
    assert worker._next_backend_names() == ["DSHOW"]


def test_single_camera_worker_uses_ten_second_open_timeout_by_default():
    worker = SingleCameraWorker(index=0, width=640, height=480, fps=15, label="test")

    assert worker.open_timeout_s == 10.0


def test_single_camera_worker_requires_stable_frames_before_open_success(monkeypatch):
    frame = np.full((4, 4, 3), 80, dtype=np.uint8)

    class FlakyStartupCapture:
        def __init__(self) -> None:
            self.reads = 0
            self.released = False

        def isOpened(self) -> bool:  # noqa: N802
            return True

        def set(self, prop, value):  # noqa: ANN001
            return True

        def get(self, prop):  # noqa: ANN001
            return 640

        def read(self):
            self.reads += 1
            if self.reads == 1:
                return True, frame
            return False, None

        def release(self) -> None:
            self.released = True

    capture = FlakyStartupCapture()
    monkeypatch.setattr(camera_worker_module.cv2, "VideoCapture", lambda index, backend: capture)
    monkeypatch.setattr(camera_worker_module.time, "sleep", lambda seconds: None)
    worker = SingleCameraWorker(index=0, width=640, height=480, fps=15, label="left", open_timeout_s=0.01)

    backend = worker._open()

    assert backend == "none"
    assert worker._capture is None
    assert worker.latest_frame is None
    assert capture.released is True
    assert worker.open_attempts[-1].error == "open timeout"


def test_single_camera_worker_stores_startup_frame_after_stable_open(monkeypatch):
    frame = np.full((4, 4, 3), 80, dtype=np.uint8)

    class StableStartupCapture:
        def isOpened(self) -> bool:  # noqa: N802
            return True

        def set(self, prop, value):  # noqa: ANN001
            return True

        def get(self, prop):  # noqa: ANN001
            return 640

        def read(self):
            return True, frame

        def release(self) -> None:
            pass

    monkeypatch.setattr(camera_worker_module.cv2, "VideoCapture", lambda index, backend: StableStartupCapture())
    worker = SingleCameraWorker(index=0, width=640, height=480, fps=15, label="left", open_timeout_s=0.2)

    backend = worker._open()

    assert backend == "DSHOW"
    assert worker.get_latest() is not None
    assert worker.received_frame_count == 1


def test_open_failure_summary_reports_attempt_details():
    attempts = [
        OpenAttempt(
            backend_name="DSHOW",
            opened=True,
            read_ok=False,
            usable=False,
            actual_width=1280,
            actual_height=960,
            actual_fps=15,
            shape=None,
        )
    ]

    summary = open_failure_summary("left", 0, 1280, 960, 15, "MJPG", attempts)

    assert "left: failed to open camera 0" in summary
    assert "requested=1280x960@15 MJPG" in summary
    assert "DSHOW opened=True read_ok=False usable=False actual=1280x960@15.0 shape=None" in summary


def test_configure_capture_does_not_raise_when_backend_rejects_setting():
    class RejectingCapture:
        def set(self, prop, value):  # noqa: ANN001
            raise RuntimeError(f"reject {prop}={value}")

    assert configure_capture(RejectingCapture(), 1024, 768, 15, "MJPG") is False


def test_configure_capture_allows_false_set_return_values():
    class FalseReturningCapture:
        def set(self, prop, value):  # noqa: ANN001
            return False

    assert configure_capture(FalseReturningCapture(), 640, 480, 15, "MJPG") is True


def test_camera_worker_finished_restores_open_button_state():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.open_button.setText("Close Cameras")
    window.preview_timer.start()

    window._camera_worker_finished()

    assert app is not None
    assert window.open_button.text() == "Open Cameras"
    assert not window.preview_timer.isActive()
    assert window.left_worker is None
    assert window.right_worker is None


def test_camera_opening_state_disables_button_until_all_workers_report_opened():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window._opening_labels = {"left", "right"}
    window.open_button.setText("Opening Cameras")
    window.open_button.setEnabled(False)

    window._handle_camera_status("left: opened index 0 with DSHOW")
    assert window.open_button.isEnabled() is False

    window._handle_camera_status("right: opened index 1 with DSHOW")
    assert app is not None
    assert window.open_button.text() == "Close Cameras"
    assert window.open_button.isEnabled() is True


def test_toggle_cameras_closes_when_only_right_worker_is_running():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    left_worker = FakeStoppedWorker()
    right_worker = FakeRunningWorker()
    window.left_worker = left_worker
    window.right_worker = right_worker
    window.open_button.setText("Close Cameras")

    window._toggle_cameras()

    assert app is not None
    assert left_worker.stop_called is True
    assert right_worker.stop_called is True
    assert window.open_button.text() == "Stopping Cameras"
    assert window.open_button.isEnabled() is False


def test_stereo_open_failure_closes_other_camera_and_restores_open_state():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    left_worker = FakeRunningWorker()
    right_worker = FakeStoppedWorker()
    window.left_worker = left_worker
    window.right_worker = right_worker
    window._opening_labels = {"left", "right"}
    window.open_button.setText("Opening Cameras")
    window.open_button.setEnabled(False)

    window._handle_camera_status("left: opened index 0 with MSMF")
    window._handle_camera_status(
        "right: failed to open camera 1; requested=1280x960@30 MJPG; MSMF opened=True read_ok=False usable=False actual=1280x960@30.0 shape=None error=open timeout"
    )

    assert app is not None
    assert left_worker.stop_called is True
    assert right_worker.stop_called is True
    assert window.open_button.text() == "Open Cameras"
    assert window.open_button.isEnabled() is True


def test_refresh_preview_can_show_right_frame_while_waiting_for_left():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    right = np.full((20, 20, 3), 80, dtype=np.uint8)
    window.left_worker = FakeFrameWorker(None)
    window.right_worker = FakeFrameWorker(right)

    window._refresh_preview()

    assert app is not None
    assert window.latest_left is None
    assert window.latest_right is right
    assert window.left_view.text() == "Waiting for left camera"


def test_main_window_freeze_uses_frozen_frames_until_unfrozen():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    left_a = np.full((20, 20, 3), 10, dtype=np.uint8)
    right_a = np.full((20, 20, 3), 20, dtype=np.uint8)
    left_b = np.full((20, 20, 3), 30, dtype=np.uint8)
    right_b = np.full((20, 20, 3), 40, dtype=np.uint8)
    window.latest_left = left_a
    window.latest_right = right_a

    window._toggle_freeze()
    window._update_frames(left_b, right_b)

    assert app is not None
    assert window.freeze_button.text() == "Unfreeze"
    assert np.array_equal(window.frozen_left, left_a)
    assert np.array_equal(window.frozen_right, right_a)
    assert np.array_equal(window.latest_left, left_b)
    assert np.array_equal(window.latest_right, right_b)
    assert window.preview_info.text() == "Frozen Preview"


def test_main_window_freeze_ignores_request_when_no_frames_available():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())

    window._toggle_freeze()

    assert app is not None
    assert window.freeze_button.text() == "Freeze Frame"
    assert window.frozen_left is None
    assert window.frozen_right is None
    assert "No camera frames available to freeze" in window.status_box.toPlainText()


def test_main_window_preview_click_uses_original_frame_coordinates():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    window.view_mode.setCurrentText("Live Preview")

    window._handle_preview_click("left", 400, 300)

    assert app is not None
    assert window.distance_state.text() == "Point: left 400, 300"
    assert "Clicked left image point x=400, y=300" in window.status_box.toPlainText()


def test_original_view_buttons_require_frames():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())

    window._open_original_view("left")

    assert app is not None
    assert "No left frame available" in window.status_box.toPlainText()


def test_single_camera_capture_saves_single_image_separate_from_pairs(tmp_path):
    app = QApplication.instance() or QApplication([])
    config = load_config()
    config.capture.output_dir = str(tmp_path / "images")
    window = MainWindow(config)
    frame = np.full((20, 20, 3), 90, dtype=np.uint8)
    window.single_mode.setChecked(True)
    window.latest_left = frame
    window.latest_right = None

    window._capture_pair()

    assert app is not None
    assert len(list_single_images(config.capture.output_dir)) == 1
    assert list_image_pairs(config.capture.output_dir) == []
    assert "Saved single image 0001" in window.status_box.toPlainText()


def test_single_camera_calibrate_uses_mono_output_without_stereo_rectifier(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    config = load_config()
    config.capture.output_dir = str(tmp_path / "images")
    config.output.dir = str(tmp_path / "output")
    window = MainWindow(config)
    window.single_mode.setChecked(True)
    result = MonoCalibrationResult(
        error=0.55,
        image_size=(40, 32),
        K=np.eye(3),
        D=np.zeros((1, 5)),
        valid_images=18,
    )
    called = {}

    def fake_calibrate_mono(image_root, output_dir, aruco_config):  # noqa: ANN001
        called["image_root"] = image_root
        called["output_dir"] = output_dir
        return result

    monkeypatch.setattr(main_window_module, "calibrate_mono_from_images", fake_calibrate_mono)

    window._calibrate()

    assert app is not None
    assert called == {"image_root": config.capture.output_dir, "output_dir": config.output.dir}
    assert window.rectifier is None
    assert window.calibration_state.text() == "Calibration: mono error 0.5500"
    assert window.baseline_state.text() == "Baseline: mono N/A"
    assert window.output_path_label.text().endswith("mono_calib.npz")
    assert "Mono calibration done" in window.status_box.toPlainText()


def test_stop_cameras_keeps_running_worker_until_finished():
    app = QApplication.instance() or QApplication([])
    window = MainWindow(load_config())
    worker = FakeRunningWorker()
    window.left_worker = worker
    window.open_button.setText("Close Cameras")

    window._stop_cameras("test stop")

    assert app is not None
    assert worker.stop_called is True
    assert window.left_worker is worker
    assert window.open_button.text() == "Stopping Cameras"
    assert window.open_button.isEnabled() is False


def test_receive_frame_pair_does_not_emit_qt_frame_signal_by_default():
    worker = CameraWorker(CameraConfig())
    frame = np.full((4, 4, 3), 20, dtype=np.uint8)

    worker.receive_frame_pair(frame, frame)

    assert not hasattr(worker, "frames_ready")
