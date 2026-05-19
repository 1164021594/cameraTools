from __future__ import annotations

import numpy as np

from stereo_aruco_gui.app.aruco_board import create_board, dictionary_id
from stereo_aruco_gui.app.camera_probe import scan_camera_indices
from stereo_aruco_gui.app.camera_worker import (
    CameraWorker,
    SingleCameraWorker,
    preview_copy,
    read_stereo_pair,
    should_decode_frame,
    usable_frame,
)
from stereo_aruco_gui.app.config import CameraConfig
from stereo_aruco_gui.app.config import ArucoConfig, load_config
from stereo_aruco_gui.app.main_window import parse_resolution_label
from stereo_aruco_gui.app.storage import delete_last_image_pair, list_image_pairs, save_image_pair
from stereo_aruco_gui.app.ui_state import camera_selection_warning, pair_count_text, view_mode_enabled


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


def test_save_and_list_image_pair(tmp_path):
    left = np.zeros((24, 32, 3), dtype=np.uint8)
    right = np.full((24, 32, 3), 255, dtype=np.uint8)

    pair = save_image_pair(tmp_path, left, right)
    pairs = list_image_pairs(tmp_path)

    assert pair.index == 1
    assert pair.left_path.exists()
    assert pair.right_path.exists()
    assert len(pairs) == 1


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


def test_usable_frame_rejects_black_frame():
    black = np.zeros((20, 20, 3), dtype=np.uint8)
    visible = np.full((20, 20, 3), 80, dtype=np.uint8)

    assert usable_frame(black) is False
    assert usable_frame(visible) is True


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


def test_should_decode_frame_respects_interval():
    assert should_decode_frame(now=1.0, last_decode=0.0, interval=0.5) is True
    assert should_decode_frame(now=1.2, last_decode=1.0, interval=0.5) is False


def test_single_camera_worker_auto_backend_avoids_msmf_fallback():
    worker = SingleCameraWorker(index=0, width=640, height=480, fps=15, label="test", backend="AUTO")

    assert worker._next_backend_names() == ["DSHOW"]
    worker._mark_backend_failed("DSHOW")
    assert worker._next_backend_names() == ["DSHOW"]


def test_receive_frame_pair_does_not_emit_qt_frame_signal_by_default():
    worker = CameraWorker(CameraConfig())
    frame = np.full((4, 4, 3), 20, dtype=np.uint8)

    worker.receive_frame_pair(frame, frame)

    assert not hasattr(worker, "frames_ready")
