from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from stereo_aruco_gui.app.config import ArucoConfig


ARUCO_DICTIONARIES = {
    name: value
    for name, value in vars(cv2.aruco).items()
    if name.startswith("DICT_") and isinstance(value, int)
}


@dataclass(frozen=True)
class DetectionResult:
    corners: tuple[np.ndarray, ...]
    ids: np.ndarray | None
    rejected: tuple[np.ndarray, ...]

    @property
    def count(self) -> int:
        return 0 if self.ids is None else int(len(self.ids))


def dictionary_id(name: str) -> int:
    try:
        return ARUCO_DICTIONARIES[name]
    except KeyError as exc:
        names = ", ".join(sorted(ARUCO_DICTIONARIES))
        raise ValueError(f"Unknown ArUco dictionary '{name}'. Available: {names}") from exc


def create_dictionary(name: str) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(dictionary_id(name))


def create_board(config: ArucoConfig) -> cv2.aruco.GridBoard:
    if config.markers_x < 1 or config.markers_y < 1:
        raise ValueError("markers_x and markers_y must be positive")
    if config.marker_length_m <= 0 or config.marker_separation_m < 0:
        raise ValueError("marker sizes must be positive")
    return cv2.aruco.GridBoard(
        (config.markers_x, config.markers_y),
        config.marker_length_m,
        config.marker_separation_m,
        create_dictionary(config.dictionary),
    )


def create_detector(config: ArucoConfig) -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    return cv2.aruco.ArucoDetector(create_dictionary(config.dictionary), params)


def detect_markers(image: np.ndarray, config: ArucoConfig) -> DetectionResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    corners, ids, rejected = create_detector(config).detectMarkers(gray)
    return DetectionResult(tuple(corners), ids, tuple(rejected))


def draw_detection(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    output = image.copy()
    if result.ids is not None and result.count:
        cv2.aruco.drawDetectedMarkers(output, list(result.corners), result.ids)
    return output


def common_board_points(
    board: cv2.aruco.GridBoard,
    left: DetectionResult,
    right: DetectionResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if left.ids is None or right.ids is None:
        return _empty_points()

    board_ids = board.getIds().flatten()
    board_obj_points = board.getObjPoints()
    left_ids = left.ids.flatten()
    right_ids = right.ids.flatten()
    common_ids = np.intersect1d(left_ids, right_ids)

    obj_points: list[np.ndarray] = []
    img_points_l: list[np.ndarray] = []
    img_points_r: list[np.ndarray] = []

    for marker_id in common_ids:
        board_match = np.where(board_ids == marker_id)[0]
        if len(board_match) == 0:
            continue
        idx_board = int(board_match[0])
        idx_l = int(np.where(left_ids == marker_id)[0][0])
        idx_r = int(np.where(right_ids == marker_id)[0][0])

        obj_points.extend(np.asarray(board_obj_points[idx_board], dtype=np.float32))
        img_points_l.extend(np.asarray(left.corners[idx_l][0], dtype=np.float32))
        img_points_r.extend(np.asarray(right.corners[idx_r][0], dtype=np.float32))

    if not obj_points:
        return _empty_points()
    return (
        np.asarray(obj_points, dtype=np.float32),
        np.asarray(img_points_l, dtype=np.float32),
        np.asarray(img_points_r, dtype=np.float32),
    )


def _empty_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    empty3 = np.empty((0, 3), dtype=np.float32)
    empty2 = np.empty((0, 2), dtype=np.float32)
    return empty3, empty2, empty2.copy()
