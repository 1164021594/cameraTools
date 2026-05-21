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


def board_object_map(config: ArucoConfig) -> dict[int, np.ndarray]:
    if config.markers_x < 1 or config.markers_y < 1:
        raise ValueError("markers_x and markers_y must be positive")
    if config.marker_length_m <= 0 or config.marker_separation_m < 0:
        raise ValueError("marker sizes must be positive")
    step = config.marker_length_m + config.marker_separation_m
    object_points: dict[int, np.ndarray] = {}
    for row in range(config.markers_y):
        for col in range(config.markers_x):
            marker_id = row * config.markers_x + col
            mapped_col = config.markers_x - 1 - col if config.board_mirror_x else col
            x0 = float(mapped_col * step)
            y0 = float(row * step)
            length = float(config.marker_length_m)
            object_points[marker_id] = np.asarray(
                [
                    [x0, y0, 0.0],
                    [x0 + length, y0, 0.0],
                    [x0 + length, y0 + length, 0.0],
                    [x0, y0 + length, 0.0],
                ],
                dtype=np.float32,
            )
    return object_points


def create_detector(config: ArucoConfig) -> cv2.aruco.ArucoDetector:
    params = cv2.aruco.DetectorParameters()
    if "APRILTAG_36" in config.dictionary.upper():
        params.markerBorderBits = 2
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
    board: cv2.aruco.GridBoard | dict[int, np.ndarray],
    left: DetectionResult,
    right: DetectionResult,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if left.ids is None or right.ids is None:
        return _empty_points()

    if isinstance(board, dict):
        object_points_by_id = board
    else:
        board_ids = board.getIds().flatten()
        board_obj_points = board.getObjPoints()
        object_points_by_id = {
            int(marker_id): np.asarray(board_obj_points[index], dtype=np.float32)
            for index, marker_id in enumerate(board_ids)
        }
    left_ids = left.ids.flatten()
    right_ids = right.ids.flatten()
    common_ids = np.intersect1d(left_ids, right_ids)

    obj_points: list[np.ndarray] = []
    img_points_l: list[np.ndarray] = []
    img_points_r: list[np.ndarray] = []

    for marker_id in common_ids:
        marker_id = int(marker_id)
        if marker_id not in object_points_by_id:
            continue
        idx_l = int(np.where(left_ids == marker_id)[0][0])
        idx_r = int(np.where(right_ids == marker_id)[0][0])

        obj_points.extend(np.asarray(object_points_by_id[marker_id], dtype=np.float32))
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
