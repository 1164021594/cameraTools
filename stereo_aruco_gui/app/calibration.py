from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stereo_aruco_gui.app.aruco_board import common_board_points, create_board, detect_markers
from stereo_aruco_gui.app.config import ArucoConfig
from stereo_aruco_gui.app.storage import ImagePair, list_image_pairs, save_calibration_npz, save_calibration_yaml


@dataclass
class StereoCalibrationResult:
    left_error: float
    right_error: float
    stereo_error: float
    image_size: tuple[int, int]
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray | float | tuple[int, int]]:
        return {
            "left_error": float(self.left_error),
            "right_error": float(self.right_error),
            "stereo_error": float(self.stereo_error),
            "image_size": self.image_size,
            "K1": self.K1,
            "D1": self.D1,
            "K2": self.K2,
            "D2": self.D2,
            "R": self.R,
            "T": self.T,
            "E": self.E,
            "F": self.F,
            "R1": self.R1,
            "R2": self.R2,
            "P1": self.P1,
            "P2": self.P2,
            "Q": self.Q,
        }


def collect_calibration_points(
    pairs: list[ImagePair],
    aruco_config: ArucoConfig,
    min_points_per_pair: int = 16,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int], int]:
    board = create_board(aruco_config)
    all_obj_points: list[np.ndarray] = []
    all_left_points: list[np.ndarray] = []
    all_right_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for pair in pairs:
        left_image = cv2.imread(str(pair.left_path))
        right_image = cv2.imread(str(pair.right_path))
        if left_image is None or right_image is None:
            continue
        if image_size is None:
            image_size = (left_image.shape[1], left_image.shape[0])
        if (right_image.shape[1], right_image.shape[0]) != image_size:
            continue

        left_detection = detect_markers(left_image, aruco_config)
        right_detection = detect_markers(right_image, aruco_config)
        obj, left_pts, right_pts = common_board_points(board, left_detection, right_detection)
        if len(obj) >= min_points_per_pair:
            all_obj_points.append(obj)
            all_left_points.append(left_pts)
            all_right_points.append(right_pts)

    if image_size is None:
        raise RuntimeError("No readable image pairs found")

    return all_obj_points, all_left_points, all_right_points, image_size, len(all_obj_points)


def calibrate_from_pairs(
    image_root: Path | str,
    output_dir: Path | str,
    aruco_config: ArucoConfig,
    min_valid_pairs: int = 15,
) -> StereoCalibrationResult:
    pairs = list_image_pairs(image_root)
    if len(pairs) < min_valid_pairs:
        raise RuntimeError(f"Need at least {min_valid_pairs} image pairs, found {len(pairs)}")

    obj_points, left_points, right_points, image_size, valid_pairs = collect_calibration_points(pairs, aruco_config)
    if valid_pairs < min_valid_pairs:
        raise RuntimeError(f"Need at least {min_valid_pairs} valid detected pairs, found {valid_pairs}")

    left_error, K1, D1, _, _ = cv2.calibrateCamera(obj_points, left_points, image_size, None, None)
    right_error, K2, D2, _, _ = cv2.calibrateCamera(obj_points, right_points, image_size, None, None)

    stereo_error, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        obj_points,
        left_points,
        right_points,
        K1,
        D1,
        K2,
        D2,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
    )

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )

    result = StereoCalibrationResult(
        left_error=float(left_error),
        right_error=float(right_error),
        stereo_error=float(stereo_error),
        image_size=image_size,
        K1=K1,
        D1=D1,
        K2=K2,
        D2=D2,
        R=R,
        T=T,
        E=E,
        F=F,
        R1=R1,
        R2=R2,
        P1=P1,
        P2=P2,
        Q=Q,
    )
    save_calibration_npz(output_dir, result.as_dict())
    save_calibration_yaml(output_dir, result.as_dict())
    return result
