from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from stereo_aruco_gui.app.aruco_board import DetectionResult, board_object_map, common_board_points, detect_markers
from stereo_aruco_gui.app.config import ArucoConfig
from stereo_aruco_gui.app.storage import (
    ImagePair,
    list_image_pairs,
    list_single_images,
    save_calibration_npz,
    save_calibration_yaml,
)


MIN_DETECTED_POINTS_PER_PAIR = 16
MIN_CALIBRATION_POINTS_PER_PAIR = 40
GOOD_STEREO_ERROR = 1.0
CHECK_STEREO_ERROR = 3.0


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


@dataclass(frozen=True)
class MonoCalibrationResult:
    error: float
    image_size: tuple[int, int]
    K: np.ndarray
    D: np.ndarray
    valid_images: int

    def as_dict(self) -> dict[str, np.ndarray | float | int | str | tuple[int, int]]:
        return {
            "calibration_type": "mono",
            "error": float(self.error),
            "image_size": self.image_size,
            "K": self.K,
            "D": self.D,
            "valid_images": int(self.valid_images),
        }


@dataclass(frozen=True)
class PairDetectionStats:
    index: int
    filename: str
    readable: bool
    left_markers: int
    right_markers: int
    common_points: int
    valid: bool
    status: str
    reason: str


@dataclass(frozen=True)
class CalibrationPairPreview:
    stats: PairDetectionStats
    left_image: np.ndarray | None
    right_image: np.ndarray | None
    left_detection: DetectionResult | None
    right_detection: DetectionResult | None


@dataclass(frozen=True)
class CalibrationDatasetReport:
    total_pairs: int
    readable_pairs: int
    valid_pairs: int
    image_size: tuple[int, int] | None
    pair_stats: list[PairDetectionStats]

    @property
    def invalid_pairs(self) -> int:
        return sum(1 for stats in self.pair_stats if stats.status == "INVALID")

    @property
    def weak_pairs(self) -> int:
        return sum(1 for stats in self.pair_stats if stats.status == "WEAK")

    @property
    def invalid_filenames(self) -> list[str]:
        return [stats.filename for stats in self.pair_stats if stats.status == "INVALID"]

    @property
    def weak_filenames(self) -> list[str]:
        return [stats.filename for stats in self.pair_stats if stats.status == "WEAK"]

    @property
    def average_common_points(self) -> float:
        if not self.pair_stats:
            return 0.0
        return float(sum(stats.common_points for stats in self.pair_stats) / len(self.pair_stats))


def calibration_quality_label(stereo_error: float) -> str:
    if stereo_error <= GOOD_STEREO_ERROR:
        return "GOOD"
    if stereo_error <= CHECK_STEREO_ERROR:
        return "CHECK"
    return "POOR"


def pair_quality_status(
    common_points: int,
    min_detected_points: int = MIN_DETECTED_POINTS_PER_PAIR,
    min_calibration_points: int = MIN_CALIBRATION_POINTS_PER_PAIR,
) -> tuple[str, bool, str]:
    if common_points < min_detected_points:
        return "INVALID", False, f"Need {min_detected_points} common corners, found {common_points}"
    if common_points < min_calibration_points:
        return "WEAK", False, f"Weak pair: need {min_calibration_points} common corners for calibration, found {common_points}"
    return "VALID", True, "OK"


def analyze_calibration_pairs(
    image_root: Path | str,
    aruco_config: ArucoConfig,
    min_points_per_pair: int = MIN_CALIBRATION_POINTS_PER_PAIR,
    on_pair: Callable[[CalibrationPairPreview], None] | None = None,
) -> CalibrationDatasetReport:
    pairs = list_image_pairs(image_root)
    board = board_object_map(aruco_config)
    pair_stats: list[PairDetectionStats] = []
    readable_pairs = 0
    valid_pairs = 0
    image_size: tuple[int, int] | None = None

    for pair in pairs:
        left_image = cv2.imread(str(pair.left_path))
        right_image = cv2.imread(str(pair.right_path))
        if left_image is None or right_image is None:
            stats = PairDetectionStats(
                index=pair.index,
                filename=pair.left_path.name,
                readable=False,
                left_markers=0,
                right_markers=0,
                common_points=0,
                valid=False,
                status="INVALID",
                reason="Unreadable image pair",
            )
            pair_stats.append(stats)
            if on_pair is not None:
                on_pair(CalibrationPairPreview(stats, left_image, right_image, None, None))
            continue

        readable_pairs += 1
        current_size = (left_image.shape[1], left_image.shape[0])
        right_size = (right_image.shape[1], right_image.shape[0])
        if image_size is None:
            image_size = current_size

        if current_size != image_size or right_size != image_size:
            stats = PairDetectionStats(
                index=pair.index,
                filename=pair.left_path.name,
                readable=True,
                left_markers=0,
                right_markers=0,
                common_points=0,
                valid=False,
                status="INVALID",
                reason=f"Image size mismatch: left={current_size}, right={right_size}, expected={image_size}",
            )
            pair_stats.append(stats)
            if on_pair is not None:
                on_pair(CalibrationPairPreview(stats, left_image, right_image, None, None))
            continue

        left_detection = detect_markers(left_image, aruco_config)
        right_detection = detect_markers(right_image, aruco_config)
        obj, _, _ = common_board_points(board, left_detection, right_detection)
        common_points = int(len(obj))
        status, valid, reason = pair_quality_status(common_points, min_calibration_points=min_points_per_pair)
        if valid:
            valid_pairs += 1

        stats = PairDetectionStats(
            index=pair.index,
            filename=pair.left_path.name,
            readable=True,
            left_markers=left_detection.count,
            right_markers=right_detection.count,
            common_points=common_points,
            valid=valid,
            status=status,
            reason=reason,
        )
        pair_stats.append(stats)
        if on_pair is not None:
            on_pair(CalibrationPairPreview(stats, left_image, right_image, left_detection, right_detection))

    return CalibrationDatasetReport(
        total_pairs=len(pairs),
        readable_pairs=readable_pairs,
        valid_pairs=valid_pairs,
        image_size=image_size,
        pair_stats=pair_stats,
    )


def collect_calibration_points(
    pairs: list[ImagePair],
    aruco_config: ArucoConfig,
    min_points_per_pair: int = MIN_CALIBRATION_POINTS_PER_PAIR,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int], int]:
    board = board_object_map(aruco_config)
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


def collect_mono_calibration_points(
    image_root: Path | str,
    aruco_config: ArucoConfig,
    min_points_per_image: int = MIN_CALIBRATION_POINTS_PER_PAIR,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int], int]:
    board = board_object_map(aruco_config)
    all_obj_points: list[np.ndarray] = []
    all_image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for image in list_single_images(image_root):
        frame = cv2.imread(str(image.path))
        if frame is None:
            continue
        if image_size is None:
            image_size = (frame.shape[1], frame.shape[0])
        if (frame.shape[1], frame.shape[0]) != image_size:
            continue

        detection = detect_markers(frame, aruco_config)
        if detection.ids is None:
            continue
        obj_points: list[np.ndarray] = []
        image_points: list[np.ndarray] = []
        for marker_index, marker_id in enumerate(detection.ids.flatten()):
            marker_id = int(marker_id)
            if marker_id not in board:
                continue
            obj_points.extend(np.asarray(board[marker_id], dtype=np.float32))
            image_points.extend(np.asarray(detection.corners[marker_index][0], dtype=np.float32))
        if len(obj_points) >= min_points_per_image:
            all_obj_points.append(np.asarray(obj_points, dtype=np.float32))
            all_image_points.append(np.asarray(image_points, dtype=np.float32))

    if image_size is None:
        raise RuntimeError("No readable single camera images found")

    return all_obj_points, all_image_points, image_size, len(all_obj_points)


def calibrate_mono_from_images(
    image_root: Path | str,
    output_dir: Path | str,
    aruco_config: ArucoConfig,
    min_valid_images: int = 15,
) -> MonoCalibrationResult:
    images = list_single_images(image_root)
    if len(images) < min_valid_images:
        raise RuntimeError(f"Need at least {min_valid_images} single camera images, found {len(images)}")

    obj_points, image_points, image_size, valid_images = collect_mono_calibration_points(image_root, aruco_config)
    if valid_images < min_valid_images:
        raise RuntimeError(f"Need at least {min_valid_images} valid single camera images, found {valid_images}")

    error, K, D, _, _ = cv2.calibrateCamera(obj_points, image_points, image_size, None, None)
    result = MonoCalibrationResult(
        error=float(error),
        image_size=image_size,
        K=K,
        D=D,
        valid_images=valid_images,
    )
    save_calibration_npz(output_dir, result.as_dict(), stem="mono_calib")
    save_calibration_yaml(output_dir, result.as_dict(), stem="mono_calib")
    return result


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
