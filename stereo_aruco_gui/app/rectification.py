from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Rectifier:
    map1_left: np.ndarray
    map2_left: np.ndarray
    map1_right: np.ndarray
    map2_right: np.ndarray
    Q: np.ndarray

    @classmethod
    def from_calibration(cls, calibration: dict[str, np.ndarray]) -> "Rectifier":
        image_size_raw = calibration["image_size"]
        image_size = tuple(int(x) for x in image_size_raw.tolist()) if hasattr(image_size_raw, "tolist") else tuple(image_size_raw)
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            calibration["K1"],
            calibration["D1"],
            calibration["R1"],
            calibration["P1"],
            image_size,
            cv2.CV_16SC2,
        )
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            calibration["K2"],
            calibration["D2"],
            calibration["R2"],
            calibration["P2"],
            image_size,
            cv2.CV_16SC2,
        )
        return cls(map1_left, map2_left, map1_right, map2_right, calibration["Q"])

    def rectify(self, left: np.ndarray, right: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        left_rect = cv2.remap(left, self.map1_left, self.map2_left, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, self.map1_right, self.map2_right, cv2.INTER_LINEAR)
        return left_rect, right_rect


def draw_horizontal_guides(image: np.ndarray, step: int = 80) -> np.ndarray:
    output = image.copy()
    height = output.shape[0]
    for y in range(step, height, step):
        cv2.line(output, (0, y), (output.shape[1], y), (0, 255, 255), 1)
    return output


def compute_disparity(left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY) if left_rect.ndim == 3 else left_rect
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY) if right_rect.ndim == 3 else right_rect
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=128,
        blockSize=5,
        P1=8 * 5 * 5,
        P2=32 * 5 * 5,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=2,
        disp12MaxDiff=1,
    )
    return matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0


def disparity_preview(disparity: np.ndarray) -> np.ndarray:
    valid = disparity > 0
    output = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(disparity, 0, np.percentile(disparity[valid], 95))
        output = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(output, cv2.COLORMAP_TURBO)


def distance_at(disparity: np.ndarray, Q: np.ndarray, x: int, y: int) -> float | None:
    if y < 0 or y >= disparity.shape[0] or x < 0 or x >= disparity.shape[1]:
        return None
    if disparity[y, x] <= 0:
        return None
    points = cv2.reprojectImageTo3D(disparity, Q)
    z = float(points[y, x, 2])
    if not np.isfinite(z) or z <= 0:
        return None
    return z
