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


@dataclass(frozen=True)
class DisparityConfig:
    num_disparities: int = 192
    block_size: int = 7
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    disp12_max_diff: int = 1

    def normalized(self) -> "DisparityConfig":
        num_disparities = max(16, int(round(self.num_disparities / 16)) * 16)
        block_size = max(3, int(self.block_size))
        if block_size % 2 == 0:
            block_size += 1
        return DisparityConfig(
            num_disparities=num_disparities,
            block_size=block_size,
            uniqueness_ratio=max(0, int(self.uniqueness_ratio)),
            speckle_window_size=max(0, int(self.speckle_window_size)),
            speckle_range=max(0, int(self.speckle_range)),
            disp12_max_diff=int(self.disp12_max_diff),
        )


def draw_horizontal_guides(image: np.ndarray, step: int = 80) -> np.ndarray:
    output = image.copy()
    height = output.shape[0]
    for y in range(step, height, step):
        cv2.line(output, (0, y), (output.shape[1], y), (0, 255, 255), 1)
    return output


def draw_depth_roi(image: np.ndarray, x_offset: int) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    x_offset = max(0, min(int(x_offset), width - 1))
    cv2.rectangle(output, (x_offset, 0), (width - 1, height - 1), (0, 255, 0), 2)
    return output


def compute_disparity(left_rect: np.ndarray, right_rect: np.ndarray, config: DisparityConfig | None = None) -> np.ndarray:
    config = (config or DisparityConfig()).normalized()
    left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY) if left_rect.ndim == 3 else left_rect
    right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY) if right_rect.ndim == 3 else right_rect
    matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=config.num_disparities,
        blockSize=config.block_size,
        P1=8 * config.block_size * config.block_size,
        P2=32 * config.block_size * config.block_size,
        uniquenessRatio=config.uniqueness_ratio,
        speckleWindowSize=config.speckle_window_size,
        speckleRange=config.speckle_range,
        disp12MaxDiff=config.disp12_max_diff,
    )
    return matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0


def disparity_preview(disparity: np.ndarray) -> np.ndarray:
    valid = disparity > 0
    output = np.zeros(disparity.shape, dtype=np.uint8)
    if np.any(valid):
        clipped = np.clip(disparity, 0, np.percentile(disparity[valid], 95))
        output = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(output, cv2.COLORMAP_TURBO)


def filtered_disparity_preview(disparity: np.ndarray) -> np.ndarray:
    filtered = disparity.copy()
    valid = filtered > 0
    if np.any(valid):
        median = cv2.medianBlur(np.where(valid, filtered, 0).astype(np.float32), 5)
        filtered = np.where(valid, filtered, median)
        filtered = cv2.bilateralFilter(filtered.astype(np.float32), 5, 16, 16)
    return disparity_preview(filtered)


def distance_at(disparity: np.ndarray, Q: np.ndarray, x: int, y: int, window_size: int = 1) -> float | None:
    if y < 0 or y >= disparity.shape[0] or x < 0 or x >= disparity.shape[1]:
        return None
    sample_disparity = float(disparity[y, x])
    if sample_disparity <= 0 and window_size > 1:
        radius = max(0, window_size // 2)
        y0 = max(0, y - radius)
        y1 = min(disparity.shape[0], y + radius + 1)
        x0 = max(0, x - radius)
        x1 = min(disparity.shape[1], x + radius + 1)
        values = disparity[y0:y1, x0:x1]
        valid = values[values > 0]
        if valid.size:
            sample_disparity = float(np.median(valid))
    if sample_disparity <= 0:
        return None
    sample = np.zeros_like(disparity, dtype=np.float32)
    sample[y, x] = sample_disparity
    points = cv2.reprojectImageTo3D(sample, Q)
    z = float(points[y, x, 2])
    if not np.isfinite(z) or z <= 0:
        return None
    return z
