from __future__ import annotations

import cv2
import numpy as np

from industrial_code_reader.core.preprocess import to_gray
from industrial_code_reader.core.types import Roi


class DataMatrixRoiGenerator:
    def __init__(self, min_area: int = 120, max_area: int = 20000, padding: int = 8, max_rois: int = 32) -> None:
        self.min_area = min_area
        self.max_area = max_area
        self.padding = padding
        self.max_rois = max_rois

    def generate(self, image: np.ndarray) -> tuple[Roi, ...]:
        gray = to_gray(image)
        edges = cv2.Canny(gray, 50, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        dense = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dense, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[tuple[float, Roi]] = []
        height, width = gray.shape[:2]
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area < self.min_area or area > self.max_area:
                continue
            aspect = w / max(h, 1)
            if aspect < 0.45 or aspect > 2.2:
                continue
            roi = _padded_roi(len(candidates) + 1, x, y, w, h, self.padding, width, height)
            density = _edge_density(edges, roi)
            if density < 0.03:
                continue
            module_texture = _module_texture(gray, roi)
            if module_texture < 0.08:
                continue
            dark_ratio = _dark_ratio(gray, roi)
            if dark_ratio < 0.08 or dark_ratio > 0.92:
                continue
            roi = Roi(
                id=roi.id,
                x=roi.x,
                y=roi.y,
                width=roi.width,
                height=roi.height,
                quality={"area": area, "aspect": aspect, "edge_density": density, "module_texture": module_texture, "dark_ratio": dark_ratio},
            )
            score = area * (density + module_texture * 2.0)
            candidates.append((score, roi))
        candidates.sort(key=lambda item: item[0], reverse=True)
        return tuple(roi for _, roi in candidates[: self.max_rois])


def _padded_roi(id_: int, x: int, y: int, width: int, height: int, padding: int, image_width: int, image_height: int) -> Roi:
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image_width, x + width + padding)
    bottom = min(image_height, y + height + padding)
    return Roi(id=id_, x=left, y=top, width=right - left, height=bottom - top)


def _edge_density(edges: np.ndarray, roi: Roi) -> float:
    patch = edges[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    if patch.size == 0:
        return 0.0
    return float(np.count_nonzero(patch)) / float(patch.size)


def _threshold_patch(gray: np.ndarray, roi: Roi) -> np.ndarray:
    patch = gray[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width]
    if patch.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def _module_texture(gray: np.ndarray, roi: Roi) -> float:
    binary = _threshold_patch(gray, roi)
    if binary.shape[0] < 2 or binary.shape[1] < 2:
        return 0.0
    horizontal = np.mean(binary[:, 1:] != binary[:, :-1])
    vertical = np.mean(binary[1:, :] != binary[:-1, :])
    return float((horizontal + vertical) / 2.0)


def _dark_ratio(gray: np.ndarray, roi: Roi) -> float:
    binary = _threshold_patch(gray, roi)
    return float(np.mean(binary == 0))
