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
            finder_score = _finder_pattern_score(gray, roi)
            dark_ratio = _dark_ratio(gray, roi)
            if dark_ratio < 0.08 or dark_ratio > 0.92:
                continue
            roi = Roi(
                id=roi.id,
                x=roi.x,
                y=roi.y,
                width=roi.width,
                height=roi.height,
                quality={
                    "area": area,
                    "aspect": aspect,
                    "edge_density": density,
                    "module_texture": module_texture,
                    "finder_score": finder_score,
                    "dark_ratio": dark_ratio,
                },
            )
            score = area * (density + module_texture * 2.0) * (0.35 + finder_score * 6.0)
            candidates.append((score, roi))
        candidates.extend(_mser_saturation_candidates(image, gray, edges, self.min_area, self.max_area, self.padding))
        candidates = _merge_overlapping_candidates(candidates, width, height)
        candidates = _deduplicate_candidates(candidates)
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


def _finder_pattern_score(gray: np.ndarray, roi: Roi) -> float:
    binary = _threshold_patch(gray, roi)
    if binary.shape[0] < 10 or binary.shape[1] < 10:
        return 0.0
    dark = binary == 0
    border = max(2, round(min(binary.shape[:2]) * 0.12))
    left = float(np.mean(dark[:, :border]))
    bottom = float(np.mean(dark[-border:, :]))
    top = float(np.mean(dark[:border, :]))
    right = float(np.mean(dark[:, -border:]))
    solid_l = (left + bottom) / 2.0
    alternating_edges = 1.0 - abs(top - 0.5) * 2.0
    alternating_edges += 1.0 - abs(right - 0.5) * 2.0
    alternating_edges /= 2.0
    squareness = min(roi.width, roi.height) / max(roi.width, roi.height, 1)
    return max(0.0, solid_l) * max(0.0, alternating_edges) * float(squareness)


def _mser_saturation_candidates(
    image: np.ndarray,
    gray: np.ndarray,
    edges: np.ndarray,
    min_area: int,
    max_area: int,
    padding: int,
) -> list[tuple[float, Roi]]:
    if image.ndim < 3:
        return []
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    work, scale = _resize_for_mser(saturation)
    scaled_min_area = max(20, round(max(20, min_area // 4) * scale * scale))
    scaled_max_area = max(scaled_min_area + 1, round(max_area * scale * scale))
    mser = cv2.MSER_create(delta=5, min_area=scaled_min_area, max_area=scaled_max_area)
    _, boxes = mser.detectRegions(work)
    candidates: list[tuple[float, Roi]] = []
    height, width = gray.shape[:2]
    for x, y, w, h in boxes:
        if scale != 1.0:
            x = round(int(x) / scale)
            y = round(int(y) / scale)
            w = round(int(w) / scale)
            h = round(int(h) / scale)
        area = int(w) * int(h)
        if area < min_area or area > max_area:
            continue
        aspect = float(w) / max(float(h), 1.0)
        if aspect < 0.6 or aspect > 1.6:
            continue
        roi_padding = max(padding, round(max(int(w), int(h)) * 0.35))
        roi = _padded_roi(len(candidates) + 1, int(x), int(y), int(w), int(h), roi_padding, width, height)
        density = _edge_density(edges, roi)
        module_texture = _module_texture(gray, roi)
        if density < 0.025 or module_texture < 0.015:
            continue
        finder_score = _finder_pattern_score(gray, roi)
        dark_ratio = _dark_ratio(gray, roi)
        if dark_ratio < 0.05 or dark_ratio > 0.95:
            continue
        roi = Roi(
            id=roi.id,
            x=roi.x,
            y=roi.y,
            width=roi.width,
            height=roi.height,
            quality={
                "source": "mser_saturation",
                "area": area,
                "aspect": aspect,
                "edge_density": density,
                "module_texture": module_texture,
                "finder_score": finder_score,
                "dark_ratio": dark_ratio,
            },
        )
        score = area * (density + module_texture * 3.0) * (0.35 + finder_score * 7.0)
        candidates.append((score, roi))
    return candidates


def _resize_for_mser(saturation: np.ndarray, max_side: int = 1200) -> tuple[np.ndarray, float]:
    height, width = saturation.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return saturation, 1.0
    scale = max_side / float(longest)
    resized = cv2.resize(saturation, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, scale


def _deduplicate_candidates(candidates: list[tuple[float, Roi]]) -> list[tuple[float, Roi]]:
    deduped: list[tuple[float, Roi]] = []
    for score, roi in sorted(candidates, key=lambda item: item[0], reverse=True):
        if any(_roi_iou(roi, existing) > 0.65 for _, existing in deduped):
            continue
        deduped.append((score, Roi(id=len(deduped) + 1, x=roi.x, y=roi.y, width=roi.width, height=roi.height, quality=roi.quality)))
    return deduped


def _merge_overlapping_candidates(candidates: list[tuple[float, Roi]], image_width: int, image_height: int) -> list[tuple[float, Roi]]:
    merged: list[tuple[float, Roi]] = []
    for score, roi in sorted(candidates, key=lambda item: item[0], reverse=True):
        merged_into_existing = False
        for index, (existing_score, existing) in enumerate(merged):
            if _roi_iou(roi, existing) <= 0.35:
                continue
            left = max(0, min(roi.x, existing.x))
            top = max(0, min(roi.y, existing.y))
            right = min(image_width, max(roi.x + roi.width, existing.x + existing.width))
            bottom = min(image_height, max(roi.y + roi.height, existing.y + existing.height))
            quality = dict(existing.quality)
            quality["merged_candidates"] = int(quality.get("merged_candidates", 1)) + 1
            merged[index] = (
                existing_score + score * 0.25,
                Roi(id=existing.id, x=left, y=top, width=right - left, height=bottom - top, quality=quality),
            )
            merged_into_existing = True
            break
        if not merged_into_existing:
            merged.append((score, roi))
    return merged


def _roi_iou(first: Roi, second: Roi) -> float:
    left = max(first.x, second.x)
    top = max(first.y, second.y)
    right = min(first.x + first.width, second.x + second.width)
    bottom = min(first.y + first.height, second.y + second.height)
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection == 0:
        return 0.0
    first_area = first.width * first.height
    second_area = second.width * second.height
    return float(intersection) / float(first_area + second_area - intersection)
