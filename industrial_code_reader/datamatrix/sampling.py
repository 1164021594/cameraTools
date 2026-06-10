from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from industrial_code_reader.core.preprocess import to_gray


SQUARE_SINGLE_REGION_SIZES = (10, 12, 14, 16, 18, 20, 22, 24, 26)


@dataclass(frozen=True)
class SampledSymbol:
    modules: np.ndarray
    symbol_size: int
    bbox: tuple[int, int, int, int]


def binary_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = to_gray(image)
    variants: list[tuple[str, np.ndarray]] = []
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    variants.append(("native_otsu", otsu))
    variants.append(("native_otsu_inverted", cv2.bitwise_not(otsu)))
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)
    variants.append(("native_adaptive", adaptive))
    variants.append(("native_adaptive_inverted", cv2.bitwise_not(adaptive)))
    return variants


def sample_symbol(binary: np.ndarray) -> SampledSymbol | None:
    dark = binary < 128
    ys, xs = np.where(dark)
    if xs.size == 0 or ys.size == 0:
        return None
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    width = right - left
    height = bottom - top
    if width < 8 or height < 8:
        return None
    if min(width, height) / max(width, height) < 0.75:
        return None
    crop = dark[top:bottom, left:right]
    candidates: list[SampledSymbol] = []
    for size in SQUARE_SINGLE_REGION_SIZES:
        modules = _sample_modules(crop, size)
        score = finder_score(modules)
        if score >= 0.82:
            candidates.append(SampledSymbol(modules=modules, symbol_size=size, bbox=(left, top, width, height)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: finder_score(item.modules))


def finder_score(modules: np.ndarray) -> float:
    size = modules.shape[0]
    left = float(np.mean(modules[:, 0]))
    bottom = float(np.mean(modules[-1, :]))
    top_expected = np.array([index % 2 == 0 for index in range(size)], dtype=bool)
    right_expected = np.array([index % 2 == 1 for index in range(size)], dtype=bool)
    top = float(np.mean(modules[0, :] == top_expected))
    right = float(np.mean(modules[:, -1] == right_expected))
    return (left + bottom + top + right) / 4.0


def _sample_modules(crop: np.ndarray, size: int) -> np.ndarray:
    height, width = crop.shape[:2]
    modules = np.zeros((size, size), dtype=bool)
    for row in range(size):
        y0 = round(row * height / size)
        y1 = round((row + 1) * height / size)
        for col in range(size):
            x0 = round(col * width / size)
            x1 = round((col + 1) * width / size)
            cell = crop[y0:y1, x0:x1]
            modules[row, col] = bool(np.mean(cell) >= 0.5) if cell.size else False
    return modules
