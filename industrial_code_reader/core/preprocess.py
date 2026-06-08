from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessedImage:
    name: str
    image: np.ndarray
    scale: float = 1.0


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.copy()
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def datamatrix_preprocess_bank(image: np.ndarray) -> list[PreprocessedImage]:
    gray = to_gray(image)
    variants: list[tuple[str, np.ndarray]] = [("gray", gray)]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    variants.append(("gray_clahe", clahe))

    blurred = cv2.GaussianBlur(gray, (0, 0), 1.0)
    sharpened = cv2.addWeighted(gray, 1.6, blurred, -0.6, 0)
    variants.append(("gray_sharpen", sharpened))

    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    variants.append(("gray_otsu", otsu))

    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)
    variants.append(("gray_adaptive", adaptive))

    output: list[PreprocessedImage] = []
    for name, variant in variants:
        for scale in (1.0, 2.0, 3.0, 4.0):
            if scale == 1.0:
                scaled = variant
            else:
                scaled = cv2.resize(variant, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            output.append(PreprocessedImage(f"{name}_{int(scale)}x", scaled, scale))
    return output
