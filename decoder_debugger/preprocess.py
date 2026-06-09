from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Callable

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    view: str = "Original"
    channel: str = "Original"
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    blur_mode: str = "None"
    blur_kernel: int = 3
    sharpen_enabled: bool = False
    sharpen_strength: float = 1.5
    sharpen_radius: float = 1.0
    threshold_mode: str = "None"
    adaptive_block_size: int = 21
    adaptive_c: int = 4
    manual_threshold: int = 128
    threshold_invert: bool = False
    morphology: str = "None"
    morphology_kernel: int = 3
    morphology_iterations: int = 1
    scale_factor: float = 1.0
    quiet_zone_padding: int = 0


@dataclass(frozen=True)
class PreprocessResult:
    output: np.ndarray
    stages: dict[str, np.ndarray]
    timings_ms: dict[str, float]
    config: PreprocessConfig


def apply_preprocess(image: np.ndarray, config: PreprocessConfig) -> PreprocessResult:
    stages: dict[str, np.ndarray] = {"Original": image.copy()}
    timings: dict[str, float] = {}
    current = _timed("Channel", timings, lambda: _select_channel(image, config.channel))
    stages[_stage_name_for_channel(config.channel)] = current.copy()

    if config.clahe_enabled:
        current = _timed("CLAHE", timings, lambda: _apply_clahe(current, config))
        stages["CLAHE"] = current.copy()
    if config.blur_mode != "None":
        current = _timed("Blur", timings, lambda: _apply_blur(current, config))
        stages["Blur"] = current.copy()
    if config.sharpen_enabled:
        current = _timed("Sharpen", timings, lambda: _apply_sharpen(current, config))
        stages["Sharpen"] = current.copy()
    if config.threshold_mode != "None":
        current = _timed("Threshold", timings, lambda: _apply_threshold(current, config))
        stages["Threshold"] = current.copy()
    if config.morphology != "None":
        current = _timed("Morphology", timings, lambda: _apply_morphology(current, config))
        stages["Morphology"] = current.copy()
    if config.scale_factor != 1.0:
        current = _timed(
            "Scale",
            timings,
            lambda: cv2.resize(current, None, fx=config.scale_factor, fy=config.scale_factor, interpolation=cv2.INTER_CUBIC),
        )
        stages["Scale"] = current.copy()
    if config.quiet_zone_padding > 0:
        pad = config.quiet_zone_padding
        current = _timed("Padding", timings, lambda: cv2.copyMakeBorder(current, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255))
        stages["Padding"] = current.copy()

    return PreprocessResult(output=current, stages=stages, timings_ms=timings, config=config)


def config_to_json(config: PreprocessConfig) -> str:
    return json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True)


def config_from_json(text: str) -> PreprocessConfig:
    return PreprocessConfig(**json.loads(text))


def _timed(name: str, timings: dict[str, float], func: Callable[[], np.ndarray]) -> np.ndarray:
    start = perf_counter()
    result = func()
    timings[name] = (perf_counter() - start) * 1000.0
    return result


def _stage_name_for_channel(channel: str) -> str:
    return "Original" if channel == "Original" else channel


def _select_channel(image: np.ndarray, channel: str) -> np.ndarray:
    if channel == "Original":
        return image.copy()
    if channel == "Gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if image.ndim == 2:
        return image.copy()
    if channel in {"B", "G", "R"}:
        return image[:, :, {"B": 0, "G": 1, "R": 2}[channel]].copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if channel == "HSV Saturation":
        return hsv[:, :, 1]
    if channel == "HSV Value":
        return hsv[:, :, 2]
    return image.copy()


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _apply_clahe(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    gray = _to_gray(image)
    tile = max(1, int(config.clahe_tile_size))
    return cv2.createCLAHE(clipLimit=max(0.1, float(config.clahe_clip_limit)), tileGridSize=(tile, tile)).apply(gray)


def _apply_blur(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    kernel = _odd(config.blur_kernel)
    if config.blur_mode == "Gaussian":
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    if config.blur_mode == "Median":
        return cv2.medianBlur(image, kernel)
    return image


def _apply_sharpen(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), max(0.1, float(config.sharpen_radius)))
    return cv2.addWeighted(image, float(config.sharpen_strength), blurred, 1.0 - float(config.sharpen_strength), 0)


def _apply_threshold(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    gray = _to_gray(image)
    threshold_type = cv2.THRESH_BINARY_INV if config.threshold_invert else cv2.THRESH_BINARY
    if config.threshold_mode == "Otsu":
        _, binary = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
        return binary
    if config.threshold_mode == "Adaptive":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type,
            _odd(config.adaptive_block_size),
            int(config.adaptive_c),
        )
    if config.threshold_mode == "Manual":
        _, binary = cv2.threshold(gray, int(config.manual_threshold), 255, threshold_type)
        return binary
    return image


def _apply_morphology(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    kernel_size = max(1, int(config.morphology_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    op = {
        "Erode": cv2.MORPH_ERODE,
        "Dilate": cv2.MORPH_DILATE,
        "Open": cv2.MORPH_OPEN,
        "Close": cv2.MORPH_CLOSE,
    }.get(config.morphology)
    if op is None:
        return image
    return cv2.morphologyEx(image, op, kernel, iterations=max(1, int(config.morphology_iterations)))
