from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


SUPPORTED_BARCODE_LABELS = (
    "Code 39",
    "Code 128",
    "Codabar",
    "EAN",
    "ITF25",
    "Code 93",
    "QR Code",
    "DataMatrix",
)


@dataclass(frozen=True)
class BarcodeDetection:
    text: str
    format: str
    points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None


def require_zxingcpp() -> Any:
    try:
        import zxingcpp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Barcode detection requires zxing-cpp. Run: python -m pip install -r requirements.txt") from exc
    return zxingcpp


def barcode_formats_for_labels(labels: list[str] | tuple[str, ...]) -> Any:
    zxingcpp = require_zxingcpp()
    if not labels or "All" in labels:
        return None
    mapped = []
    for label in labels:
        if label == "Code 39":
            mapped.append(zxingcpp.BarcodeFormat.Code39)
        elif label == "Code 128":
            mapped.append(zxingcpp.BarcodeFormat.Code128)
        elif label == "Codabar":
            mapped.append(zxingcpp.BarcodeFormat.Codabar)
        elif label == "EAN":
            mapped.extend((zxingcpp.BarcodeFormat.EAN8, zxingcpp.BarcodeFormat.EAN13))
        elif label == "ITF25":
            mapped.append(zxingcpp.BarcodeFormat.ITF)
        elif label == "Code 93":
            mapped.append(zxingcpp.BarcodeFormat.Code93)
        elif label == "QR Code":
            mapped.append(zxingcpp.BarcodeFormat.QRCode)
        elif label == "DataMatrix":
            mapped.append(zxingcpp.BarcodeFormat.DataMatrix)
    return tuple(mapped) if mapped else None


def decode_barcodes(frame: np.ndarray, enabled_labels: list[str] | tuple[str, ...]) -> list[BarcodeDetection]:
    zxingcpp = require_zxingcpp()
    formats = barcode_formats_for_labels(enabled_labels)
    barcodes = zxingcpp.read_barcodes(frame, formats=formats)
    detections: list[BarcodeDetection] = []
    for barcode in barcodes:
        if not getattr(barcode, "valid", True):
            continue
        text = str(getattr(barcode, "text", ""))
        if not text:
            continue
        detections.append(
            BarcodeDetection(
                text=text,
                format=str(getattr(barcode, "format", getattr(barcode, "symbology", "Unknown"))),
                points=_points_from_position(getattr(barcode, "position", None)),
            )
        )
    return detections


def draw_barcode_detections(image: np.ndarray, detections: list[BarcodeDetection]) -> np.ndarray:
    if not detections:
        return image
    output = image.copy()
    for detection in detections:
        if detection.points is None:
            continue
        points = np.asarray(detection.points, dtype=np.int32)
        cv2.polylines(output, [points], True, (0, 255, 0), 2, cv2.LINE_AA)
        x = int(points[:, 0].min())
        y = max(int(points[:, 1].min()) - 8, 14)
        cv2.putText(output, detection.text[:40], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    return output


class BarcodeConfirmation:
    def __init__(self, required_count: int = 3) -> None:
        self.required_count = max(1, int(required_count))
        self._last_key: tuple[str, str] | None = None
        self._last_detection: BarcodeDetection | None = None
        self._count = 0

    def update(self, detections: list[BarcodeDetection]) -> BarcodeDetection | None:
        if not detections:
            self._last_key = None
            self._last_detection = None
            self._count = 0
            return None
        detection = detections[0]
        key = (detection.text, detection.format)
        if key == self._last_key:
            self._count += 1
        else:
            self._last_key = key
            self._last_detection = detection
            self._count = 1
        self._last_detection = detection
        if self._count >= self.required_count:
            return detection
        return None

    def reset(self) -> None:
        self._last_key = None
        self._last_detection = None
        self._count = 0

    def status_text(self) -> str:
        if self._last_detection is None:
            return "No barcode"
        return f"{self._last_detection.text} ({self._last_detection.format}) x{self._count}"


def _points_from_position(position: Any) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if position is None:
        return None
    names = ("top_left", "top_right", "bottom_right", "bottom_left")
    points = []
    for name in names:
        point = getattr(position, name, None)
        if point is None:
            return None
        points.append((int(point.x), int(point.y)))
    return tuple(points)  # type: ignore[return-value]
