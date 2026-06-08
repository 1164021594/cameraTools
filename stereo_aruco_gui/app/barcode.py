from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from industrial_code_reader import CodeReaderEngine, DataMatrixDecoder, DecodeOptions
from industrial_code_reader.core.types import CodeResult


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

BARCODE_ROI_COLORS = (
    (0, 255, 0),
    (255, 128, 0),
    (0, 192, 255),
    (255, 0, 255),
    (0, 128, 255),
    (192, 255, 0),
)

BARCODE_DECODER_OPTIONS = ("Auto", "Industrial DataMatrix", "ZXing", "HALCON")


@dataclass(frozen=True)
class BarcodeDetection:
    text: str
    format: str
    points: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None
    roi_id: int | None = None
    bbox: tuple[int, int, int, int] | None = None
    preprocessing: str = "original"
    failure_reason: str | None = None
    quality: dict[str, Any] | None = None


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


def decode_barcodes(frame: np.ndarray, enabled_labels: list[str] | tuple[str, ...], decoder_name: str = "Auto") -> list[BarcodeDetection]:
    if decoder_name == "HALCON":
        return [
            BarcodeDetection(
                text="",
                format="HALCON",
                points=None,
                failure_reason="HALCON decoder is not configured",
                quality={"decoder": "HALCON"},
            )
        ]
    if decoder_name == "Industrial DataMatrix":
        return _decode_industrial_datamatrix(frame)
    if decoder_name == "ZXing":
        return _decode_zxing(frame, enabled_labels)
    industrial_results: list[BarcodeDetection] = []
    if _should_use_industrial_datamatrix(enabled_labels):
        industrial_results = _decode_industrial_datamatrix(frame)
        if enabled_labels and "All" not in enabled_labels and set(enabled_labels) == {"DataMatrix"}:
            return industrial_results
    detections = _decode_zxing(frame, enabled_labels)
    return _merge_detections(industrial_results, detections)


def _decode_industrial_datamatrix(frame: np.ndarray) -> list[BarcodeDetection]:
    return [
        _detection_from_code_result(result)
        for result in _datamatrix_engine().decode(frame, DecodeOptions(symbologies=("DataMatrix",), auto_rois=True, return_failures=True))
    ]


def _decode_zxing(frame: np.ndarray, enabled_labels: list[str] | tuple[str, ...]) -> list[BarcodeDetection]:
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


def _datamatrix_engine() -> CodeReaderEngine:
    return CodeReaderEngine(decoders=[DataMatrixDecoder()])


def _should_use_industrial_datamatrix(enabled_labels: list[str] | tuple[str, ...]) -> bool:
    return not enabled_labels or "All" in enabled_labels or "DataMatrix" in enabled_labels


def _detection_from_code_result(result: CodeResult) -> BarcodeDetection:
    points = result.points if result.points is None else tuple((int(x), int(y)) for x, y in result.points)
    return BarcodeDetection(
        text=result.text,
        format=result.symbology,
        points=points,  # type: ignore[arg-type]
        roi_id=result.roi_id,
        bbox=result.bbox,
        preprocessing=result.preprocessing,
        failure_reason=result.failure_reason,
        quality=result.quality,
    )


def _merge_detections(primary: list[BarcodeDetection], secondary: list[BarcodeDetection]) -> list[BarcodeDetection]:
    merged: list[BarcodeDetection] = []
    seen: set[tuple[str, str]] = set()
    for detection in [*primary, *secondary]:
        if detection.failure_reason is not None and not detection.text:
            merged.append(detection)
            continue
        key = (detection.text, detection.format)
        if key in seen:
            continue
        seen.add(key)
        merged.append(detection)
    return merged


def draw_barcode_detections(image: np.ndarray, detections: list[BarcodeDetection]) -> np.ndarray:
    if not detections:
        return image
    output = image.copy()
    for index, detection in enumerate(detections, start=1):
        color = (0, 0, 255) if detection.failure_reason else BARCODE_ROI_COLORS[(index - 1) % len(BARCODE_ROI_COLORS)]
        if detection.points is None and detection.bbox is None:
            continue
        if detection.points is not None:
            points = np.asarray(detection.points, dtype=np.int32)
            cv2.polylines(output, [points], True, color, 2, cv2.LINE_AA)
            x = int(points[:, 0].min())
            y = max(int(points[:, 1].min()) - 8, 14)
        else:
            x, y, width, height = detection.bbox or (0, 0, 0, 0)
            cv2.rectangle(output, (x, y), (x + width, y + height), color, 2, cv2.LINE_AA)
            y = max(y - 8, 14)
        roi_label = detection.roi_id if detection.roi_id is not None else index
        label_text = "decode failed" if detection.failure_reason else detection.text
        label = f"ROI {roi_label} | {detection.format} | {label_text}"[:80]
        cv2.putText(output, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
    return output


def barcode_roi_summary(index: int, detection: BarcodeDetection) -> str:
    roi_label = detection.roi_id if detection.roi_id is not None else index
    if detection.failure_reason is not None:
        bbox = _bbox_text(detection)
        metrics = _quality_text(detection)
        return f"ROI {roi_label} | {detection.format} | decode failed: {detection.failure_reason} | {bbox} | preprocessing={detection.preprocessing}{metrics}"
    if detection.points is None:
        return f"ROI {roi_label} | {detection.format} | {detection.text} | position unavailable | preprocessing={detection.preprocessing}"
    xs = [point[0] for point in detection.points]
    ys = [point[1] for point in detection.points]
    bbox = f"bbox x={min(xs)}, y={min(ys)}, w={max(xs) - min(xs)}, h={max(ys) - min(ys)}"
    points = "points=" + ", ".join(f"({x},{y})" for x, y in detection.points)
    return f"ROI {roi_label} | {detection.format} | {detection.text} | {bbox} | {points} | preprocessing={detection.preprocessing}"


def _bbox_text(detection: BarcodeDetection) -> str:
    if detection.bbox is not None:
        x, y, width, height = detection.bbox
        return f"bbox x={x}, y={y}, w={width}, h={height}"
    return "bbox unavailable"


def _quality_text(detection: BarcodeDetection) -> str:
    roi_quality = (detection.quality or {}).get("roi", {})
    if not roi_quality:
        return ""
    parts = []
    for key in ("edge_density", "area", "aspect"):
        if key not in roi_quality:
            continue
        value = roi_quality[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.3f}")
        else:
            parts.append(f"{key}={value}")
    return " | " + ", ".join(parts) if parts else ""


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
