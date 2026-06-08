from __future__ import annotations

from typing import Any

import numpy as np

from industrial_code_reader.core.preprocess import PreprocessedImage, datamatrix_preprocess_bank
from industrial_code_reader.core.types import CodeResult, DecodeOptions


def require_zxingcpp() -> Any:
    try:
        import zxingcpp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("DataMatrix decoding requires zxing-cpp. Run: python -m pip install -r requirements.txt") from exc
    return zxingcpp


class DataMatrixDecoder:
    symbology = "DataMatrix"

    def decode(self, image: np.ndarray, options: DecodeOptions) -> list[CodeResult]:
        zxingcpp = require_zxingcpp()
        formats = (zxingcpp.BarcodeFormat.DataMatrix,)
        variants = datamatrix_preprocess_bank(image) if options.enable_preprocessing else [PreprocessedImage("original", image, 1.0)]
        seen: set[tuple[str, str]] = set()
        results: list[CodeResult] = []
        for variant in variants:
            for barcode in zxingcpp.read_barcodes(variant.image, formats=formats):
                if not getattr(barcode, "valid", True):
                    continue
                text = str(getattr(barcode, "text", ""))
                if not text:
                    continue
                key = (text, str(getattr(barcode, "format", "DataMatrix")))
                if key in seen:
                    continue
                seen.add(key)
                points = _points_from_position(getattr(barcode, "position", None), variant.scale)
                results.append(
                    CodeResult(
                        text=text,
                        symbology="DataMatrix",
                        roi_id=options.roi_id,
                        bbox=_bbox_from_points(points),
                        points=points,
                        confidence=0.8,
                        quality={"backend": "zxing-cpp", "scale": variant.scale},
                        preprocessing=variant.name,
                    )
                )
                if len(results) >= options.max_results:
                    return results
        return results


def _points_from_position(position: Any, scale: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None:
    if position is None:
        return None
    names = ("top_left", "top_right", "bottom_right", "bottom_left")
    points = []
    for name in names:
        point = getattr(position, name, None)
        if point is None:
            return None
        points.append((round(int(point.x) / scale), round(int(point.y) / scale)))
    return tuple(points)  # type: ignore[return-value]


def _bbox_from_points(points: tuple[tuple[int, int], ...] | None) -> tuple[int, int, int, int] | None:
    if not points:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
