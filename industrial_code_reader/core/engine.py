from __future__ import annotations

import numpy as np

from industrial_code_reader.core.decoder import CodeDecoder
from industrial_code_reader.core.types import CodeResult, DecodeOptions, Roi


class CodeReaderEngine:
    def __init__(self, decoders: list[CodeDecoder]) -> None:
        self.decoders = decoders

    def decode(self, image: np.ndarray, options: DecodeOptions | None = None) -> list[CodeResult]:
        decode_options = options or DecodeOptions()
        if decode_options.rois:
            return self._decode_rois(image, decode_options)
        if decode_options.auto_rois and _wants_datamatrix(decode_options):
            auto_rois = _generate_datamatrix_rois(image)[: decode_options.max_rois]
            if not auto_rois:
                return []
            auto_options = DecodeOptions(
                symbologies=decode_options.symbologies,
                rois=auto_rois,
                max_results=decode_options.max_results,
                max_rois=decode_options.max_rois,
                enable_preprocessing=decode_options.enable_preprocessing,
                return_failures=decode_options.return_failures,
            )
            return self._decode_rois(image, auto_options)
        return self._decode_image(image, decode_options)

    def _decode_image(self, image: np.ndarray, decode_options: DecodeOptions) -> list[CodeResult]:
        wanted = {symbology.lower() for symbology in decode_options.symbologies}
        results: list[CodeResult] = []
        for decoder in self.decoders:
            if wanted and decoder.symbology.lower() not in wanted and "all" not in wanted:
                continue
            results.extend(decoder.decode(image, decode_options))
            if len(results) >= decode_options.max_results:
                return results[: decode_options.max_results]
        return results

    def _decode_rois(self, image: np.ndarray, decode_options: DecodeOptions) -> list[CodeResult]:
        results: list[CodeResult] = []
        for roi in decode_options.rois:
            clipped = _clip_roi(roi, image.shape[1], image.shape[0])
            if clipped is None:
                continue
            roi_image = image[clipped.y : clipped.y + clipped.height, clipped.x : clipped.x + clipped.width]
            roi_options = DecodeOptions(
                symbologies=decode_options.symbologies,
                roi_id=clipped.id,
                max_results=decode_options.max_results,
                max_rois=decode_options.max_rois,
                enable_preprocessing=decode_options.enable_preprocessing,
                return_failures=decode_options.return_failures,
            )
            roi_results = self._decode_image(roi_image, roi_options)
            if not roi_results and decode_options.return_failures:
                roi_results = [_failure_result(clipped)]
            for result in roi_results:
                results.append(_offset_result(result, clipped.x, clipped.y, clipped.id))
                if result.text and len([item for item in results if item.text]) >= decode_options.max_results:
                    return results[: decode_options.max_results]
        return results


def _clip_roi(roi: Roi, image_width: int, image_height: int) -> Roi | None:
    x = max(0, roi.x)
    y = max(0, roi.y)
    right = min(image_width, roi.x + roi.width)
    bottom = min(image_height, roi.y + roi.height)
    width = right - x
    height = bottom - y
    if width <= 0 or height <= 0:
        return None
    return Roi(id=roi.id, x=x, y=y, width=width, height=height, quality=roi.quality)


def _offset_result(result: CodeResult, x_offset: int, y_offset: int, roi_id: int | None = None) -> CodeResult:
    bbox = None
    if result.bbox is not None:
        x, y, width, height = result.bbox
        bbox = (x + x_offset, y + y_offset, width, height)
    points = None
    if result.points is not None:
        points = tuple((x + x_offset, y + y_offset) for x, y in result.points)
    return CodeResult(
        text=result.text,
        symbology=result.symbology,
        roi_id=roi_id if roi_id is not None else result.roi_id,
        bbox=bbox,
        points=points,
        confidence=result.confidence,
        quality=result.quality,
        preprocessing=result.preprocessing,
        failure_reason=result.failure_reason,
    )


def _failure_result(roi: Roi) -> CodeResult:
    return CodeResult(
        text="",
        symbology="DataMatrix",
        roi_id=roi.id,
        bbox=(0, 0, roi.width, roi.height),
        points=None,
        confidence=0.0,
        quality={"roi": roi.quality},
        preprocessing="none",
        failure_reason="No DataMatrix decoded in ROI",
    )


def _wants_datamatrix(options: DecodeOptions) -> bool:
    wanted = {symbology.lower() for symbology in options.symbologies}
    return not wanted or "all" in wanted or "datamatrix" in wanted


def _generate_datamatrix_rois(image: np.ndarray) -> tuple[Roi, ...]:
    from industrial_code_reader.datamatrix.roi import DataMatrixRoiGenerator

    return DataMatrixRoiGenerator().generate(image)
