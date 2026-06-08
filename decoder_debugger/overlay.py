from __future__ import annotations

import cv2
import numpy as np

from industrial_code_reader.core.types import CodeResult, Roi


def draw_debug_overlay(image: np.ndarray, rois: tuple[Roi, ...], results: list[CodeResult]) -> np.ndarray:
    output = image.copy()
    result_by_roi = {result.roi_id: result for result in results}
    for roi in rois:
        result = result_by_roi.get(roi.id)
        color = (0, 0, 255) if result is not None and result.failure_reason else (0, 255, 0)
        cv2.rectangle(output, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), color, 2, cv2.LINE_AA)
        label = _roi_label(roi, result)
        y = max(roi.y - 8, 14)
        cv2.putText(output, label[:80], (roi.x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return output


def _roi_label(roi: Roi, result: CodeResult | None) -> str:
    if result is None:
        return f"ROI {roi.id}"
    if result.failure_reason:
        return f"ROI {roi.id}: failed"
    return f"ROI {roi.id}: {result.text}"
