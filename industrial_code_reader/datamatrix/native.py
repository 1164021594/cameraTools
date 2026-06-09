from __future__ import annotations

import numpy as np

from industrial_code_reader.core.types import CodeResult, DecodeOptions


class NativeDataMatrixDecoder:
    symbology = "DataMatrix"

    def decode(self, image: np.ndarray, options: DecodeOptions) -> list[CodeResult]:
        if not options.return_failures:
            return []
        return [
            CodeResult(
                text="",
                symbology="DataMatrix",
                roi_id=options.roi_id,
                confidence=0.0,
                quality={"backend": "native", "stage": "skeleton", "image_shape": tuple(int(value) for value in image.shape[:2])},
                preprocessing="native",
                failure_reason="Native DataMatrix decoder is not complete yet",
            )
        ]
