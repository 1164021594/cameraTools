from __future__ import annotations

import numpy as np

from industrial_code_reader.core.types import CodeResult, DecodeOptions
from industrial_code_reader.datamatrix.ecc200 import decode_ecc200_modules
from industrial_code_reader.datamatrix.sampling import binary_variants, sample_symbol


class NativeDataMatrixDecoder:
    symbology = "DataMatrix"

    def decode(self, image: np.ndarray, options: DecodeOptions) -> list[CodeResult]:
        failures: list[str] = []
        for binary_name, binary in binary_variants(image):
            sampled = sample_symbol(binary)
            if sampled is None:
                failures.append(f"{binary_name}: no stable square symbol")
                continue
            decoded = decode_ecc200_modules(sampled.modules)
            if decoded is None:
                failures.append(f"{binary_name}: ecc200 decode failed")
                continue
            return [
                CodeResult(
                    text=decoded.text,
                    symbology="DataMatrix",
                    roi_id=options.roi_id,
                    bbox=sampled.bbox,
                    points=_points_from_bbox(sampled.bbox),
                    confidence=0.55,
                    quality={
                        "backend": "native",
                        "symbol_size": sampled.symbol_size,
                        "codewords": len(decoded.raw_codewords),
                        "data_codewords": len(decoded.data_codewords),
                        "ecc_codewords": decoded.ecc_codewords,
                        "errors_corrected": decoded.errors_corrected,
                    },
                    preprocessing=binary_name,
                )
            ]
        if not options.return_failures:
            return []
        return [
            CodeResult(
                text="",
                symbology="DataMatrix",
                roi_id=options.roi_id,
                confidence=0.0,
                quality={
                    "backend": "native",
                    "stage": "decode",
                    "failures": failures[:8],
                    "image_shape": tuple(int(value) for value in image.shape[:2]),
                },
                preprocessing="native",
                failure_reason="Native DataMatrix decoder is not complete yet",
            )
        ]


def _points_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    x, y, width, height = bbox
    return ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
