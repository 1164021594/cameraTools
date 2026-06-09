from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from industrial_code_reader.core.preprocess import to_gray
from industrial_code_reader.core.types import CodeResult, DecodeOptions


_SQUARE_SINGLE_REGION_SIZES = (10, 12, 14, 16, 18, 20, 22, 24, 26)


@dataclass(frozen=True)
class SampledSymbol:
    modules: np.ndarray
    symbol_size: int
    bbox: tuple[int, int, int, int]


class NativeDataMatrixDecoder:
    symbology = "DataMatrix"

    def decode(self, image: np.ndarray, options: DecodeOptions) -> list[CodeResult]:
        failures: list[str] = []
        for binary_name, binary in _binary_variants(image):
            sampled = _sample_symbol(binary)
            if sampled is None:
                failures.append(f"{binary_name}: no stable square symbol")
                continue
            codewords = _extract_codewords(sampled.modules)
            if not codewords:
                failures.append(f"{binary_name}: no codewords")
                continue
            text = _decode_ascii_payload(codewords)
            if not text:
                failures.append(f"{binary_name}: payload decode failed")
                continue
            return [
                CodeResult(
                    text=text,
                    symbology="DataMatrix",
                    roi_id=options.roi_id,
                    bbox=sampled.bbox,
                    points=_points_from_bbox(sampled.bbox),
                    confidence=0.55,
                    quality={"backend": "native", "symbol_size": sampled.symbol_size, "codewords": len(codewords)},
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
                quality={"backend": "native", "stage": "decode", "failures": failures[:8], "image_shape": tuple(int(value) for value in image.shape[:2])},
                preprocessing="native",
                failure_reason="Native DataMatrix decoder is not complete yet",
            )
        ]


def _binary_variants(image: np.ndarray) -> list[tuple[str, np.ndarray]]:
    gray = to_gray(image)
    variants: list[tuple[str, np.ndarray]] = []
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    variants.append(("native_otsu", otsu))
    variants.append(("native_otsu_inverted", cv2.bitwise_not(otsu)))
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 4)
    variants.append(("native_adaptive", adaptive))
    variants.append(("native_adaptive_inverted", cv2.bitwise_not(adaptive)))
    return variants


def _sample_symbol(binary: np.ndarray) -> SampledSymbol | None:
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
    for size in _SQUARE_SINGLE_REGION_SIZES:
        modules = _sample_modules(crop, size)
        score = _finder_score(modules)
        if score >= 0.82:
            candidates.append(SampledSymbol(modules=modules, symbol_size=size, bbox=(left, top, width, height)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: _finder_score(item.modules))


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


def _finder_score(modules: np.ndarray) -> float:
    size = modules.shape[0]
    left = float(np.mean(modules[:, 0]))
    bottom = float(np.mean(modules[-1, :]))
    top_expected = np.array([index % 2 == 0 for index in range(size)], dtype=bool)
    right_expected = np.array([index % 2 == 1 for index in range(size)], dtype=bool)
    top = float(np.mean(modules[0, :] == top_expected))
    right = float(np.mean(modules[:, -1] == right_expected))
    return (left + bottom + top + right) / 4.0


def _extract_codewords(modules: np.ndarray) -> list[int]:
    if modules.shape[0] != modules.shape[1] or modules.shape[0] < 10:
        return []
    data = modules[1:-1, 1:-1]
    placement, count = _placement(data.shape[0], data.shape[1])
    codewords: list[int] = []
    for codeword_index in range(1, count + 1):
        value = 0
        for bit_index in range(1, 9):
            position = placement.get((codeword_index, bit_index))
            if position is None:
                continue
            row, col = position
            if data[row, col]:
                value |= 1 << (8 - bit_index)
        codewords.append(value)
    return codewords


def _placement(rows: int, cols: int) -> tuple[dict[tuple[int, int], tuple[int, int]], int]:
    cell_to_bit: dict[tuple[int, int], tuple[int, int]] = {}
    bit_to_cell: dict[tuple[int, int], tuple[int, int]] = {}

    def module(row: int, col: int, codeword: int, bit: int) -> None:
        if row < 0:
            row += rows
            col += 4 - ((rows + 4) % 8)
        if col < 0:
            col += cols
            row += 4 - ((cols + 4) % 8)
        cell_to_bit[(row, col)] = (codeword, bit)
        bit_to_cell[(codeword, bit)] = (row, col)

    def utah(row: int, col: int, codeword: int) -> None:
        module(row - 2, col - 2, codeword, 1)
        module(row - 2, col - 1, codeword, 2)
        module(row - 1, col - 2, codeword, 3)
        module(row - 1, col - 1, codeword, 4)
        module(row - 1, col, codeword, 5)
        module(row, col - 2, codeword, 6)
        module(row, col - 1, codeword, 7)
        module(row, col, codeword, 8)

    def corner1(codeword: int) -> None:
        module(rows - 1, 0, codeword, 1)
        module(rows - 1, 1, codeword, 2)
        module(rows - 1, 2, codeword, 3)
        module(0, cols - 2, codeword, 4)
        module(0, cols - 1, codeword, 5)
        module(1, cols - 1, codeword, 6)
        module(2, cols - 1, codeword, 7)
        module(3, cols - 1, codeword, 8)

    def corner2(codeword: int) -> None:
        module(rows - 3, 0, codeword, 1)
        module(rows - 2, 0, codeword, 2)
        module(rows - 1, 0, codeword, 3)
        module(0, cols - 4, codeword, 4)
        module(0, cols - 3, codeword, 5)
        module(0, cols - 2, codeword, 6)
        module(0, cols - 1, codeword, 7)
        module(1, cols - 1, codeword, 8)

    def corner3(codeword: int) -> None:
        module(rows - 3, 0, codeword, 1)
        module(rows - 2, 0, codeword, 2)
        module(rows - 1, 0, codeword, 3)
        module(0, cols - 2, codeword, 4)
        module(0, cols - 1, codeword, 5)
        module(1, cols - 1, codeword, 6)
        module(2, cols - 1, codeword, 7)
        module(3, cols - 1, codeword, 8)

    def corner4(codeword: int) -> None:
        module(rows - 1, 0, codeword, 1)
        module(rows - 1, cols - 1, codeword, 2)
        module(0, cols - 3, codeword, 3)
        module(0, cols - 2, codeword, 4)
        module(0, cols - 1, codeword, 5)
        module(1, cols - 3, codeword, 6)
        module(1, cols - 2, codeword, 7)
        module(1, cols - 1, codeword, 8)

    row = 4
    col = 0
    codeword = 1
    while True:
        if row == rows and col == 0:
            corner1(codeword)
            codeword += 1
        if row == rows - 2 and col == 0 and cols % 4 != 0:
            corner2(codeword)
            codeword += 1
        if row == rows - 2 and col == 0 and cols % 8 == 4:
            corner3(codeword)
            codeword += 1
        if row == rows + 4 and col == 2 and cols % 8 == 0:
            corner4(codeword)
            codeword += 1
        while row >= 0 and col < cols:
            if row < rows and col >= 0 and (row, col) not in cell_to_bit:
                utah(row, col, codeword)
                codeword += 1
            row -= 2
            col += 2
        row += 1
        col += 3
        while row < rows and col >= 0:
            if row >= 0 and col < cols and (row, col) not in cell_to_bit:
                utah(row, col, codeword)
                codeword += 1
            row += 2
            col -= 2
        row += 3
        col += 1
        if row >= rows and col >= cols:
            break
    return bit_to_cell, codeword - 1


def _decode_ascii_payload(codewords: list[int]) -> str:
    chars: list[str] = []
    index = 0
    while index < len(codewords):
        codeword = codewords[index]
        if codeword == 129:
            break
        if 1 <= codeword <= 128:
            chars.append(chr(codeword - 1))
        elif 130 <= codeword <= 229:
            chars.append(f"{codeword - 130:02d}")
        else:
            return ""
        index += 1
    return "".join(chars)


def _points_from_bbox(bbox: tuple[int, int, int, int]) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]]:
    x, y, width, height = bbox
    return ((x, y), (x + width, y), (x + width, y + height), (x, y + height))
