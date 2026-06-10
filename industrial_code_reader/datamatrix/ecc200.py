from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


SYMBOL_CODEWORDS = {
    10: (3, 5),
    12: (5, 7),
    14: (8, 10),
    16: (12, 12),
    18: (18, 14),
    20: (22, 18),
    22: (30, 20),
    24: (36, 24),
    26: (44, 28),
}
RS_PRIMITIVE_POLY = 0x12D
RS_GENERATOR_BASE = 1


@dataclass(frozen=True)
class Ecc200Decode:
    text: str
    raw_codewords: list[int]
    data_codewords: list[int]
    ecc_codewords: int
    errors_corrected: int


def decode_ecc200_modules(modules: np.ndarray) -> Ecc200Decode | None:
    codewords = extract_codewords(modules)
    if not codewords:
        return None
    corrected = correct_codewords(modules.shape[0], codewords)
    if corrected is None:
        return None
    data_codewords, errors_corrected, ecc_codewords = corrected
    text = decode_ascii_payload(data_codewords)
    if not text:
        return None
    return Ecc200Decode(
        text=text,
        raw_codewords=codewords,
        data_codewords=data_codewords,
        ecc_codewords=ecc_codewords,
        errors_corrected=errors_corrected,
    )


def extract_codewords(modules: np.ndarray) -> list[int]:
    if modules.shape[0] != modules.shape[1] or modules.shape[0] < 10:
        return []
    data = modules[1:-1, 1:-1]
    placement, count = placement_map(data.shape[0], data.shape[1])
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


def correct_codewords(symbol_size: int, codewords: list[int]) -> tuple[list[int], int, int] | None:
    capacity = SYMBOL_CODEWORDS.get(symbol_size)
    if capacity is None:
        return None
    data_count, ecc_count = capacity
    total_count = data_count + ecc_count
    if len(codewords) < total_count:
        return None
    received = codewords[:total_count]
    if _rs_is_valid(received, ecc_count):
        return received[:data_count], 0, ecc_count
    corrected = _rs_correct(received, ecc_count, max_errors=min(2, ecc_count // 2))
    if corrected is None:
        return None
    errors_corrected = sum(1 for before, after in zip(received, corrected) if before != after)
    return corrected[:data_count], errors_corrected, ecc_count


def decode_ascii_payload(codewords: list[int]) -> str:
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


def _build_gf_tables() -> tuple[list[int], list[int]]:
    exp = [0] * 512
    log = [0] * 256
    value = 1
    for index in range(255):
        exp[index] = value
        log[value] = index
        value <<= 1
        if value & 0x100:
            value ^= RS_PRIMITIVE_POLY
    for index in range(255, 512):
        exp[index] = exp[index - 255]
    return exp, log


GF_EXP, GF_LOG = _build_gf_tables()


def _rs_correct(received: list[int], ecc_count: int, max_errors: int) -> list[int] | None:
    syndromes = _rs_syndromes(received, ecc_count)
    if not any(syndromes):
        return received
    positions = range(len(received))
    for error_count in range(1, max_errors + 1):
        for error_positions in combinations(positions, error_count):
            magnitudes = _solve_error_magnitudes(received, syndromes, error_positions)
            if magnitudes is None or not any(magnitudes):
                continue
            corrected = received.copy()
            for position, magnitude in zip(error_positions, magnitudes):
                corrected[position] ^= magnitude
            if _rs_is_valid(corrected, ecc_count):
                return corrected
    return None


def _solve_error_magnitudes(received: list[int], syndromes: list[int], error_positions: tuple[int, ...]) -> list[int] | None:
    size = len(error_positions)
    codeword_count = len(received)
    matrix: list[list[int]] = []
    for row in range(size):
        syndrome_power = RS_GENERATOR_BASE + row
        matrix.append([_gf_pow(syndrome_power * (codeword_count - 1 - position)) for position in error_positions])
    return _gf_solve(matrix, syndromes[:size])


def _rs_is_valid(codewords: list[int], ecc_count: int) -> bool:
    return not any(_rs_syndromes(codewords, ecc_count))


def _rs_syndromes(codewords: list[int], ecc_count: int) -> list[int]:
    return [_rs_poly_eval(codewords, _gf_pow(RS_GENERATOR_BASE + index)) for index in range(ecc_count)]


def _rs_poly_eval(poly: list[int], x_value: int) -> int:
    result = 0
    for coefficient in poly:
        result = _gf_mul(result, x_value) ^ coefficient
    return result


def _gf_solve(matrix: list[list[int]], values: list[int]) -> list[int] | None:
    size = len(values)
    rows = [matrix[row][:] + [values[row]] for row in range(size)]
    for col in range(size):
        pivot = next((row for row in range(col, size) if rows[row][col] != 0), None)
        if pivot is None:
            return None
        if pivot != col:
            rows[col], rows[pivot] = rows[pivot], rows[col]
        pivot_value = rows[col][col]
        rows[col] = [_gf_div(value, pivot_value) for value in rows[col]]
        for row in range(size):
            if row == col or rows[row][col] == 0:
                continue
            factor = rows[row][col]
            rows[row] = [current ^ _gf_mul(factor, pivot_value) for current, pivot_value in zip(rows[row], rows[col])]
    return [rows[row][-1] for row in range(size)]


def _gf_pow(power: int) -> int:
    return GF_EXP[power % 255]


def _gf_mul(left: int, right: int) -> int:
    if left == 0 or right == 0:
        return 0
    return GF_EXP[GF_LOG[left] + GF_LOG[right]]


def _gf_div(left: int, right: int) -> int:
    if right == 0:
        raise ZeroDivisionError("GF division by zero")
    if left == 0:
        return 0
    return GF_EXP[(GF_LOG[left] - GF_LOG[right]) % 255]


def placement_map(rows: int, cols: int) -> tuple[dict[tuple[int, int], tuple[int, int]], int]:
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
