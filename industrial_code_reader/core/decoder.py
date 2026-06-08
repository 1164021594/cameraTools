from __future__ import annotations

from typing import Protocol

import numpy as np

from industrial_code_reader.core.types import CodeResult, DecodeOptions


class CodeDecoder(Protocol):
    symbology: str

    def decode(self, image: np.ndarray, options: DecodeOptions) -> list[CodeResult]:
        ...
