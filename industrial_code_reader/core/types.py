from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


Point = tuple[int, int]


@dataclass(frozen=True)
class Roi:
    id: int
    x: int
    y: int
    width: int
    height: int
    quality: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecodeOptions:
    symbologies: tuple[str, ...] = ("DataMatrix",)
    roi_id: int = 1
    rois: tuple[Roi, ...] = ()
    auto_rois: bool = False
    max_results: int = 16
    max_rois: int = 16
    enable_preprocessing: bool = True
    return_failures: bool = False


@dataclass(frozen=True)
class CodeResult:
    text: str
    symbology: str
    roi_id: int
    bbox: tuple[int, int, int, int] | None = None
    points: tuple[Point, ...] | None = None
    confidence: float = 0.0
    quality: dict[str, Any] = field(default_factory=dict)
    preprocessing: str = "original"
    failure_reason: str | None = None
