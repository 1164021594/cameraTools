from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import cv2


BACKENDS: list[tuple[str, int]] = [
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
    ("DEFAULT", cv2.CAP_ANY),
]
DEFAULT_SCAN_BACKENDS: list[tuple[str, int]] = [("DSHOW", cv2.CAP_DSHOW)]


@dataclass(frozen=True)
class CameraProbeResult:
    index: int
    backend_name: str
    opened: bool
    read_ok: bool
    shape: tuple[int, ...] | None
    actual_width: float
    actual_height: float
    actual_fps: float


ProbeFn = Callable[[int, int, int, int, int], tuple[bool, tuple[int, ...] | None, float, float, float]]


def probe_camera(index: int, backend: int, width: int, height: int, fps: int) -> tuple[bool, tuple[int, ...] | None, float, float, float]:
    cap = cv2.VideoCapture(index, backend)
    if not cap.isOpened():
        cap.release()
        return False, None, 0.0, 0.0, 0.0
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    read_ok = False
    frame_shape = None
    for _ in range(10):
        read_ok, frame = cap.read()
        if read_ok and frame is not None:
            frame_shape = tuple(frame.shape)
            break
    actual_width = float(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = float(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return read_ok, frame_shape, actual_width, actual_height, actual_fps


def scan_camera_indices(
    indexes: list[int] | range = range(6),
    backends: list[tuple[str, int]] = DEFAULT_SCAN_BACKENDS,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    probe: ProbeFn = probe_camera,
) -> list[CameraProbeResult]:
    results: list[CameraProbeResult] = []
    for backend_name, backend in backends:
        for index in indexes:
            read_ok, shape, actual_width, actual_height, actual_fps = probe(index, backend, width, height, fps)
            results.append(
                CameraProbeResult(
                    index=index,
                    backend_name=backend_name,
                    opened=actual_width > 0 or read_ok,
                    read_ok=read_ok,
                    shape=shape,
                    actual_width=actual_width,
                    actual_height=actual_height,
                    actual_fps=actual_fps,
                )
            )
    return results
