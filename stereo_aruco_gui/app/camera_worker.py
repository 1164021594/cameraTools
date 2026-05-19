from __future__ import annotations

import time
from threading import Lock

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from stereo_aruco_gui.app.camera_probe import BACKENDS
from stereo_aruco_gui.app.config import CameraConfig


BACKEND_MAP = dict(BACKENDS)
AUTO_BACKENDS = [("DSHOW", BACKEND_MAP["DSHOW"])]
PIXEL_FORMATS = ("MJPG", "YUYV", "DEFAULT")


def usable_frame(frame: np.ndarray | None) -> bool:
    if frame is None:
        return False
    return float(frame.mean()) > 3.0 or int(frame.max()) > 15


def preview_copy(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    if frame.shape[1] <= max_width:
        return frame.copy()
    scale = max_width / frame.shape[1]
    size = (max_width, int(frame.shape[0] * scale))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def should_decode_frame(now: float, last_decode: float, interval: float) -> bool:
    return now - last_decode >= interval


def read_stereo_pair(left_cap, right_cap) -> tuple[bool, np.ndarray | None, np.ndarray | None]:  # noqa: ANN001
    left_grabbed = left_cap.grab()
    right_grabbed = right_cap.grab()
    if left_grabbed and right_grabbed:
        ok_l, left = left_cap.retrieve()
        ok_r, right = right_cap.retrieve()
        if ok_l and ok_r:
            return True, left, right

    ok_l, left = left_cap.read()
    ok_r, right = right_cap.read()
    if ok_l and ok_r:
        return True, left, right
    return False, None, None


class SingleCameraWorker(QThread):
    status = Signal(str)

    def __init__(
        self,
        index: int,
        width: int,
        height: int,
        fps: int,
        label: str,
        backend: str = "AUTO",
        pixel_format: str = "MJPG",
    ) -> None:
        super().__init__()
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.label = label
        self.backend = backend
        self.pixel_format = pixel_format
        self._backend_start = 0
        self._active_backend_name = "none"
        self._running = False
        self._stop_requested = False
        self._capture: cv2.VideoCapture | None = None
        self.latest_frame: np.ndarray | None = None
        self.received_frame_count = 0
        self.last_frame_time = 0.0
        self.start_time = time.monotonic()
        self._lock = Lock()

    def run(self) -> None:
        self._running = True
        self._stop_requested = False
        backend_name = self._open()
        if self._capture is None:
            self.status.emit(f"{self.label}: failed to open camera {self.index}")
            return
        self.status.emit(f"{self.label}: opened index {self.index} with {backend_name}")
        consecutive_failures = 0
        last_notice = 0.0
        last_bad_notice = 0.0
        last_decode = 0.0
        decode_interval = 1.0 / max(1, min(self.fps, 30))

        while self._running:
            if self._capture is None:
                backend_name = self._open()
                if self._capture is None:
                    self.status.emit(f"{self.label}: waiting for camera")
                    time.sleep(1.0)
                    continue
                self.status.emit(f"{self.label}: reopened with {backend_name}")

            grabbed = self._capture.grab()
            now = time.monotonic()
            if grabbed and not should_decode_frame(now, last_decode, decode_interval):
                time.sleep(0.001)
                continue
            if grabbed:
                ok, frame = self._capture.retrieve()
            else:
                ok, frame = self._capture.read()
            if not ok:
                consecutive_failures += 1
                if now - last_notice > 1.0:
                    self.status.emit(f"{self.label}: failed to read frame ({consecutive_failures} consecutive)")
                    last_notice = now
                if consecutive_failures >= 10:
                    self.status.emit(f"{self.label}: reopening after repeated read failures")
                    self._mark_backend_failed(self._active_backend_name)
                    self._release()
                    time.sleep(0.5)
                    consecutive_failures = 0
                continue

            consecutive_failures = 0
            last_decode = now
            if not usable_frame(frame):
                now = time.monotonic()
                if now - last_bad_notice > 1.0:
                    self.status.emit(f"{self.label}: black frame")
                    last_bad_notice = now
                time.sleep(0.05)
                continue

            self.receive_frame(frame)

        self._release()
        reason = "stop requested" if self._stop_requested else "thread exited"
        self.status.emit(f"{self.label}: closed ({reason})")

    def stop(self) -> None:
        self._stop_requested = True
        self._running = False
        if self.isRunning():
            self.wait(2000)

    def receive_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            self.latest_frame = frame
            self.received_frame_count += 1
            self.last_frame_time = time.monotonic()

    def get_latest(self) -> np.ndarray | None:
        with self._lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def snapshot_stats(self) -> dict[str, float | int | str]:
        with self._lock:
            elapsed = max(time.monotonic() - self.start_time, 0.001)
            age = time.monotonic() - self.last_frame_time if self.last_frame_time else -1.0
            return {
                "label": self.label,
                "index": self.index,
                "frames": self.received_frame_count,
                "fps": self.received_frame_count / elapsed,
                "age": age,
            }

    def _open(self) -> str:
        backends = self._next_backends()
        for backend_name, backend in backends:
            cap = cv2.VideoCapture(self.index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            configure_capture(cap, self.width, self.height, self.fps, self.pixel_format)
            for _ in range(10):
                ok, frame = cap.read()
                if ok and usable_frame(frame):
                    self._capture = cap
                    self._active_backend_name = backend_name
                    return backend_name
                time.sleep(0.03)
            cap.release()
        self._capture = None
        self._active_backend_name = "none"
        return "none"

    def _next_backends(self) -> list[tuple[str, int]]:
        if self.backend != "AUTO":
            return [(self.backend, BACKEND_MAP[self.backend])]
        return AUTO_BACKENDS[self._backend_start :] + AUTO_BACKENDS[: self._backend_start]

    def _next_backend_names(self) -> list[str]:
        return [name for name, _ in self._next_backends()]

    def _mark_backend_failed(self, backend_name: str) -> None:
        if self.backend != "AUTO":
            return
        names = [name for name, _ in AUTO_BACKENDS]
        if backend_name in names:
            self._backend_start = (names.index(backend_name) + 1) % len(names)

    def _release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class CameraWorker(QThread):
    status = Signal(str)

    def __init__(self, config: CameraConfig) -> None:
        super().__init__()
        self.config = config
        self._running = False
        self._left: cv2.VideoCapture | None = None
        self._right: cv2.VideoCapture | None = None
        self.latest_left: np.ndarray | None = None
        self.latest_right: np.ndarray | None = None
        self.received_frame_count = 0
        self._lock = Lock()

    def run(self) -> None:
        self._running = True
        self._left, left_backend = self._open_capture(self.config.left_index)
        right_backend = "single"
        if not self.config.single_mode:
            self._right, right_backend = self._open_capture(self.config.right_index)
        if self._left is None or (not self.config.single_mode and self._right is None):
            self.status.emit("Failed to open camera")
            self._release()
            return

        if self.config.single_mode:
            self.status.emit(f"Single camera opened: left={left_backend}")
        else:
            self.status.emit(f"Cameras opened: left={left_backend}, right={right_backend}")
        last_bad_notice = 0.0
        last_read_notice = 0.0
        consecutive_failures = 0
        while self._running:
            if self._left is None or (not self.config.single_mode and self._right is None):
                self._release()
                self._left, left_backend = self._open_capture(self.config.left_index)
                if not self.config.single_mode:
                    self._right, right_backend = self._open_capture(self.config.right_index)
                if self._left is None or (not self.config.single_mode and self._right is None):
                    self.status.emit("Waiting for camera to become available")
                    time.sleep(1.0)
                    continue
                if self.config.single_mode:
                    self.status.emit(f"Single camera reopened: left={left_backend}")
                else:
                    self.status.emit(f"Cameras reopened: left={left_backend}, right={right_backend}")
            ok, left, right = self._read_current_frames()
            if not ok:
                consecutive_failures += 1
                now = time.monotonic()
                if now - last_read_notice > 1.0:
                    self.status.emit(f"Failed to read frame ({consecutive_failures} consecutive)")
                    last_read_notice = now
                if consecutive_failures >= 10:
                    self.status.emit("Reopening cameras after repeated read failures")
                    self._release()
                    time.sleep(0.5)
                    self._left, left_backend = self._open_capture(self.config.left_index)
                    if not self.config.single_mode:
                        self._right, right_backend = self._open_capture(self.config.right_index)
                    consecutive_failures = 0
                    if self._left is None or (not self.config.single_mode and self._right is None):
                        self.status.emit("Failed to reopen camera")
                        time.sleep(1.0)
                        continue
                    if self.config.single_mode:
                        self.status.emit(f"Single camera reopened: left={left_backend}")
                    else:
                        self.status.emit(f"Cameras reopened: left={left_backend}, right={right_backend}")
                time.sleep(0.1)
                continue
            consecutive_failures = 0
            if not usable_frame(left) or (right is not None and not usable_frame(right)):
                now = time.monotonic()
                if now - last_bad_notice > 1.0:
                    self.status.emit("Camera returned a black frame; check exposure, lens cover, USB bandwidth, or try 640x480")
                    last_bad_notice = now
                time.sleep(0.1)
                continue
            self.receive_frame_pair(left, right)
        self._release()
        self.status.emit("Cameras closed")

    def stop(self) -> None:
        self._running = False
        if self.isRunning():
            self.wait(2000)

    def get_latest(self) -> tuple[np.ndarray, np.ndarray | None] | None:
        with self._lock:
            if self.latest_left is None or (not self.config.single_mode and self.latest_right is None):
                return None
            return self.latest_left.copy(), None if self.latest_right is None else self.latest_right.copy()

    def receive_frame_pair(self, left: np.ndarray, right: np.ndarray | None) -> None:
        with self._lock:
            self.latest_left = left
            self.latest_right = right
            self.received_frame_count += 1

    def _read_current_frames(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        if self._left is None:
            return False, None, None
        if self.config.single_mode:
            ok, frame = self._left.read()
            return (ok, frame, None) if ok else (False, None, None)
        if self._right is None:
            return False, None, None
        return read_stereo_pair(self._left, self._right)

    def _open_capture(self, index: int) -> tuple[cv2.VideoCapture | None, str]:
        for backend_name, backend in BACKENDS:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue
            self._configure_capture(cap)
            for _ in range(10):
                ok, frame = cap.read()
                if ok and usable_frame(frame):
                    return cap, backend_name
                time.sleep(0.03)
            cap.release()
        return None, "none"

    def _configure_capture(self, cap: cv2.VideoCapture) -> None:
        configure_capture(cap, self.config.width, self.config.height, self.config.fps)

    def _release(self) -> None:
        for cap in (self._left, self._right):
            if cap is not None:
                cap.release()


def configure_capture(cap: cv2.VideoCapture, width: int, height: int, fps: int, pixel_format: str = "MJPG") -> None:
    if pixel_format != "DEFAULT":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*pixel_format))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
