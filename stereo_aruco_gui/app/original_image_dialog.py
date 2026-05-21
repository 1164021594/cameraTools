from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent
from PySide6.QtWidgets import QDialog, QLabel, QScrollArea, QVBoxLayout, QWidget


def _frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class ZoomImageLabel(QLabel):
    point_selected = Signal(int, int)

    def __init__(self, frame: np.ndarray) -> None:
        super().__init__()
        self._frame = frame
        self._base_pixmap = _frame_to_pixmap(frame)
        self._scale = 1.0
        self._last_point: tuple[int, int] | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.setMouseTracking(True)
        self._update_pixmap()

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self._scale = min(8.0, max(0.25, self._scale * (1.25 if delta > 0 else 0.8)))
        self._update_pixmap()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            point = self._map_to_source(event.position().toPoint())
            if point is not None:
                self._last_point = point
                self.point_selected.emit(point[0], point[1])

    def _map_to_source(self, pos: QPoint) -> tuple[int, int] | None:
        width = max(int(self._base_pixmap.width() * self._scale), 1)
        height = max(int(self._base_pixmap.height() * self._scale), 1)
        if pos.x() < 0 or pos.y() < 0 or pos.x() >= width or pos.y() >= height:
            return None
        source_w = self._frame.shape[1]
        source_h = self._frame.shape[0]
        return int(pos.x() * source_w / width), int(pos.y() * source_h / height)

    def _update_pixmap(self) -> None:
        width = max(int(self._base_pixmap.width() * self._scale), 1)
        height = max(int(self._base_pixmap.height() * self._scale), 1)
        scaled = self._base_pixmap.scaled(width, height, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        self.setPixmap(scaled)
        self.resize(scaled.size())


class OriginalImageDialog(QDialog):
    point_selected = Signal(int, int)

    def __init__(self, title: str, frame: np.ndarray, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(1000, 800)
        layout = QVBoxLayout(self)
        self.position_label = QLabel("Point: --")
        layout.addWidget(self.position_label)
        self.image_label = ZoomImageLabel(frame)
        self.image_label.point_selected.connect(self._handle_point_selected)
        scroll = QScrollArea()
        scroll.setWidgetResizable(False)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll)

    def _handle_point_selected(self, x: int, y: int) -> None:
        self.position_label.setText(f"Point: {x}, {y}")
        self.point_selected.emit(x, y)
