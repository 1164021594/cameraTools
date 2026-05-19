from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap
from PySide6.QtWidgets import QLabel


def frame_to_pixmap(frame: np.ndarray) -> QPixmap:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    height, width, channels = rgb.shape
    bytes_per_line = channels * width
    image = QImage(rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


class ImageView(QLabel):
    image_clicked = Signal(int, int)

    def __init__(self, title: str) -> None:
        super().__init__(title)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(480, 360)
        self.setStyleSheet("background: #1f2328; color: #d0d7de; border: 1px solid #444c56;")
        self.setScaledContents(False)
        self._frame_shape: tuple[int, int] | None = None
        self._pixmap: QPixmap | None = None

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame_shape = (frame.shape[1], frame.shape[0])
        self._pixmap = frame_to_pixmap(frame)
        self._update_pixmap()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._update_pixmap()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._frame_shape is None or self._pixmap is None:
            return
        point = self._label_to_image(event.position().toPoint())
        if point is not None:
            self.image_clicked.emit(point[0], point[1])

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        super().setPixmap(scaled)

    def _label_to_image(self, pos: QPoint) -> tuple[int, int] | None:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull() or self._frame_shape is None:
            return None
        x_offset = (self.width() - pixmap.width()) // 2
        y_offset = (self.height() - pixmap.height()) // 2
        x = pos.x() - x_offset
        y = pos.y() - y_offset
        if x < 0 or y < 0 or x >= pixmap.width() or y >= pixmap.height():
            return None
        image_w, image_h = self._frame_shape
        return int(x * image_w / pixmap.width()), int(y * image_h / pixmap.height())
