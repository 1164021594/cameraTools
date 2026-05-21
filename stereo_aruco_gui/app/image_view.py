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
        self.overlay_label = QLabel(self)
        self.overlay_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.overlay_label.setStyleSheet(
            "background: rgba(245, 245, 245, 120); color: #24292f; border: 0; border-radius: 3px; padding: 4px 6px;"
        )
        self.overlay_label.hide()
        self._frame_shape: tuple[int, int] | None = None
        self._source_size: tuple[int, int] | None = None
        self._source_rect: tuple[int, int, int, int] | None = None
        self._pixmap: QPixmap | None = None
        self._last_frame: np.ndarray | None = None

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame_shape = (frame.shape[1], frame.shape[0])
        self._source_size = self._frame_shape
        self._source_rect = (0, 0, frame.shape[1], frame.shape[0])
        self._last_frame = frame.copy()
        self._pixmap = frame_to_pixmap(frame)
        self._update_pixmap()

    def set_source_size(self, width: int, height: int) -> None:
        self._source_size = (width, height)
        self._source_rect = (0, 0, width, height)

    def set_source_rect(self, x: int, y: int, width: int, height: int) -> None:
        self._source_size = (width, height)
        self._source_rect = (x, y, width, height)

    def set_overlay_text(self, text: str) -> None:
        self.overlay_label.setText(text)
        self.overlay_label.setVisible(bool(text))
        self._position_overlay()

    def overlay_text(self) -> str:
        return self.overlay_label.text()

    def resizeEvent(self, event) -> None:  # noqa: ANN001
        super().resizeEvent(event)
        self._update_pixmap()
        self._position_overlay()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._frame_shape is None or self._pixmap is None:
            return
        point = self.map_label_point_to_image(event.position().toPoint())
        if point is not None:
            self.image_clicked.emit(point[0], point[1])

    def map_label_point_to_image(self, pos: QPoint) -> tuple[int, int] | None:
        return self._label_to_image(pos)

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        super().setPixmap(scaled)
        self._position_overlay()

    def _position_overlay(self) -> None:
        if not self.overlay_label.isVisible():
            return
        margin = 8
        self.overlay_label.adjustSize()
        max_width = max(self.width() - margin * 2, 1)
        if self.overlay_label.width() > max_width:
            self.overlay_label.setFixedWidth(max_width)
            self.overlay_label.adjustSize()
        x = max(self.width() - self.overlay_label.width() - margin, margin)
        y = max(self.height() - self.overlay_label.height() - margin, margin)
        self.overlay_label.move(x, y)
        self.overlay_label.raise_()

    def _label_to_image(self, pos: QPoint) -> tuple[int, int] | None:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull() or self._source_size is None:
            return None
        x_offset = (self.width() - pixmap.width()) // 2
        y_offset = (self.height() - pixmap.height()) // 2
        x = pos.x() - x_offset
        y = pos.y() - y_offset
        if x < 0 or y < 0 or x >= pixmap.width() or y >= pixmap.height():
            return None
        source_x, source_y, image_w, image_h = self._source_rect or (0, 0, *self._source_size)
        return source_x + round(x * image_w / pixmap.width()), source_y + round(y * image_h / pixmap.height())
