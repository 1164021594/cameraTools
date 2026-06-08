from __future__ import annotations

import cv2
import numpy as np
from PySide6.QtCore import QPoint, QPointF, QSize, Qt, Signal
from PySide6.QtGui import QImage, QMouseEvent, QPixmap, QWheelEvent
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
        self._zoom_enabled = False
        self._zoom_factor = 1.0
        self._view_center: QPointF | None = None
        self._drag_start: QPoint | None = None

    @property
    def zoom_enabled(self) -> bool:
        return self._zoom_enabled

    @property
    def zoom_factor(self) -> float:
        return self._zoom_factor

    def set_zoom_enabled(self, enabled: bool) -> None:
        self._zoom_enabled = enabled
        if not enabled:
            self.reset_zoom()
        else:
            self._update_pixmap()

    def reset_zoom(self) -> None:
        self._zoom_factor = 1.0
        self._view_center = None
        self._update_pixmap()

    def zoom_by(self, factor: float) -> None:
        if not self._zoom_enabled or self._pixmap is None:
            return
        self._ensure_view_center()
        self._zoom_factor = max(0.1, min(8.0, self._zoom_factor * factor))
        self._update_pixmap()

    def pan_by(self, dx: int, dy: int) -> None:
        if not self._zoom_enabled or self._pixmap is None:
            return
        self._ensure_view_center()
        display = self._display_size()
        if display.width() <= 0 or display.height() <= 0 or self._view_center is None:
            return
        image_dx = dx * self._pixmap.width() / display.width()
        image_dy = dy * self._pixmap.height() / display.height()
        self._view_center = QPointF(self._view_center.x() - image_dx, self._view_center.y() - image_dy)
        self._clamp_view_center()
        self._update_pixmap()

    def set_frame(self, frame: np.ndarray) -> None:
        self._frame_shape = (frame.shape[1], frame.shape[0])
        self._source_size = self._frame_shape
        self._source_rect = (0, 0, frame.shape[1], frame.shape[0])
        self._last_frame = frame.copy()
        self._pixmap = frame_to_pixmap(frame)
        self._view_center = None
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

    def wheelEvent(self, event: QWheelEvent) -> None:
        if not self._zoom_enabled:
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        self.zoom_by(1.25 if delta > 0 else 0.8)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if self._frame_shape is None or self._pixmap is None:
            return
        if self._zoom_enabled and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = event.position().toPoint()
            event.accept()
            return
        point = self.map_label_point_to_image(event.position().toPoint())
        if point is not None:
            self.image_clicked.emit(point[0], point[1])

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is None:
            super().mouseMoveEvent(event)
            return
        current = event.position().toPoint()
        delta = current - self._drag_start
        self.pan_by(delta.x(), delta.y())
        self._drag_start = current
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._drag_start is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag_start = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def map_label_point_to_image(self, pos: QPoint) -> tuple[int, int] | None:
        return self._label_to_image(pos)

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
            return
        if self._zoom_enabled:
            self._ensure_view_center()
            scaled = self._zoomed_pixmap()
        else:
            scaled = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        super().setPixmap(scaled)
        self._position_overlay()

    def _display_size(self) -> QSize:
        if self._pixmap is None:
            return QSize(0, 0)
        fit = self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
        return QSize(max(int(fit.width() * self._zoom_factor), 1), max(int(fit.height() * self._zoom_factor), 1))

    def _ensure_view_center(self) -> None:
        if self._pixmap is None:
            return
        if self._view_center is None:
            self._view_center = QPointF(self._pixmap.width() / 2, self._pixmap.height() / 2)
        self._clamp_view_center()

    def _clamp_view_center(self) -> None:
        if self._pixmap is None or self._view_center is None:
            return
        display = self._display_size()
        visible_w = min(self._pixmap.width(), self._pixmap.width() * self.width() / max(display.width(), 1))
        visible_h = min(self._pixmap.height(), self._pixmap.height() * self.height() / max(display.height(), 1))
        min_x = visible_w / 2
        max_x = self._pixmap.width() - visible_w / 2
        min_y = visible_h / 2
        max_y = self._pixmap.height() - visible_h / 2
        x = min(max(self._view_center.x(), min_x), max_x) if min_x <= max_x else self._pixmap.width() / 2
        y = min(max(self._view_center.y(), min_y), max_y) if min_y <= max_y else self._pixmap.height() / 2
        self._view_center = QPointF(x, y)

    def _zoomed_pixmap(self) -> QPixmap:
        if self._pixmap is None or self._view_center is None:
            return QPixmap()
        display = self._display_size()
        visible_w = min(self._pixmap.width(), self._pixmap.width() * self.width() / max(display.width(), 1))
        visible_h = min(self._pixmap.height(), self._pixmap.height() * self.height() / max(display.height(), 1))
        left = int(round(self._view_center.x() - visible_w / 2))
        top = int(round(self._view_center.y() - visible_h / 2))
        left = max(0, min(left, max(self._pixmap.width() - int(round(visible_w)), 0)))
        top = max(0, min(top, max(self._pixmap.height() - int(round(visible_h)), 0)))
        cropped = self._pixmap.copy(left, top, max(int(round(visible_w)), 1), max(int(round(visible_h)), 1))
        target_w = min(display.width(), self.width())
        target_h = min(display.height(), self.height())
        return cropped.scaled(target_w, target_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)

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
        if self._zoom_enabled and self._pixmap is not None and self._view_center is not None:
            display = self._display_size()
            visible_w = min(self._pixmap.width(), self._pixmap.width() * self.width() / max(display.width(), 1))
            visible_h = min(self._pixmap.height(), self._pixmap.height() * self.height() / max(display.height(), 1))
            left = self._view_center.x() - visible_w / 2
            top = self._view_center.y() - visible_h / 2
            image_x = left + x * visible_w / pixmap.width()
            image_y = top + y * visible_h / pixmap.height()
            return source_x + round(image_x * image_w / self._pixmap.width()), source_y + round(image_y * image_h / self._pixmap.height())
        return source_x + round(x * image_w / pixmap.width()), source_y + round(y * image_h / pixmap.height())
