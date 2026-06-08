from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from industrial_code_reader.core.engine import CodeReaderEngine
from industrial_code_reader.core.types import CodeResult, DecodeOptions, Roi
from industrial_code_reader.datamatrix.decoder import DataMatrixDecoder
from industrial_code_reader.datamatrix.roi import DataMatrixRoiGenerator
from stereo_aruco_gui.app.image_view import ImageView

from decoder_debugger.image_io import list_image_files, read_image_color
from decoder_debugger.overlay import draw_debug_overlay


class DecoderDebuggerWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("DataMatrix Decoder Debugger")
        self.resize(1400, 850)
        self.current_path: Path | None = None
        self.current_image: np.ndarray | None = None
        self.current_folder_images: list[Path] = []
        self.current_rois: tuple[Roi, ...] = ()
        self.current_results: list[CodeResult] = []
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QHBoxLayout(root)

        controls = QVBoxLayout()
        open_image = QPushButton("Open Image")
        open_image.clicked.connect(self.open_image)
        open_folder = QPushButton("Open Folder")
        open_folder.clicked.connect(self.open_folder)
        run_button = QPushButton("Run Decode")
        run_button.clicked.connect(self.run_decode)
        self.path_label = QLabel("No image loaded")
        self.path_label.setWordWrap(True)
        controls.addWidget(open_image)
        controls.addWidget(open_folder)
        controls.addWidget(run_button)
        controls.addWidget(self.path_label)
        controls.addStretch(1)

        self.image_view = ImageView("Decoder image")
        self.image_view.set_zoom_enabled(True)

        inspector = QVBoxLayout()
        self.timing_label = QLabel("Timing: --")
        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        inspector.addWidget(QLabel("Decode Inspector"))
        inspector.addWidget(self.timing_label)
        inspector.addWidget(self.result_box, stretch=1)

        layout.addLayout(controls, stretch=0)
        layout.addWidget(self.image_view, stretch=1)
        layout.addLayout(inspector, stretch=0)
        self.setCentralWidget(root)

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open DataMatrix Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)",
        )
        if path:
            self.load_image_path(Path(path))

    def open_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open Image Folder", "")
        if not folder:
            return
        self.current_folder_images = list_image_files(folder)
        if self.current_folder_images:
            self.load_image_path(self.current_folder_images[0])

    def load_image_path(self, path: str | Path) -> None:
        image_path = Path(path)
        image = read_image_color(image_path)
        if image is None:
            QMessageBox.warning(self, "Error", f"Failed to read image: {image_path}")
            return
        self.current_path = image_path
        self.current_image = image
        self.current_rois = ()
        self.current_results = []
        self.path_label.setText(str(image_path))
        self.result_box.setPlainText("Image loaded. Click Run Decode.")
        self.timing_label.setText("Timing: --")
        self.image_view.set_frame(image)

    def run_decode(self) -> None:
        if self.current_image is None:
            self.result_box.setPlainText("Load an image first.")
            return
        start = perf_counter()
        roi_start = perf_counter()
        self.current_rois = DataMatrixRoiGenerator(max_rois=32).generate(self.current_image)
        roi_elapsed_ms = (perf_counter() - roi_start) * 1000.0
        decode_start = perf_counter()
        engine = CodeReaderEngine([DataMatrixDecoder()])
        self.current_results = engine.decode(
            self.current_image,
            DecodeOptions(rois=self.current_rois[:16], max_results=16, max_rois=16, return_failures=True),
        )
        decode_elapsed_ms = (perf_counter() - decode_start) * 1000.0
        total_elapsed_ms = (perf_counter() - start) * 1000.0
        overlay = draw_debug_overlay(self.current_image, self.current_rois, self.current_results)
        self.image_view.set_frame(overlay)
        self.timing_label.setText(
            f"Total {total_elapsed_ms:.1f} ms | ROI {roi_elapsed_ms:.1f} ms | Decode {decode_elapsed_ms:.1f} ms"
        )
        self.result_box.setPlainText(self._result_text())

    def _result_text(self) -> str:
        lines = [f"Candidates: {len(self.current_rois)}", f"Results: {len(self.current_results)}"]
        for result in self.current_results:
            if result.failure_reason:
                lines.append(f"ROI {result.roi_id}: failed - {result.failure_reason}")
            else:
                lines.append(f"ROI {result.roi_id}: {result.text} ({result.symbology}, {result.preprocessing})")
        return "\n".join(lines)
