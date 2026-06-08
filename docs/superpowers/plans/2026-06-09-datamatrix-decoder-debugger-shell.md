# DataMatrix Decoder Debugger Shell Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a standalone PySide6 decoder debugger that loads PCB images, runs the current industrial DataMatrix pipeline, and visualizes candidate ROIs, decode results, timing, and failure diagnostics independently from the existing stereo camera GUI.

**Architecture:** Add a new `decoder_debugger` package with a small app entry point, a pure image-loading helper, an overlay renderer, and a main window. Reuse `industrial_code_reader` for ROI/decode work and reuse the existing `ImageView` widget for zoomable image display. Keep `stereo_aruco_gui` unchanged except for shared helper reuse if needed.

**Tech Stack:** Python, PySide6, OpenCV, NumPy, pytest, existing `industrial_code_reader`, existing `stereo_aruco_gui.app.image_view.ImageView`.

---

## File Structure

- Create `decoder_debugger/__init__.py`: package marker and public app description.
- Create `decoder_debugger/main.py`: `python -m decoder_debugger.main` entry point.
- Create `decoder_debugger/image_io.py`: Unicode-safe image loading and folder image listing.
- Create `decoder_debugger/overlay.py`: draw ROI/result overlays without GUI dependencies.
- Create `decoder_debugger/main_window.py`: standalone debugger window.
- Create `tests/test_decoder_debugger.py`: unit and GUI smoke tests for image IO, overlays, and window behavior.
- Modify `README.md`: add a short command for launching the decoder debugger.

## Task 1: Image IO Helper

**Files:**
- Create: `decoder_debugger/__init__.py`
- Create: `decoder_debugger/image_io.py`
- Test: `tests/test_decoder_debugger.py`

- [ ] **Step 1: Write the failing tests**

Add this to `tests/test_decoder_debugger.py`:

```python
from __future__ import annotations

from pathlib import Path

import numpy as np

import decoder_debugger.image_io as image_io_module
from decoder_debugger.image_io import list_image_files, read_image_color


def test_read_image_color_uses_unicode_safe_decode_before_imread(monkeypatch, tmp_path):
    image_path = tmp_path / "unicode-image.bmp"
    image_path.write_bytes(b"encoded image bytes")
    decoded = np.zeros((8, 9, 3), dtype=np.uint8)
    calls = {"imread": 0, "imdecode": 0}

    def fake_imread(path, flags):  # noqa: ANN001
        calls["imread"] += 1
        return None

    def fake_imdecode(data, flags):  # noqa: ANN001
        calls["imdecode"] += 1
        assert data.tobytes() == b"encoded image bytes"
        return decoded

    monkeypatch.setattr(image_io_module.cv2, "imread", fake_imread)
    monkeypatch.setattr(image_io_module.cv2, "imdecode", fake_imdecode)

    assert read_image_color(image_path) is decoded
    assert calls == {"imread": 0, "imdecode": 1}


def test_list_image_files_returns_supported_images_sorted(tmp_path):
    (tmp_path / "b.bmp").write_bytes(b"b")
    (tmp_path / "a.PNG").write_bytes(b"a")
    (tmp_path / "note.txt").write_text("skip", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.jpg").write_bytes(b"c")

    files = list_image_files(tmp_path)

    assert [path.name for path in files] == ["a.PNG", "b.bmp"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_read_image_color_uses_unicode_safe_decode_before_imread tests/test_decoder_debugger.py::test_list_image_files_returns_supported_images_sorted -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'decoder_debugger'`.

- [ ] **Step 3: Implement package and image IO**

Create `decoder_debugger/__init__.py`:

```python
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
```

Create `decoder_debugger/image_io.py`:

```python
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_image_color(path: str | Path) -> np.ndarray | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def list_image_files(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_read_image_color_uses_unicode_safe_decode_before_imread tests/test_decoder_debugger.py::test_list_image_files_returns_supported_images_sorted -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add decoder_debugger/__init__.py decoder_debugger/image_io.py tests/test_decoder_debugger.py
git commit -m "Add decoder debugger image IO"
```

## Task 2: Overlay Renderer

**Files:**
- Create: `decoder_debugger/overlay.py`
- Test: `tests/test_decoder_debugger.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_decoder_debugger.py`:

```python
from industrial_code_reader.core.types import CodeResult, Roi
from decoder_debugger.overlay import draw_debug_overlay


def test_draw_debug_overlay_marks_rois_results_and_failures():
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    rois = (
        Roi(id=1, x=10, y=12, width=20, height=18, quality={"source": "test"}),
        Roi(id=2, x=50, y=20, width=24, height=22, quality={"source": "test"}),
    )
    results = [
        CodeResult(text="DM-1", symbology="DataMatrix", roi_id=1, bbox=(10, 12, 20, 18)),
        CodeResult(text="", symbology="DataMatrix", roi_id=2, bbox=(50, 20, 24, 22), failure_reason="No DataMatrix decoded in ROI"),
    ]

    overlay = draw_debug_overlay(image, rois, results)

    assert overlay.shape == image.shape
    assert np.any(overlay != image)
    assert np.array_equal(overlay[12, 10], np.array([0, 255, 0], dtype=np.uint8))
    assert np.array_equal(overlay[20, 50], np.array([0, 0, 255], dtype=np.uint8))
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_draw_debug_overlay_marks_rois_results_and_failures -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'decoder_debugger.overlay'`.

- [ ] **Step 3: Implement overlay renderer**

Create `decoder_debugger/overlay.py`:

```python
from __future__ import annotations

import cv2
import numpy as np

from industrial_code_reader.core.types import CodeResult, Roi


def draw_debug_overlay(image: np.ndarray, rois: tuple[Roi, ...], results: list[CodeResult]) -> np.ndarray:
    output = image.copy()
    result_by_roi = {result.roi_id: result for result in results}
    for roi in rois:
        result = result_by_roi.get(roi.id)
        color = (0, 0, 255) if result is not None and result.failure_reason else (0, 255, 0)
        cv2.rectangle(output, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), color, 2, cv2.LINE_AA)
        label = _roi_label(roi, result)
        y = max(roi.y - 8, 14)
        cv2.putText(output, label[:80], (roi.x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return output


def _roi_label(roi: Roi, result: CodeResult | None) -> str:
    if result is None:
        return f"ROI {roi.id}"
    if result.failure_reason:
        return f"ROI {roi.id}: failed"
    return f"ROI {roi.id}: {result.text}"
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_draw_debug_overlay_marks_rois_results_and_failures -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```powershell
git add decoder_debugger/overlay.py tests/test_decoder_debugger.py
git commit -m "Add decoder debugger overlay renderer"
```

## Task 3: Standalone Main Window

**Files:**
- Create: `decoder_debugger/main_window.py`
- Test: `tests/test_decoder_debugger.py`

- [ ] **Step 1: Write the failing GUI tests**

Append to `tests/test_decoder_debugger.py`:

```python
import os

from PySide6.QtWidgets import QApplication

import decoder_debugger.main_window as debugger_window_module
from decoder_debugger.main_window import DecoderDebuggerWindow

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_decoder_debugger_window_loads_image_and_runs_decode(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "sample.bmp"
    image_path.write_bytes(b"fake")
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    rois = (Roi(id=1, x=5, y=6, width=20, height=20),)
    results = [CodeResult(text="ABC123", symbology="DataMatrix", roi_id=1, bbox=(5, 6, 20, 20), preprocessing="fake")]

    monkeypatch.setattr(debugger_window_module, "read_image_color", lambda path: frame.copy())
    monkeypatch.setattr(debugger_window_module.DataMatrixRoiGenerator, "generate", lambda self, image: rois)

    class FakeEngine:
        def __init__(self, decoders):  # noqa: ANN001
            pass

        def decode(self, image, options):  # noqa: ANN001
            return results

    monkeypatch.setattr(debugger_window_module, "CodeReaderEngine", FakeEngine)

    window = DecoderDebuggerWindow()
    window.load_image_path(image_path)
    window.run_decode()

    assert app is not None
    assert window.current_path == image_path
    assert window.image_view._last_frame is not None
    assert "ABC123" in window.result_box.toPlainText()
    assert "ROI 1" in window.result_box.toPlainText()
    assert "Total" in window.timing_label.text()
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_decoder_debugger_window_loads_image_and_runs_decode -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'decoder_debugger.main_window'`.

- [ ] **Step 3: Implement main window**

Create `decoder_debugger/main_window.py`:

```python
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
            DecodeOptions(auto_rois=True, max_results=16, max_rois=16, return_failures=True),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_decoder_debugger_window_loads_image_and_runs_decode -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```powershell
git add decoder_debugger/main_window.py tests/test_decoder_debugger.py
git commit -m "Add standalone decoder debugger window"
```

## Task 4: Entry Point And README

**Files:**
- Create: `decoder_debugger/main.py`
- Modify: `README.md`
- Test: `tests/test_decoder_debugger.py`

- [ ] **Step 1: Write failing test for app construction**

Append to `tests/test_decoder_debugger.py`:

```python
import decoder_debugger.main as debugger_main_module


def test_main_creates_decoder_debugger_window(monkeypatch):
    created = {}

    class FakeApp:
        def __init__(self, argv):  # noqa: ANN001
            created["argv"] = argv

        def exec(self):  # noqa: A003
            return 0

    class FakeWindow:
        def show(self):
            created["shown"] = True

    monkeypatch.setattr(debugger_main_module, "QApplication", FakeApp)
    monkeypatch.setattr(debugger_main_module, "DecoderDebuggerWindow", lambda: FakeWindow())

    assert debugger_main_module.main(["decoder"]) == 0
    assert created == {"argv": ["decoder"], "shown": True}
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_main_creates_decoder_debugger_window -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'decoder_debugger.main'`.

- [ ] **Step 3: Implement entry point**

Create `decoder_debugger/main.py`:

```python
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from decoder_debugger.main_window import DecoderDebuggerWindow


def main(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    window = DecoderDebuggerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Add README command**

Add this near the run instructions in `README.md`:

```markdown
## Decoder Debugger

Run the standalone DataMatrix decoder debugger without opening cameras:

```powershell
python -m decoder_debugger.main
```
```

- [ ] **Step 5: Run entry point test**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_main_creates_decoder_debugger_window -q
```

Expected: `1 passed`.

- [ ] **Step 6: Commit**

```powershell
git add decoder_debugger/main.py README.md tests/test_decoder_debugger.py
git commit -m "Add decoder debugger entry point"
```

## Task 5: Full Verification

**Files:**
- Verify all touched files.

- [ ] **Step 1: Run debugger tests**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py -q
```

Expected: all decoder debugger tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Manual launch check**

Run:

```powershell
python -m decoder_debugger.main
```

Expected: a standalone window titled `DataMatrix Decoder Debugger` opens. Close it manually after confirming.

- [ ] **Step 4: Final commit if verification-only changes were needed**

If files changed during verification, commit them:

```powershell
git add decoder_debugger tests README.md
git commit -m "Stabilize decoder debugger shell"
```

If no files changed, do not create an empty commit.

## Self-Review

- Spec coverage: This plan implements Phase 1 from the design spec: standalone debugger shell, image/folder loading, current industrial pipeline execution, ROI/result overlays, timing, and independent entry point. ECC200 self-owned decoding, finder/grid/sampling, and PCB hardening are explicitly Phase 2+ and need separate plans.
- Placeholder scan: No placeholder markers are used. Each task includes exact files, test names, commands, and code.
- Type consistency: The plan uses existing `Roi`, `CodeResult`, `DecodeOptions`, `DataMatrixRoiGenerator`, `CodeReaderEngine`, `DataMatrixDecoder`, and `ImageView` names consistently.
