# Decoder Debugger Preprocess UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a preprocessing workbench to the standalone decoder debugger with top tabs, operator controls, live original/preprocessed preview switching, and find/decode controls that can use either original or preprocessed input.

**Architecture:** Introduce a pure `decoder_debugger.preprocess` module for configuration, operator execution, stage images, and preset JSON. Refactor `DecoderDebuggerWindow` into a tabbed UI that binds widgets to `PreprocessConfig`, keeps original image as the default display, and passes selected original/preprocessed input to ROI search and selected-ROI decode. Preserve the current standalone entry point.

**Tech Stack:** Python, PySide6, OpenCV, NumPy, pytest, existing `decoder_debugger`, existing `industrial_code_reader`, existing `ImageView`.

---

## File Structure

- Create `decoder_debugger/preprocess.py`: pure preprocessing config, stage execution, preset serialization.
- Modify `decoder_debugger/main_window.py`: replace single layout with tabs and workflow controls.
- Modify `decoder_debugger/overlay.py`: allow selected ROI highlighting.
- Modify `tests/test_decoder_debugger.py`: add preprocess model tests and GUI workflow smoke tests.

## Task 1: Preprocess Model And Operators

**Files:**
- Create: `decoder_debugger/preprocess.py`
- Test: `tests/test_decoder_debugger.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_decoder_debugger.py`:

```python
from decoder_debugger.preprocess import PreprocessConfig, apply_preprocess, config_from_json, config_to_json


def test_preprocess_config_defaults_show_original_without_processing():
    config = PreprocessConfig()

    assert config.view == "Original"
    assert config.channel == "Original"
    assert config.threshold_mode == "None"
    assert config.morphology == "None"


def test_apply_preprocess_returns_stage_images_and_threshold_result():
    image = np.zeros((20, 30, 3), dtype=np.uint8)
    image[:, :15] = (20, 20, 20)
    image[:, 15:] = (220, 220, 220)
    config = PreprocessConfig(channel="Gray", threshold_mode="Manual", manual_threshold=100)

    result = apply_preprocess(image, config)

    assert result.output.shape == (20, 30)
    assert "Original" in result.stages
    assert "Gray" in result.stages
    assert "Threshold" in result.stages
    assert int(result.output[0, 0]) == 0
    assert int(result.output[0, 20]) == 255
    assert result.timings_ms


def test_preprocess_config_json_round_trip():
    config = PreprocessConfig(
        view="Preprocessed",
        channel="HSV Saturation",
        clahe_enabled=True,
        clahe_clip_limit=3.0,
        blur_mode="Gaussian",
        blur_kernel=5,
        threshold_mode="Adaptive",
        adaptive_block_size=31,
        adaptive_c=5,
        morphology="Open",
        morphology_kernel=3,
        morphology_iterations=2,
        scale_factor=2.0,
        quiet_zone_padding=8,
    )

    loaded = config_from_json(config_to_json(config))

    assert loaded == config
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_preprocess_config_defaults_show_original_without_processing tests/test_decoder_debugger.py::test_apply_preprocess_returns_stage_images_and_threshold_result tests/test_decoder_debugger.py::test_preprocess_config_json_round_trip -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'decoder_debugger.preprocess'`.

- [ ] **Step 3: Implement preprocessing module**

Create `decoder_debugger/preprocess.py`:

```python
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from time import perf_counter

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    view: str = "Original"
    channel: str = "Original"
    clahe_enabled: bool = False
    clahe_clip_limit: float = 2.0
    clahe_tile_size: int = 8
    blur_mode: str = "None"
    blur_kernel: int = 3
    sharpen_enabled: bool = False
    sharpen_strength: float = 1.5
    sharpen_radius: float = 1.0
    threshold_mode: str = "None"
    adaptive_block_size: int = 21
    adaptive_c: int = 4
    manual_threshold: int = 128
    threshold_invert: bool = False
    morphology: str = "None"
    morphology_kernel: int = 3
    morphology_iterations: int = 1
    scale_factor: float = 1.0
    quiet_zone_padding: int = 0


@dataclass(frozen=True)
class PreprocessResult:
    output: np.ndarray
    stages: dict[str, np.ndarray]
    timings_ms: dict[str, float]
    config: PreprocessConfig


def apply_preprocess(image: np.ndarray, config: PreprocessConfig) -> PreprocessResult:
    stages: dict[str, np.ndarray] = {"Original": image.copy()}
    timings: dict[str, float] = {}
    current = _timed("Channel", timings, lambda: _select_channel(image, config.channel))
    stages[_stage_name_for_channel(config.channel)] = current.copy()

    if config.clahe_enabled:
        current = _timed("CLAHE", timings, lambda: _apply_clahe(current, config))
        stages["CLAHE"] = current.copy()

    if config.blur_mode != "None":
        current = _timed("Blur", timings, lambda: _apply_blur(current, config))
        stages["Blur"] = current.copy()

    if config.sharpen_enabled:
        current = _timed("Sharpen", timings, lambda: _apply_sharpen(current, config))
        stages["Sharpen"] = current.copy()

    if config.threshold_mode != "None":
        current = _timed("Threshold", timings, lambda: _apply_threshold(current, config))
        stages["Threshold"] = current.copy()

    if config.morphology != "None":
        current = _timed("Morphology", timings, lambda: _apply_morphology(current, config))
        stages["Morphology"] = current.copy()

    if config.scale_factor != 1.0:
        current = _timed("Scale", timings, lambda: cv2.resize(current, None, fx=config.scale_factor, fy=config.scale_factor, interpolation=cv2.INTER_CUBIC))
        stages["Scale"] = current.copy()

    if config.quiet_zone_padding > 0:
        pad = config.quiet_zone_padding
        current = _timed("Padding", timings, lambda: cv2.copyMakeBorder(current, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255))
        stages["Padding"] = current.copy()

    return PreprocessResult(output=current, stages=stages, timings_ms=timings, config=config)


def config_to_json(config: PreprocessConfig) -> str:
    return json.dumps(asdict(config), ensure_ascii=False, indent=2, sort_keys=True)


def config_from_json(text: str) -> PreprocessConfig:
    values = json.loads(text)
    return PreprocessConfig(**values)


def _timed(name: str, timings: dict[str, float], func):  # noqa: ANN001
    start = perf_counter()
    result = func()
    timings[name] = (perf_counter() - start) * 1000.0
    return result


def _stage_name_for_channel(channel: str) -> str:
    return "Original" if channel == "Original" else channel


def _select_channel(image: np.ndarray, channel: str) -> np.ndarray:
    if channel == "Original":
        return image.copy()
    if channel == "Gray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if image.ndim == 2:
        return image.copy()
    if channel in {"B", "G", "R"}:
        return image[:, :, {"B": 0, "G": 1, "R": 2}[channel]].copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    if channel == "HSV Saturation":
        return hsv[:, :, 1]
    if channel == "HSV Value":
        return hsv[:, :, 2]
    return image.copy()


def _to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image


def _odd(value: int) -> int:
    value = max(1, int(value))
    return value if value % 2 else value + 1


def _apply_clahe(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    gray = _to_gray(image)
    tile = max(1, int(config.clahe_tile_size))
    return cv2.createCLAHE(clipLimit=max(0.1, float(config.clahe_clip_limit)), tileGridSize=(tile, tile)).apply(gray)


def _apply_blur(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    kernel = _odd(config.blur_kernel)
    if config.blur_mode == "Gaussian":
        return cv2.GaussianBlur(image, (kernel, kernel), 0)
    if config.blur_mode == "Median":
        return cv2.medianBlur(image, kernel)
    return image


def _apply_sharpen(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), max(0.1, float(config.sharpen_radius)))
    return cv2.addWeighted(image, float(config.sharpen_strength), blurred, 1.0 - float(config.sharpen_strength), 0)


def _apply_threshold(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    gray = _to_gray(image)
    threshold_type = cv2.THRESH_BINARY_INV if config.threshold_invert else cv2.THRESH_BINARY
    if config.threshold_mode == "Otsu":
        _, binary = cv2.threshold(gray, 0, 255, threshold_type | cv2.THRESH_OTSU)
        return binary
    if config.threshold_mode == "Adaptive":
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type,
            _odd(config.adaptive_block_size),
            int(config.adaptive_c),
        )
    if config.threshold_mode == "Manual":
        _, binary = cv2.threshold(gray, int(config.manual_threshold), 255, threshold_type)
        return binary
    return image


def _apply_morphology(image: np.ndarray, config: PreprocessConfig) -> np.ndarray:
    kernel_size = max(1, int(config.morphology_kernel))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    op = {
        "Erode": cv2.MORPH_ERODE,
        "Dilate": cv2.MORPH_DILATE,
        "Open": cv2.MORPH_OPEN,
        "Close": cv2.MORPH_CLOSE,
    }.get(config.morphology)
    if op is None:
        return image
    return cv2.morphologyEx(image, op, kernel, iterations=max(1, int(config.morphology_iterations)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_preprocess_config_defaults_show_original_without_processing tests/test_decoder_debugger.py::test_apply_preprocess_returns_stage_images_and_threshold_result tests/test_decoder_debugger.py::test_preprocess_config_json_round_trip -q
```

Expected: `3 passed`.

- [ ] **Step 5: Commit**

```powershell
git add decoder_debugger/preprocess.py tests/test_decoder_debugger.py
git commit -m "Add decoder debugger preprocessing model"
```

## Task 2: Tabbed Debugger Layout

**Files:**
- Modify: `decoder_debugger/main_window.py`
- Test: `tests/test_decoder_debugger.py`

- [ ] **Step 1: Write failing GUI test**

Append to `tests/test_decoder_debugger.py`:

```python
def test_decoder_debugger_has_preprocess_find_decode_and_logs_tabs():
    app = QApplication.instance() or QApplication([])
    window = DecoderDebuggerWindow()

    tab_names = [window.tabs.tabText(index) for index in range(window.tabs.count())]

    assert app is not None
    assert tab_names == ["Preprocess", "Find / Decode", "Batch / Logs"]
    assert window.preview_view_mode.currentText() == "Original"
    assert window.find_input_view.currentText() == "Original"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_decoder_debugger_has_preprocess_find_decode_and_logs_tabs -q
```

Expected: FAIL with `AttributeError: 'DecoderDebuggerWindow' object has no attribute 'tabs'`.

- [ ] **Step 3: Refactor main window to tabs**

Modify imports in `decoder_debugger/main_window.py` to include:

```python
from PySide6.QtWidgets import (
    QFileDialog,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
```

Add imports:

```python
from decoder_debugger.preprocess import PreprocessConfig, PreprocessResult, apply_preprocess
```

Add state in `__init__` before `_build_ui()`:

```python
self.preprocess_config = PreprocessConfig()
self.preprocess_result: PreprocessResult | None = None
self.selected_roi_id: int | None = None
```

Replace `_build_ui()` with:

```python
def _build_ui(self) -> None:
    self.tabs = QTabWidget()
    self.tabs.addTab(self._preprocess_tab(), "Preprocess")
    self.tabs.addTab(self._find_decode_tab(), "Find / Decode")
    self.tabs.addTab(self._logs_tab(), "Batch / Logs")
    self.setCentralWidget(self.tabs)


def _preprocess_tab(self) -> QWidget:
    root = QWidget()
    layout = QHBoxLayout(root)
    controls = QVBoxLayout()
    open_image = QPushButton("Open Image")
    open_image.clicked.connect(self.open_image)
    self.preview_view_mode = QComboBox()
    self.preview_view_mode.addItems(("Original", "Preprocessed"))
    self.preview_view_mode.currentTextChanged.connect(lambda _text: self._refresh_preprocess_preview())
    controls.addWidget(open_image)
    controls.addWidget(QLabel("Preview View"))
    controls.addWidget(self.preview_view_mode)
    controls.addWidget(self._operator_group())
    controls.addStretch(1)
    self.image_view = ImageView("Preprocess preview")
    self.image_view.set_zoom_enabled(True)
    layout.addLayout(controls, stretch=0)
    layout.addWidget(self.image_view, stretch=1)
    return root


def _operator_group(self) -> QGroupBox:
    group = QGroupBox("Preprocess Operators")
    form = QFormLayout(group)
    self.channel_select = QComboBox()
    self.channel_select.addItems(("Original", "Gray", "B", "G", "R", "HSV Saturation", "HSV Value"))
    self.channel_select.currentTextChanged.connect(lambda _text: self._preprocess_controls_changed())
    self.threshold_mode = QComboBox()
    self.threshold_mode.addItems(("None", "Otsu", "Adaptive", "Manual"))
    self.threshold_mode.currentTextChanged.connect(lambda _text: self._preprocess_controls_changed())
    self.manual_threshold = QSpinBox()
    self.manual_threshold.setRange(0, 255)
    self.manual_threshold.setValue(128)
    self.manual_threshold.valueChanged.connect(lambda _value: self._preprocess_controls_changed())
    form.addRow("Channel", self.channel_select)
    form.addRow("Threshold", self.threshold_mode)
    form.addRow("Manual value", self.manual_threshold)
    return group


def _find_decode_tab(self) -> QWidget:
    root = QWidget()
    layout = QHBoxLayout(root)
    controls = QVBoxLayout()
    self.find_input_view = QComboBox()
    self.find_input_view.addItems(("Original", "Preprocessed"))
    find_button = QPushButton("Find Code")
    find_button.clicked.connect(self.find_code)
    decode_button = QPushButton("Decode Selected ROI")
    decode_button.clicked.connect(self.decode_selected_roi)
    self.find_decode_view = ImageView("Find / Decode")
    self.find_decode_view.set_zoom_enabled(True)
    self.find_decode_result_box = QTextEdit()
    self.find_decode_result_box.setReadOnly(True)
    controls.addWidget(QLabel("Input View"))
    controls.addWidget(self.find_input_view)
    controls.addWidget(find_button)
    controls.addWidget(decode_button)
    controls.addStretch(1)
    layout.addLayout(controls, stretch=0)
    layout.addWidget(self.find_decode_view, stretch=1)
    layout.addWidget(self.find_decode_result_box, stretch=0)
    return root


def _logs_tab(self) -> QWidget:
    root = QWidget()
    layout = QVBoxLayout(root)
    self.batch_log_box = QTextEdit()
    self.batch_log_box.setReadOnly(True)
    self.batch_log_box.setPlainText("Batch / Logs")
    layout.addWidget(self.batch_log_box)
    return root
```

Keep existing `open_image`, `open_folder`, `load_image_path`, and `run_decode` for compatibility. They will be refined in later tasks.

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_decoder_debugger_has_preprocess_find_decode_and_logs_tabs -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```powershell
git add decoder_debugger/main_window.py tests/test_decoder_debugger.py
git commit -m "Add decoder debugger tab layout"
```

## Task 3: Live Preprocess Preview

**Files:**
- Modify: `decoder_debugger/main_window.py`
- Test: `tests/test_decoder_debugger.py`

- [ ] **Step 1: Write failing GUI test**

Append to `tests/test_decoder_debugger.py`:

```python
def test_preprocess_controls_update_preprocessed_preview(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "sample.bmp"
    image_path.write_bytes(b"fake")
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, 15:] = 220
    monkeypatch.setattr(debugger_window_module, "read_image_color", lambda path: frame.copy())
    window = DecoderDebuggerWindow()

    window.load_image_path(image_path)
    window.preview_view_mode.setCurrentText("Preprocessed")
    window.channel_select.setCurrentText("Gray")
    window.threshold_mode.setCurrentText("Manual")
    window.manual_threshold.setValue(100)

    assert app is not None
    assert window.preprocess_result is not None
    assert window.image_view._last_frame is not None
    assert window.image_view._last_frame.shape[:2] == (20, 30)
    assert int(window.image_view._last_frame[0, 0, 0]) == 0
    assert int(window.image_view._last_frame[0, 20, 0]) == 255
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_preprocess_controls_update_preprocessed_preview -q
```

Expected: FAIL because `_preprocess_controls_changed` or preview refresh behavior is missing.

- [ ] **Step 3: Implement config binding and preview refresh**

Add to `decoder_debugger/main_window.py`:

```python
def _preprocess_controls_changed(self) -> None:
    self.preprocess_config = PreprocessConfig(
        view=self.preview_view_mode.currentText(),
        channel=self.channel_select.currentText(),
        threshold_mode=self.threshold_mode.currentText(),
        manual_threshold=self.manual_threshold.value(),
    )
    self._refresh_preprocess_preview()


def _refresh_preprocess_preview(self) -> None:
    if self.current_image is None:
        return
    self.preprocess_result = apply_preprocess(self.current_image, self.preprocess_config)
    if self.preview_view_mode.currentText() == "Preprocessed":
        self.image_view.set_frame(_displayable_image(self.preprocess_result.output))
    else:
        self.image_view.set_frame(self.current_image)


def _selected_input_image(self) -> np.ndarray | None:
    if self.current_image is None:
        return None
    if self.find_input_view.currentText() == "Preprocessed":
        if self.preprocess_result is None:
            self.preprocess_result = apply_preprocess(self.current_image, self.preprocess_config)
        return _displayable_image(self.preprocess_result.output)
    return self.current_image


def _displayable_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image
```

Also import `cv2` at the top of `decoder_debugger/main_window.py`.

Update `load_image_path` after `self.image_view.set_frame(image)`:

```python
self.preprocess_result = apply_preprocess(image, self.preprocess_config)
self.find_decode_view.set_frame(image)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_preprocess_controls_update_preprocessed_preview -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```powershell
git add decoder_debugger/main_window.py tests/test_decoder_debugger.py
git commit -m "Add live preprocessing preview"
```

## Task 4: Find And Decode Selected ROI With Selected Input

**Files:**
- Modify: `decoder_debugger/main_window.py`
- Modify: `decoder_debugger/overlay.py`
- Test: `tests/test_decoder_debugger.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_decoder_debugger.py`:

```python
def test_find_code_uses_preprocessed_input_when_selected(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "sample.bmp"
    image_path.write_bytes(b"fake")
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    frame[:, 15:] = 220
    captured = {}
    rois = (Roi(id=1, x=2, y=3, width=10, height=10),)
    monkeypatch.setattr(debugger_window_module, "read_image_color", lambda path: frame.copy())

    def fake_generate(self, image):  # noqa: ANN001
        captured["shape"] = image.shape
        captured["right_pixel"] = int(image[0, 20, 0])
        return rois

    monkeypatch.setattr(debugger_window_module.DataMatrixRoiGenerator, "generate", fake_generate)
    window = DecoderDebuggerWindow()
    window.load_image_path(image_path)
    window.channel_select.setCurrentText("Gray")
    window.threshold_mode.setCurrentText("Manual")
    window.manual_threshold.setValue(100)
    window.find_input_view.setCurrentText("Preprocessed")

    window.find_code()

    assert app is not None
    assert captured == {"shape": (20, 30, 3), "right_pixel": 255}
    assert window.current_rois == rois
    assert "Candidates: 1" in window.find_decode_result_box.toPlainText()


def test_decode_selected_roi_decodes_only_selected_roi(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "sample.bmp"
    image_path.write_bytes(b"fake")
    frame = np.zeros((40, 50, 3), dtype=np.uint8)
    rois = (
        Roi(id=1, x=5, y=6, width=10, height=10),
        Roi(id=2, x=20, y=10, width=12, height=12),
    )
    results = [CodeResult(text="ROI2", symbology="DataMatrix", roi_id=2, bbox=(20, 10, 12, 12), preprocessing="fake")]
    captured = {}
    monkeypatch.setattr(debugger_window_module, "read_image_color", lambda path: frame.copy())

    class FakeEngine:
        def __init__(self, decoders):  # noqa: ANN001
            pass

        def decode(self, image, options):  # noqa: ANN001
            captured["rois"] = options.rois
            return results

    monkeypatch.setattr(debugger_window_module, "CodeReaderEngine", FakeEngine)
    window = DecoderDebuggerWindow()
    window.load_image_path(image_path)
    window.current_rois = rois
    window.selected_roi_id = 2

    window.decode_selected_roi()

    assert app is not None
    assert captured["rois"] == (rois[1],)
    assert "ROI2" in window.find_decode_result_box.toPlainText()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_find_code_uses_preprocessed_input_when_selected tests/test_decoder_debugger.py::test_decode_selected_roi_decodes_only_selected_roi -q
```

Expected: FAIL because `find_code` and `decode_selected_roi` are not implemented.

- [ ] **Step 3: Implement find/decode methods**

Add to `decoder_debugger/main_window.py`:

```python
def find_code(self) -> None:
    image = self._selected_input_image()
    if image is None:
        self.find_decode_result_box.setPlainText("Load an image first.")
        return
    start = perf_counter()
    self.current_rois = DataMatrixRoiGenerator(max_rois=32).generate(image)
    self.selected_roi_id = self.current_rois[0].id if self.current_rois else None
    overlay = draw_debug_overlay(image, self.current_rois, [])
    self.find_decode_view.set_frame(overlay)
    elapsed_ms = (perf_counter() - start) * 1000.0
    self.find_decode_result_box.setPlainText(f"Candidates: {len(self.current_rois)}\nFind Code: {elapsed_ms:.1f} ms")


def decode_selected_roi(self) -> None:
    image = self._selected_input_image()
    if image is None:
        self.find_decode_result_box.setPlainText("Load an image first.")
        return
    if not self.current_rois:
        self.find_decode_result_box.setPlainText("Run Find Code first.")
        return
    selected = self._selected_roi()
    start = perf_counter()
    engine = CodeReaderEngine([DataMatrixDecoder()])
    self.current_results = engine.decode(
        image,
        DecodeOptions(rois=(selected,), max_results=16, max_rois=16, return_failures=True),
    )
    elapsed_ms = (perf_counter() - start) * 1000.0
    overlay = draw_debug_overlay(image, self.current_rois, self.current_results, selected_roi_id=selected.id)
    self.find_decode_view.set_frame(overlay)
    self.find_decode_result_box.setPlainText(self._result_text() + f"\nDecode Selected ROI: {elapsed_ms:.1f} ms")


def _selected_roi(self) -> Roi:
    if self.selected_roi_id is not None:
        for roi in self.current_rois:
            if roi.id == self.selected_roi_id:
                return roi
    return self.current_rois[0]
```

Update `run_decode` to call:

```python
self.find_code()
self.decode_selected_roi()
```

Modify `decoder_debugger/overlay.py` signature:

```python
def draw_debug_overlay(
    image: np.ndarray,
    rois: tuple[Roi, ...],
    results: list[CodeResult],
    selected_roi_id: int | None = None,
) -> np.ndarray:
```

Inside its loop, before drawing rectangle:

```python
thickness = 3 if roi.id == selected_roi_id else 2
cv2.rectangle(output, (roi.x, roi.y), (roi.x + roi.width, roi.y + roi.height), color, thickness, cv2.LINE_AA)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_find_code_uses_preprocessed_input_when_selected tests/test_decoder_debugger.py::test_decode_selected_roi_decodes_only_selected_roi -q
```

Expected: `2 passed`.

- [ ] **Step 5: Commit**

```powershell
git add decoder_debugger/main_window.py decoder_debugger/overlay.py tests/test_decoder_debugger.py
git commit -m "Add find and decode workflow controls"
```

## Task 5: Preset Save And Load

**Files:**
- Modify: `decoder_debugger/main_window.py`
- Test: `tests/test_decoder_debugger.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_decoder_debugger.py`:

```python
def test_preprocess_preset_save_and_load_updates_controls(tmp_path):
    app = QApplication.instance() or QApplication([])
    window = DecoderDebuggerWindow()
    preset_path = tmp_path / "preset.json"
    window.channel_select.setCurrentText("HSV Saturation")
    window.threshold_mode.setCurrentText("Manual")
    window.manual_threshold.setValue(88)

    window.save_preprocess_preset(preset_path)
    window.channel_select.setCurrentText("Original")
    window.threshold_mode.setCurrentText("None")
    window.manual_threshold.setValue(128)
    window.load_preprocess_preset(preset_path)

    assert app is not None
    assert window.channel_select.currentText() == "HSV Saturation"
    assert window.threshold_mode.currentText() == "Manual"
    assert window.manual_threshold.value() == 88
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_preprocess_preset_save_and_load_updates_controls -q
```

Expected: FAIL with missing `save_preprocess_preset`.

- [ ] **Step 3: Implement preset methods and buttons**

Add imports in `decoder_debugger/main_window.py`:

```python
from decoder_debugger.preprocess import PreprocessConfig, PreprocessResult, apply_preprocess, config_from_json, config_to_json
```

In `_preprocess_tab`, add buttons under the operator group:

```python
save_preset = QPushButton("Save Preset")
save_preset.clicked.connect(self._save_preprocess_preset_dialog)
load_preset = QPushButton("Load Preset")
load_preset.clicked.connect(self._load_preprocess_preset_dialog)
controls.addWidget(save_preset)
controls.addWidget(load_preset)
```

Add methods:

```python
def save_preprocess_preset(self, path: str | Path) -> None:
    self._preprocess_controls_changed()
    Path(path).write_text(config_to_json(self.preprocess_config), encoding="utf-8")


def load_preprocess_preset(self, path: str | Path) -> None:
    config = config_from_json(Path(path).read_text(encoding="utf-8"))
    self.preprocess_config = config
    self.preview_view_mode.setCurrentText(config.view)
    self.channel_select.setCurrentText(config.channel)
    self.threshold_mode.setCurrentText(config.threshold_mode)
    self.manual_threshold.setValue(config.manual_threshold)
    self._refresh_preprocess_preview()


def _save_preprocess_preset_dialog(self) -> None:
    path, _ = QFileDialog.getSaveFileName(self, "Save Preprocess Preset", "", "JSON (*.json)")
    if path:
        self.save_preprocess_preset(path)


def _load_preprocess_preset_dialog(self) -> None:
    path, _ = QFileDialog.getOpenFileName(self, "Load Preprocess Preset", "", "JSON (*.json)")
    if path:
        self.load_preprocess_preset(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py::test_preprocess_preset_save_and_load_updates_controls -q
```

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```powershell
git add decoder_debugger/main_window.py tests/test_decoder_debugger.py
git commit -m "Add preprocess preset save and load"
```

## Task 6: Full Verification

**Files:**
- Verify all touched files.

- [ ] **Step 1: Run debugger tests**

Run:

```powershell
python -m pytest tests/test_decoder_debugger.py -q
```

Expected: all debugger tests pass.

- [ ] **Step 2: Run full test suite**

Run:

```powershell
python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Launch debugger**

Run:

```powershell
python -m decoder_debugger.main
```

Expected: a standalone window opens with tabs `Preprocess`, `Find / Decode`, and `Batch / Logs`.

- [ ] **Step 4: Commit only if verification changes files**

If any verification changes were required:

```powershell
git add decoder_debugger tests
git commit -m "Stabilize decoder preprocess UI"
```

If no files changed, do not create an empty commit.

## Self-Review

- Spec coverage: Implements top tabs, dedicated preprocessing page, original/preprocessed switching, live preprocessing preview, find/decode page, selected input image, selected ROI decode, and preset save/load. Full ECC200 self-owned decoding remains outside this UI plan by design.
- Plan marker scan: No deferred-work markers are used. Each task has exact files, tests, commands, and code.
- Type consistency: Uses `PreprocessConfig`, `PreprocessResult`, `apply_preprocess`, `config_to_json`, `config_from_json`, `DecoderDebuggerWindow`, `Roi`, and `CodeResult` consistently.
