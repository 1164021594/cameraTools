from __future__ import annotations

import numpy as np

import decoder_debugger.image_io as image_io_module
from decoder_debugger.image_io import list_image_files, read_image_color
from decoder_debugger.overlay import draw_debug_overlay
from industrial_code_reader.core.types import CodeResult, Roi

import os

from PySide6.QtWidgets import QApplication

import decoder_debugger.main_window as debugger_window_module
import decoder_debugger.main as debugger_main_module
from decoder_debugger.main_window import DecoderDebuggerWindow
from decoder_debugger.preprocess import PreprocessConfig, apply_preprocess, config_from_json, config_to_json

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


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


def test_decoder_debugger_has_preprocess_find_decode_and_logs_tabs():
    app = QApplication.instance() or QApplication([])
    window = DecoderDebuggerWindow()

    tab_names = [window.tabs.tabText(index) for index in range(window.tabs.count())]

    assert app is not None
    assert tab_names == ["Preprocess", "Find / Decode", "Batch / Logs"]
    assert window.preview_view_mode.currentText() == "Original"
    assert window.find_input_view.currentText() == "Original"


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


def test_preprocess_numeric_parameters_update_preview_immediately(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    image_path = tmp_path / "sample.bmp"
    image_path.write_bytes(b"fake")
    frame = np.zeros((20, 30, 3), dtype=np.uint8)
    monkeypatch.setattr(debugger_window_module, "read_image_color", lambda path: frame.copy())
    window = DecoderDebuggerWindow()

    window.load_image_path(image_path)
    window.preview_view_mode.setCurrentText("Preprocessed")
    window.scale_factor.setValue(2.0)

    assert app is not None
    assert window.preprocess_config.view == "Preprocessed"
    assert window.preprocess_config.scale_factor == 2.0
    assert window.image_view._last_frame is not None
    assert window.image_view._last_frame.shape[:2] == (40, 60)


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


def test_preprocess_preset_save_and_load_updates_controls(tmp_path):
    app = QApplication.instance() or QApplication([])
    preset_path = tmp_path / "preset.json"
    window = DecoderDebuggerWindow()
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
