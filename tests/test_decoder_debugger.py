from __future__ import annotations

import numpy as np

import decoder_debugger.image_io as image_io_module
from decoder_debugger.image_io import list_image_files, read_image_color
from decoder_debugger.overlay import draw_debug_overlay
from industrial_code_reader.core.types import CodeResult, Roi


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
