from __future__ import annotations

import numpy as np

import industrial_code_reader.datamatrix.decoder as dm_decoder_module
from industrial_code_reader.core.engine import CodeReaderEngine
from industrial_code_reader.core.types import CodeResult, DecodeOptions, Roi
from industrial_code_reader.datamatrix.roi import DataMatrixRoiGenerator
from industrial_code_reader.datamatrix.decoder import DataMatrixDecoder


def test_engine_returns_datamatrix_results_with_roi_and_preprocess_name(monkeypatch):
    calls: list[tuple[tuple[int, int, int], str | None]] = []

    class FakeBarcode:
        valid = True
        text = "DM-001"
        format = "DataMatrix"
        position = None

    class FakeZxing:
        class BarcodeFormat:
            DataMatrix = "DataMatrix"

        @staticmethod
        def read_barcodes(image, formats=None):  # noqa: ANN001
            calls.append((image.shape, formats[0] if formats else None))
            if image.shape[0] > 30:
                return [FakeBarcode()]
            return []

    monkeypatch.setattr(dm_decoder_module, "require_zxingcpp", lambda: FakeZxing)

    engine = CodeReaderEngine(decoders=[DataMatrixDecoder()])
    image = np.zeros((12, 12, 3), dtype=np.uint8)

    results = engine.decode(image, DecodeOptions(symbologies=("DataMatrix",), roi_id=7))

    assert results[0].text == "DM-001"
    assert results[0].symbology == "DataMatrix"
    assert results[0].roi_id == 7
    assert results[0].preprocessing in {"gray_3x", "gray_clahe_3x", "gray_sharpen_3x", "gray_otsu_3x", "gray_adaptive_3x"}
    assert len(calls) > 1


def test_datamatrix_decoder_maps_zxing_position_to_points(monkeypatch):
    class Point:
        def __init__(self, x: int, y: int) -> None:
            self.x = x
            self.y = y

    class Position:
        top_left = Point(1, 2)
        top_right = Point(8, 2)
        bottom_right = Point(8, 9)
        bottom_left = Point(1, 9)

    class FakeBarcode:
        valid = True
        text = "PCB-42"
        format = "DataMatrix"
        position = Position()

    class FakeZxing:
        class BarcodeFormat:
            DataMatrix = "DataMatrix"

        @staticmethod
        def read_barcodes(image, formats=None):  # noqa: ANN001
            return [FakeBarcode()]

    monkeypatch.setattr(dm_decoder_module, "require_zxingcpp", lambda: FakeZxing)

    results = DataMatrixDecoder().decode(np.zeros((20, 20, 3), dtype=np.uint8), DecodeOptions())

    assert results[0].points == ((1, 2), (8, 2), (8, 9), (1, 9))
    assert results[0].bbox == (1, 2, 7, 7)
    assert results[0].quality["backend"] == "zxing-cpp"


def test_datamatrix_decoder_native_method_does_not_call_zxing(monkeypatch):
    def fail_require_zxing():  # noqa: ANN001
        raise AssertionError("native decoder should not call zxing")

    monkeypatch.setattr(dm_decoder_module, "require_zxingcpp", fail_require_zxing)

    results = DataMatrixDecoder(method="native").decode(np.zeros((20, 20, 3), dtype=np.uint8), DecodeOptions(return_failures=True))

    assert results
    assert results[0].failure_reason == "Native DataMatrix decoder is not complete yet"
    assert results[0].quality["backend"] == "native"


def test_native_datamatrix_decoder_reads_generated_ascii_symbol():
    import zxingcpp

    barcode = zxingcpp.create_barcode("DM123", zxingcpp.BarcodeFormat.DataMatrix)
    image = np.array(zxingcpp.write_barcode_to_image(barcode, scale=8))

    results = DataMatrixDecoder(method="native").decode(image, DecodeOptions())

    assert results
    assert results[0].text == "DM123"
    assert results[0].quality["backend"] == "native"


def test_native_datamatrix_decoder_reports_failure_for_blank_image():
    image = np.full((80, 80), 255, dtype=np.uint8)

    results = DataMatrixDecoder(method="native").decode(image, DecodeOptions(return_failures=True))

    assert results
    assert results[0].text == ""
    assert results[0].failure_reason == "Native DataMatrix decoder is not complete yet"
    assert results[0].quality["backend"] == "native"
    assert results[0].quality["failures"]


def test_datamatrix_decoder_auto_falls_back_to_zxing_when_native_finds_nothing(monkeypatch):
    class FakeBarcode:
        valid = True
        text = "ZXING-FALLBACK"
        format = "DataMatrix"
        position = None

    class FakeZxing:
        class BarcodeFormat:
            DataMatrix = "DataMatrix"

        @staticmethod
        def read_barcodes(image, formats=None):  # noqa: ANN001
            return [FakeBarcode()] if image.shape[0] > 30 else []

    monkeypatch.setattr(dm_decoder_module, "require_zxingcpp", lambda: FakeZxing)

    results = DataMatrixDecoder(method="auto").decode(np.zeros((12, 12, 3), dtype=np.uint8), DecodeOptions())

    assert results[0].text == "ZXING-FALLBACK"
    assert results[0].quality["backend"] == "zxing-cpp"


def test_engine_decodes_supplied_rois_and_offsets_result_coordinates(monkeypatch):
    decoded_shapes: list[tuple[int, int, int]] = []

    class FakeDecoder:
        symbology = "DataMatrix"

        def decode(self, image, options):  # noqa: ANN001
            decoded_shapes.append(image.shape)
            return [
                CodeResult(
                    text="ROI-CODE",
                    symbology="DataMatrix",
                    roi_id=options.roi_id,
                    bbox=(2, 3, 4, 5),
                    points=((2, 3), (6, 3), (6, 8), (2, 8)),
                    preprocessing="fake",
                )
            ]

    image = np.zeros((100, 120, 3), dtype=np.uint8)
    engine = CodeReaderEngine(decoders=[FakeDecoder()])

    results = engine.decode(image, DecodeOptions(rois=(Roi(id=9, x=10, y=20, width=30, height=40),)))

    assert decoded_shapes == [(40, 30, 3)]
    assert results[0].roi_id == 9
    assert results[0].bbox == (12, 23, 4, 5)
    assert results[0].points == ((12, 23), (16, 23), (16, 28), (12, 28))


def test_engine_assigns_clipped_roi_id_when_decoder_reports_default_roi_id():
    class FakeDecoder:
        symbology = "DataMatrix"

        def decode(self, image, options):  # noqa: ANN001
            return [
                CodeResult(
                    text="ROI-11-CODE",
                    symbology="DataMatrix",
                    roi_id=1,
                    bbox=(1, 2, 3, 4),
                    preprocessing="fake",
                )
            ]

    image = np.zeros((100, 120, 3), dtype=np.uint8)
    engine = CodeReaderEngine(decoders=[FakeDecoder()])

    results = engine.decode(image, DecodeOptions(rois=(Roi(id=11, x=10, y=20, width=30, height=40),)))

    assert results[0].roi_id == 11


def test_datamatrix_roi_generator_finds_dense_square_candidate():
    image = np.full((120, 160), 255, dtype=np.uint8)
    cell = 4
    x0, y0 = 50, 35
    for row in range(10):
        for col in range(10):
            if (row + col) % 2 == 0:
                image[y0 + row * cell : y0 + (row + 1) * cell, x0 + col * cell : x0 + (col + 1) * cell] = 0

    rois = DataMatrixRoiGenerator(min_area=200, padding=4).generate(image)

    assert rois
    best = rois[0]
    assert best.x <= x0
    assert best.y <= y0
    assert best.x + best.width >= x0 + 40
    assert best.y + best.height >= y0 + 40
    assert best.quality["edge_density"] > 0
    assert best.quality["area"] > 0
    assert best.quality["aspect"] > 0


def test_datamatrix_roi_generator_prioritizes_internal_module_texture_over_blank_box():
    image = np.full((160, 220), 255, dtype=np.uint8)

    cv2 = __import__("cv2")
    cv2.rectangle(image, (150, 35), (166, 51), 0, 1)

    cell = 4
    x0, y0 = 45, 70
    for row in range(12):
        for col in range(12):
            if row == 0 or col == 0 or (row + col) % 2 == 0:
                image[y0 + row * cell : y0 + (row + 1) * cell, x0 + col * cell : x0 + (col + 1) * cell] = 0

    rois = DataMatrixRoiGenerator(min_area=80, padding=3).generate(image)

    assert rois
    best = rois[0]
    assert best.x <= x0
    assert best.y <= y0
    assert best.x + best.width >= x0 + 12 * cell
    assert best.y + best.height >= y0 + 12 * cell
    assert best.quality["module_texture"] > 0.10


def test_datamatrix_roi_generator_keeps_l_finder_candidate_with_chip_like_texture_present():
    cv2 = __import__("cv2")
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    image[:, :] = (0, 120, 90)

    cv2.rectangle(image, (45, 55), (88, 98), (95, 75, 25), -1)
    for offset in range(0, 44, 6):
        cv2.line(image, (45 + offset, 55), (45 + offset, 98), (170, 150, 50), 1)
        cv2.line(image, (45, 55 + offset), (88, 55 + offset), (170, 150, 50), 1)

    x0, y0, cell = 150, 60, 5
    for row in range(12):
        for col in range(12):
            is_l_finder = row == 11 or col == 0
            is_clock = (row == 0 and col % 2 == 0) or (col == 11 and row % 2 == 0)
            is_data = (row * 3 + col * 5) % 7 in {0, 2, 3}
            light = not (is_l_finder or is_clock or is_data)
            color = (225, 225, 225) if light else (0, 80, 55)
            image[y0 + row * cell : y0 + (row + 1) * cell, x0 + col * cell : x0 + (col + 1) * cell] = color

    rois = DataMatrixRoiGenerator(min_area=80, padding=4).generate(image)

    assert rois
    matching = [
        roi
        for roi in rois
        if roi.x <= x0 + 12 * cell and roi.y <= y0 + 12 * cell and roi.x + roi.width >= x0 and roi.y + roi.height >= y0
    ]
    assert matching


def test_datamatrix_roi_generator_finds_colored_pcb_code_connected_to_large_board_edge():
    cv2 = __import__("cv2")
    image = np.full((180, 260, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (230, 150), (0, 120, 90), -1)
    cv2.rectangle(image, (18, 18), (232, 152), (0, 255, 160), 6)

    x0, y0, cell = 150, 80, 5
    for row in range(14):
        for col in range(14):
            color = (230, 230, 230) if row == 0 or col == 0 or (row + col) % 2 == 0 else (0, 110, 70)
            image[y0 + row * cell : y0 + (row + 1) * cell, x0 + col * cell : x0 + (col + 1) * cell] = color

    rois = DataMatrixRoiGenerator(min_area=80, padding=6).generate(image)

    assert rois
    best = rois[0]
    assert best.x <= x0
    assert best.y <= y0
    assert best.x + best.width >= x0 + 14 * cell
    assert best.y + best.height >= y0 + 14 * cell
    assert best.quality["source"] == "mser_saturation"


def test_engine_skips_expensive_full_image_fallback_when_auto_rois_find_nothing(monkeypatch):
    class FailIfCalledDecoder:
        symbology = "DataMatrix"

        def decode(self, image, options):  # noqa: ANN001
            raise AssertionError("full image decode should not run when auto ROI detection finds nothing")

    monkeypatch.setattr("industrial_code_reader.core.engine._generate_datamatrix_rois", lambda image: ())

    image = np.full((3000, 4000, 3), 255, dtype=np.uint8)
    engine = CodeReaderEngine(decoders=[FailIfCalledDecoder()])

    results = engine.decode(image, DecodeOptions(auto_rois=True, return_failures=True))

    assert results == []


def test_engine_can_decode_auto_generated_datamatrix_rois():
    class FakeDecoder:
        symbology = "DataMatrix"

        def decode(self, image, options):  # noqa: ANN001
            if image.shape[0] >= 30 and image.shape[1] >= 30:
                return [CodeResult(text="AUTO-ROI", symbology="DataMatrix", roi_id=options.roi_id, bbox=(1, 1, 5, 5))]
            return []

    image = np.full((120, 160), 255, dtype=np.uint8)
    cell = 4
    for row in range(10):
        for col in range(10):
            if (row + col) % 2 == 0:
                image[40 + row * cell : 40 + (row + 1) * cell, 60 + col * cell : 60 + (col + 1) * cell] = 0
    engine = CodeReaderEngine(decoders=[FakeDecoder()])

    results = engine.decode(image, DecodeOptions(auto_rois=True))

    assert results
    assert results[0].text == "AUTO-ROI"
    assert results[0].bbox[0] >= 0
    assert results[0].bbox[1] >= 0


def test_engine_returns_roi_failure_diagnostics_when_enabled():
    class FailingDecoder:
        symbology = "DataMatrix"

        def decode(self, image, options):  # noqa: ANN001
            return []

    image = np.full((120, 160), 255, dtype=np.uint8)
    cell = 4
    for row in range(10):
        for col in range(10):
            if (row + col) % 2 == 0:
                image[40 + row * cell : 40 + (row + 1) * cell, 60 + col * cell : 60 + (col + 1) * cell] = 0
    engine = CodeReaderEngine(decoders=[FailingDecoder()])

    results = engine.decode(image, DecodeOptions(auto_rois=True, return_failures=True))

    assert results
    assert results[0].text == ""
    assert results[0].failure_reason == "No DataMatrix decoded in ROI"
    assert results[0].quality["roi"]["edge_density"] >= 0
    assert results[0].quality["roi"]["area"] > 0
