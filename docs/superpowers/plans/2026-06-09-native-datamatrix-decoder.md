# Native DataMatrix Decoder Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the zxing-cpp dependency for DataMatrix with a tested native ECC200 decoder while preserving the current debugger workflow.

**Architecture:** Keep ROI finding, preprocessing, and decoding as separate stages. Implement native decoding under `industrial_code_reader/datamatrix/native.py` with focused helpers for binarization, orientation, grid sampling, codeword extraction, Reed-Solomon correction, and payload decoding. During development, `DataMatrixDecoder` may use zxing-cpp only as an optional fallback until the native path passes real sample tests.

**Tech Stack:** Python, OpenCV, NumPy, pytest.

---

## File Responsibilities

- `industrial_code_reader/datamatrix/roi.py`: candidate ROI generation and ranking. It should favor DataMatrix-like finder patterns over chip/IC textures.
- `industrial_code_reader/datamatrix/native.py`: native ECC200 decoder implementation.
- `industrial_code_reader/datamatrix/decoder.py`: adapter that calls the native decoder and returns `CodeResult`.
- `tests/test_industrial_code_reader.py`: focused tests for ROI ranking and decoder behavior.
- `tests/fixtures/datamatrix/`: small generated and real-world DataMatrix samples used by decoder tests.

## Phase 1: ROI Quality

- [ ] Add sample-based regression tests for the PCB sheet image where the middle DataMatrix is currently missed.
- [ ] Expose ROI quality fields in the debugger result panel: source, area, aspect, edge density, module texture, finder score.
- [ ] Tune `DataMatrixRoiGenerator` to rank square L-finder candidates above chip/IC textures.
- [ ] Verify that all visible DataMatrix codes on the sheet appear before obvious chip false positives.
- [ ] Commit ROI quality changes.

## Phase 2: Native Decoder Skeleton

- [ ] Create `industrial_code_reader/datamatrix/native.py`.
- [ ] Define `NativeDataMatrixDecode` with text, points, bbox, confidence, preprocessing, and diagnostics.
- [ ] Add tests that native decoder returns no result, not a crash, for blank images and chip-like ROIs.
- [ ] Add tests that generated square DataMatrix samples can be binarized and sampled into a module grid.
- [ ] Commit skeleton and sampling tests.

## Phase 3: Grid Localization And Sampling

- [ ] Implement binary image normalization across gray, CLAHE, sharpen, Otsu, and adaptive variants.
- [ ] Detect finder L edges and clocking edges from a candidate ROI.
- [ ] Estimate module count from edge transitions and known ECC200 sizes.
- [ ] Warp or sample the ROI into a square module matrix.
- [ ] Add debugger stage images for binarized ROI, sampled grid, and finder edges.
- [ ] Commit localization and sampling.

## Phase 4: ECC200 Codeword Extraction

- [ ] Implement ECC200 data region traversal for square symbols.
- [ ] Support standard square symbol sizes first: 10x10 through 26x26.
- [ ] Convert sampled modules into raw codewords.
- [ ] Add tests against generated DataMatrix symbols with known payloads.
- [ ] Commit codeword extraction.

## Phase 5: Reed-Solomon Error Correction

- [ ] Implement GF(256) arithmetic using the DataMatrix primitive polynomial.
- [ ] Implement Reed-Solomon syndrome, locator, evaluator, and correction.
- [ ] Add tests using known codeword/error vectors.
- [ ] Integrate correction into native decoder.
- [ ] Commit ECC support.

## Phase 6: Payload Decoding

- [ ] Implement ASCII encodation.
- [ ] Add C40/Text/X12 support only after ASCII tests pass.
- [ ] Add Base256 support for industrial payloads if samples require it.
- [ ] Return decoded text with diagnostics showing symbol size, errors corrected, preprocessing, and confidence.
- [ ] Commit payload decoding.

## Phase 7: Replace zxing-cpp

- [ ] Change `DataMatrixDecoder` to call native decoder first.
- [ ] Keep zxing-cpp fallback behind an explicit option or environment flag during transition.
- [ ] Update tests so core DataMatrix tests no longer monkeypatch zxing-cpp.
- [ ] Remove mandatory zxing-cpp wording from DataMatrix decoder errors.
- [ ] Commit native-first decoder.

## Phase 8: Debugger Integration

- [ ] Show native decoder stages in the decoder debugger.
- [ ] Add per-ROI diagnostics for why native decode failed: no finder, unstable grid, invalid codewords, ECC failure, payload failure.
- [ ] Add a batch test view that reports found/decoded counts per image.
- [ ] Commit debugger diagnostics.

## Verification

- [ ] Run `python -m pytest tests/test_industrial_code_reader.py -q`.
- [ ] Run `python -m pytest tests/test_decoder_debugger.py -q`.
- [ ] Run `python -m pytest -q`.
- [ ] Launch `python -m decoder_debugger.main` and verify the debugger opens.
