# Industrial DataMatrix Decoder Design

## Goal

Build a full industrial DataMatrix decoding subsystem focused on small light-colored ECC200 DataMatrix marks on green PCB backgrounds. The subsystem must be debuggable outside the existing stereo camera/calibration application so decoder work can progress without camera, calibration, or main-window UI noise.

## Scope

This design adds two related deliverables:

1. A standalone `decoder_debugger` GUI for decoder development and inspection.
2. A self-owned `industrial_code_reader` DataMatrix pipeline that can eventually decode without relying on `zxing-cpp`.

The first target image domain is the current PCB sample set:

- Small white or light DataMatrix marks.
- Green solder-mask background.
- Possible glare, low contrast, mild blur, small quiet-zone violations, rotation, and perspective distortion.
- Batch images containing multiple boards and multiple codes.

This design does not attempt to solve QR Code, 1D barcode, or general OCR decoding in the first implementation cycle. `zxing-cpp` may remain as a comparison baseline, but the intended product result is the self-owned DataMatrix decoder path.

## Current Context

The repository already contains:

- `industrial_code_reader` as a pure Python algorithm package.
- A DataMatrix ROI generator using Canny contours and HSV saturation MSER candidates.
- A current `DataMatrixDecoder` wrapper that delegates final decoding to `zxing-cpp`.
- GUI integration in `stereo_aruco_gui/app/barcode.py`.
- Tests for ROI generation, engine dispatch, and barcode GUI behavior.

Recent diagnosis showed two important facts:

- Some PCB marks are missed because board edges and highlights merge with the code into oversized contours.
- Some correctly framed marks still fail in `zxing-cpp`, even after aggressive thresholding, scaling, inversion, and quiet-zone padding.

That means the project needs both better industrial localization and a real DataMatrix ECC200 decode path.

## Architecture

### Package Split

`industrial_code_reader`

- Pure algorithm package.
- No PySide dependency.
- Owns ROI generation, finder analysis, grid fitting, bit sampling, ECC200 parsing, Reed-Solomon correction, text decoding, quality metrics, and failure reasons.

`decoder_debugger`

- Standalone PySide6 GUI.
- Runs independently from `stereo_aruco_gui`.
- Loads single images or folders.
- Shows all intermediate decoder stages and timing.
- Used as the main lab for algorithm tuning.

`stereo_aruco_gui`

- Existing camera/calibration application.
- May call the shared decoder package later.
- Does not own the decoder debugging workflow.

## Decoder Pipeline

The industrial DataMatrix pipeline should be a staged, inspectable pipeline:

1. **Image normalization**
   - Convert BGR/gray inputs to stable grayscale and optional channel views.
   - Generate contrast-enhanced and polarity variants.
   - Preserve timing and chosen variant names.

2. **Candidate ROI detection**
   - Keep existing contour and saturation/MSER candidates.
   - Add finder-pattern-aware scoring over candidate squares.
   - Support multi-scale candidate generation for small marks.
   - Deduplicate overlapping candidates.
   - Return rejected candidates with rejection reasons when debugging is enabled.

3. **Finder and geometry analysis**
   - Search for DataMatrix L-shaped solid border and alternating clock tracks.
   - Estimate rotation, perspective, module count, and polarity.
   - Reject candidates with explicit reasons such as `finder_missing`, `clock_track_unstable`, `module_count_unknown`, or `low_contrast`.

4. **Grid fitting**
   - Fit a quadrilateral around the symbol.
   - Warp or sample through a perspective transform.
   - Estimate module centers.
   - Refine grid from clock-track transitions when possible.

5. **Bit matrix sampling**
   - Sample black/white modules using local thresholding.
   - Produce a boolean bit matrix plus confidence per module.
   - Keep a debug visualization of uncertain modules.

6. **ECC200 codeword extraction**
   - Implement DataMatrix ECC200 placement rules.
   - Support square symbols first.
   - Start with common sizes used in the PCB samples, then broaden to the ECC200 square size table.

7. **Error correction**
   - Implement Reed-Solomon correction for DataMatrix ECC200.
   - Report number of corrected errors and correction failure details.

8. **Text decoding**
   - Implement ASCII mode first.
   - Add C40, Text, X12, EDIFACT, and Base256 as needed.
   - Return decoded text, raw codewords, mode transitions, and any parse errors.

## Debugger GUI

The standalone debugger should open with the decoder lab, not a landing page.

### Layout

Left panel:

- Open image.
- Open folder.
- Previous / next image.
- Decoder path: `Industrial`, `ZXing baseline`, `Compare`.
- ROI mode: `Auto`, `Manual`, `Auto + Manual`.
- Run decode.
- Batch run for current folder.

Center:

- Zoomable image viewer.
- Overlay toggles for candidates, selected ROI, finder lines, fitted grid, sampled modules, failed modules, and decoded result boxes.
- Manual ROI drawing by drag rectangle.

Right panel:

- Result text.
- Failure reason.
- Timing breakdown.
- Candidate list.
- Quality metrics.
- Tabs for `ROI`, `Finder`, `Grid`, `Bits`, `Codewords`, `Logs`.

Bottom or secondary panel:

- Image-by-image batch results table.
- Export diagnostics as JSON for later regression tests.

### Interaction Model

- Loading an image runs no decode until the user clicks `Run`, unless an auto-run toggle is enabled.
- Clicking a candidate ROI selects it and updates the inspector.
- Manual ROI can be drawn and decoded directly.
- Compare mode runs both self-owned decoder and `zxing-cpp`, showing disagreements clearly.

## Data Contracts

The algorithm package should return structured objects rather than strings only.

Key result objects:

- `DecodeResult`: decoded text, symbology, success flag, ROI id, timing, quality, failure reason.
- `Candidate`: rectangle/quadrilateral, source, score, rejection reason.
- `FinderResult`: L border, clock tracks, polarity, module estimate.
- `GridResult`: transform, module count, module centers, confidence.
- `BitMatrixResult`: matrix, uncertain modules, sampling thresholds.
- `CodewordResult`: raw codewords, corrected codewords, correction status.

The debugger consumes these objects directly for visualization.

## Error Handling and Diagnostics

Every failure should identify the stage where the decoder failed:

- `no_candidate`
- `candidate_rejected`
- `finder_missing`
- `clock_track_unstable`
- `module_count_unknown`
- `grid_fit_failed`
- `bit_sampling_low_confidence`
- `codeword_placement_failed`
- `reed_solomon_failed`
- `text_decode_failed`

The GUI should display the failure reason and the visual overlay for the failed stage.

## Testing Strategy

Unit tests:

- ECC200 placement rule examples.
- Reed-Solomon correction examples.
- ASCII/C40/Text/Base256 decoding examples.
- Grid sampling on synthetic symbols.
- Finder detection on rotated and polarity-inverted synthetic symbols.

Integration tests:

- Synthetic DataMatrix images generated with known text.
- Cropped PCB ROI samples.
- Full PCB images with multiple boards.
- Regression tests for previously missed samples.

Performance tests:

- Per-image decode timing for large images.
- ROI generation timing.
- Per-candidate timing breakdown.
- Batch folder timing summary.

Manual debugger validation:

- Load known failing PCB sample.
- Confirm candidate ROI appears.
- Inspect finder/grid/bit stages.
- Export diagnostics for future tests.

## Phasing

### Phase 1: Standalone Debugger Shell

- Add `decoder_debugger` entry point.
- Load image/folder.
- Show zoomable image.
- Run current industrial pipeline.
- Show candidate overlays, result text, failure reasons, and timing.

### Phase 2: ECC200 Decode Core

- Implement symbol size table.
- Implement bit matrix placement and codeword extraction.
- Implement Reed-Solomon correction.
- Implement ASCII text decoding.
- Validate on synthetic known-good symbols.

### Phase 3: Finder/Grid/Sampling

- Implement DataMatrix finder analysis.
- Fit quadrilateral and module grid.
- Sample bit matrix from ROI.
- Validate on synthetic rotated, scaled, inverted, and noisy symbols.

### Phase 4: PCB Hardening

- Tune candidate detection and grid sampling for green PCB samples.
- Add failure overlays and quality scores.
- Add regression cases from `docs/barcode` samples where practical.

### Phase 5: Main App Integration

- Keep the standalone debugger as the algorithm lab.
- Update `stereo_aruco_gui` to call the self-owned decoder core when mature.
- Keep `zxing-cpp` as optional baseline/compare mode.

## Acceptance Criteria

- The standalone debugger runs independently with `python -m decoder_debugger.main`.
- A user can load a PCB image, inspect candidate ROIs, run decode, and see timing plus failure diagnostics.
- The algorithm package exposes a pure-Python API that can be tested without GUI dependencies.
- The self-owned ECC200 core decodes synthetic known-good square DataMatrix symbols without `zxing-cpp`.
- The debugger can compare self-owned results against `zxing-cpp` baseline.
- Large image debugging avoids multi-second full-image brute force unless explicitly requested.

## Open Risks

- Fully industrial decoding is a large algorithm effort; low-quality PCB samples may still fail until finder/grid/sampling stages mature.
- Reed-Solomon and ECC200 placement must be implemented carefully and tested against known vectors.
- Some production marks may require better image acquisition, especially if quiet zone is badly violated or modules are physically smeared.
- If pure traditional localization plateaus, a small model locator may be added later without changing the decode core contract.
