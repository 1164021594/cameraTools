# Decoder Debugger Preprocess UI Design

## Goal

Upgrade the standalone DataMatrix decoder debugger into a stage-based image processing lab where preprocessing can be tuned independently before running code finding and decoding.

The debugger should let the user adjust preprocessing operators, immediately inspect the effect image, then use either the original image or the preprocessed image as input for ROI search and decode.

## Scope

This design updates `decoder_debugger`, not the existing camera/calibration GUI.

In scope:

- Top-level tab navigation.
- A dedicated preprocessing page.
- A combined find/decode page.
- Original/preprocessed view switching.
- Live preprocessing preview.
- Operator-level parameter controls.
- Preset save/load for preprocessing parameters.
- ROI finding and selected-ROI decoding using either original or preprocessed input.

Out of scope for this UI phase:

- Full self-owned ECC200 implementation.
- Training or integrating a neural network locator.
- Full batch reporting beyond a lightweight `Batch / Logs` page shell.

## Navigation

The debugger uses three top-level tabs:

1. `Preprocess`
2. `Find / Decode`
3. `Batch / Logs`

The first screen after image load remains useful immediately. The default viewer state is the original image, not a processed view.

## Preprocess Tab

The `Preprocess` tab is the main image-processing workbench.

### Layout

Left panel:

- A vertical list of preprocessing operators.
- Each operator is collapsible.
- Clicking an operator opens its parameter controls under that operator.
- Operators can be enabled/disabled where applicable.

Right panel:

- A large zoomable preview.
- A view switch between `Original` and `Preprocessed`.
- The default view is `Original`.
- Any parameter change recomputes and displays the current preprocessing result when `Preprocessed` is selected.

### Operators

Initial operators:

- `Channel Select`
  - `Original`
  - `Gray`
  - `B`
  - `G`
  - `R`
  - `HSV Saturation`
  - `HSV Value`

- `CLAHE`
  - enabled
  - clip limit
  - tile size

- `Blur`
  - mode: none, Gaussian, median
  - kernel size

- `Sharpen`
  - enabled
  - strength
  - radius

- `Threshold`
  - mode: none, Otsu, adaptive, manual
  - adaptive block size
  - adaptive C
  - manual threshold value
  - invert

- `Morphology`
  - operation: none, erode, dilate, open, close
  - kernel size
  - iterations

- `Scale / Padding`
  - scale factor
  - interpolation
  - quiet-zone padding

### Presets

The preprocessing panel supports:

- Save preset to JSON.
- Load preset from JSON.
- Reset to defaults.

Preset files store only parameter values, not image data.

## Find / Decode Tab

The `Find / Decode` tab combines ROI search and decoding.

### Layout

Left panel:

- Input view selector: `Original` or `Preprocessed`.
- ROI mode: `Auto`, `Manual`, `Both`.
- ROI search parameters:
  - max ROI count
  - min area
  - max area
  - score threshold
- `Find Code` button.
- `Decode Selected ROI` button.

Center viewer:

- Large zoomable image.
- Defaults to original image.
- Can switch to preprocessed image.
- Shows ROI overlays after `Find Code`.
- Supports selecting an ROI by clicking its box.

Right inspector:

- ROI list.
- Selected ROI crop preview.
- Decode result.
- Failure reason.
- Timing summary.
- Basic quality metrics.

### Workflow

1. User loads an image.
2. User opens `Preprocess`.
3. User adjusts operators and inspects live preview.
4. User switches to `Find / Decode`.
5. User selects input view: original or preprocessed.
6. User clicks `Find Code`.
7. Debugger draws ROI boxes.
8. User selects one ROI.
9. User clicks `Decode Selected ROI`.
10. Debugger shows result or precise failure reason.

## Data Model

Add a preprocessing configuration object independent of GUI widgets.

Required model:

- `PreprocessConfig`
  - channel
  - clahe settings
  - blur settings
  - sharpen settings
  - threshold settings
  - morphology settings
  - scale/padding settings

Add a preprocessing result object:

- `PreprocessResult`
  - original image reference
  - output image
  - stage images by name
  - timing by operator
  - config used

The GUI should read/write this model, not scatter processing state across widgets.

## Error Handling

- If no image is loaded, preprocessing controls remain usable but preview shows `No image loaded`.
- If an operator combination fails, the inspector shows the operator name and exception text.
- If `Find Code` returns no candidates, the viewer stays on the selected input image and the inspector shows `No ROI candidates`.
- If decoding fails, the selected ROI remains visible and the inspector shows the failure reason.

## Testing Strategy

Unit tests:

- `PreprocessConfig` default values.
- Applying each operator to a synthetic image.
- Preset save/load round trip.
- Original/preprocessed input selection for ROI finding.
- Decode selected ROI uses the currently selected ROI and selected input image.

GUI smoke tests:

- Main window exposes three tabs.
- Preprocess tab updates preview after parameter change.
- Find/decode tab defaults to original view.
- Clicking `Find Code` populates ROI list.
- Clicking `Decode Selected ROI` updates inspector.

Manual validation:

- Load PCB sample.
- Tune threshold/morphology and watch live preview.
- Switch to find/decode.
- Run ROI search on original and preprocessed views.
- Select ROI 11-style candidate and inspect failure.

## Acceptance Criteria

- `python -m decoder_debugger.main` opens the standalone debugger.
- Top-level tabs are visible: `Preprocess`, `Find / Decode`, `Batch / Logs`.
- Preprocess tab has collapsible operator controls on the left.
- Preview defaults to original image.
- User can switch preview to preprocessed image.
- Parameter changes update the preprocessed preview.
- Find/decode tab can run ROI search on original or preprocessed input.
- ROI boxes are drawn after `Find Code`.
- Selected ROI can be decoded separately.
- Inspector shows decode result, failure reason, and timing.
- Presets can be saved and loaded as JSON.
