# Precise Click Viewer Design

## Goal

Add precise point selection support to the existing PySide6 stereo camera GUI without sacrificing live-preview smoothness.

## Scope

This design adds three tightly related capabilities:

1. Main preview clicks must resolve to original-frame coordinates, not preview-scaled coordinates.
2. The user must be able to freeze the current preview frame pair.
3. The user must be able to inspect a frozen original frame in a zoomable viewer and click exact source-image coordinates.

This design does not add multi-point annotation management, persistence of click sets, or editing workflows beyond single-click inspection.

## Current Constraints

- Live preview must remain lightweight because dual independent USB cameras are already near their stability limit.
- The main preview currently uses downscaled images via `preview_copy(...)`.
- Existing `ImageView` already maps display positions back to the frame currently set on that widget, but the frame being displayed is often the preview copy, not the original frame.
- Depth click logic currently consumes whatever coordinates arrive from `ImageView`.

## Recommended Approach

Keep the main preview fast and add a second, frozen high-precision path.

### Main Preview

- Continue rendering the live preview with downscaled frames.
- Always keep the latest original left/right frames in `MainWindow`.
- Continue allowing clicks on the main preview.
- Translate main-preview click intent to original-frame coordinates by using the original frame dimensions rather than the downscaled preview dimensions.

### Freeze Mode

- Add a `Freeze Frame` toggle in the capture/result workflow area.
- When activated, snapshot the latest original left/right frames and stop replacing the preview with new live data.
- While frozen, the live capture threads may continue running in the background, but the displayed content and click targets remain fixed to the frozen pair.
- When released, resume normal live preview updates.

### Original Viewer

- Add a lightweight dialog for high-precision inspection of a single frame.
- The dialog shows the original-resolution image, supports wheel zoom and drag pan, and emits original-image click coordinates.
- The dialog is opened explicitly from the main window, using the currently selected frozen frame when available, otherwise the latest original frame.

## Interaction Model

### Fast Path

- User clicks directly in the main preview.
- The click is mapped to original-frame coordinates.
- The point is logged and used by downstream ranging logic.

### Precision Path

- User freezes the current frame pair.
- User opens the original viewer for left or right image.
- User zooms, pans, and clicks precisely.
- The selected point is returned to the main window in original-frame coordinates.

## UI Changes

### Main Window

Add three compact controls near the preview/ranging area:

- `Freeze Frame` toggle button
- `View Left Original`
- `View Right Original`

When frozen:

- Show a visible frozen-state hint in the preview info/status area.
- Preserve the frozen pair until unfreezing or until a new freeze action replaces it.

When no source frame exists:

- Disable original-view buttons.
- Keep the UI stable and show a diagnostic/log message instead of raising.

### Original Viewer Dialog

The dialog should provide:

- Large image viewport
- Wheel zoom
- Drag pan
- Coordinate readout for current click
- Optional crosshair marker for the last selected point

No complex toolbar is needed in the first version.

## Data Flow

1. Camera worker produces original frames.
2. `MainWindow` stores original frames in `latest_left/latest_right`.
3. Preview rendering uses `preview_copy(...)`.
4. `ImageView` click events must resolve against original dimensions when used in live preview.
5. Freeze mode copies `latest_left/latest_right` into frozen-frame storage.
6. Original viewer receives one frozen or latest original frame and returns original-frame coordinates.
7. Main window routes those coordinates into the existing point/distance handling path.

## File Responsibilities

- `stereo_aruco_gui/app/image_view.py`
  Keep preview rendering and click mapping responsibilities. Extend it so the displayed pixmap can differ from the source coordinate space.

- `stereo_aruco_gui/app/main_window.py`
  Own freeze state, original-frame storage, viewer launch, and unified click handling.

- `stereo_aruco_gui/app/original_image_dialog.py`
  New focused dialog for zoom/pan/original-coordinate selection.

- `tests/test_aruco_backend.py`
  Extend with GUI-level behavior tests for coordinate mapping, freeze state, and viewer-launch behavior.

## Error Handling

- If no current frame exists, `Freeze Frame` should do nothing except log a clear message.
- If the user opens the original viewer without a frame, show a non-crashing warning/log message.
- If one side is unavailable in single-camera or partial-camera scenarios, only enable the valid viewer entry.

## Testing Strategy

Add focused tests for:

- Mapping preview clicks to original-frame coordinates when display frame is downscaled.
- Preserving frozen frames while live updates continue.
- Toggling freeze state from the main window.
- Opening original-view actions only when source frames exist.

Manual runtime verification should confirm:

- Live preview remains fluid.
- Clicks in the main preview return original-image coordinates.
- Frozen frames remain stable until released.
- Original viewer zoom/pan/click behavior works on both left and right frames.
