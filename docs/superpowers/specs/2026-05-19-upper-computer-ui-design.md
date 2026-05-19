# Upper Computer UI Design

## Goal

Design the first version of the stereo camera upper-computer UI around a preview-first workflow. The application should let a user connect two cameras, preview both streams, capture calibration image pairs, run ArUco stereo calibration, verify rectification, and perform basic distance validation from disparity.

The first version targets a calibration-and-ranging tool with light diagnostics. It should not become a full engineering console.

## Layout

Use a desktop layout with three stable regions:

- Left control sidebar for connection, capture, calibration, ranging, and diagnostics.
- Main preview area for camera frames and result views.
- Bottom status/log strip for current state, metrics, and recent messages.

The main window should prioritize the preview area. The sidebar should be compact but complete enough that the user can perform the full workflow without opening extra windows.

## Sidebar Groups

### Camera Connection

Controls:

- Scan cameras.
- Select left camera.
- Select right camera.
- Set resolution.
- Set frame rate.
- Open or close cameras.
- Save configuration.

State and validation:

- Show whether each camera is connected.
- Show current or measured FPS when available.
- Warn when the left and right camera indexes are the same.
- Warn when a camera cannot open, returns black frames, or fails to read.
- Warn when the requested resolution or frame rate appears not to take effect.

### Capture

Controls:

- Capture one synchronized image pair from the current left and right frames.
- Delete the last captured pair, if available.
- Open or show the capture directory.

State:

- Show captured pair count.
- Show a recommended target range, such as 30 or more pairs.
- Show the last saved pair path or filename.
- Disable capture until both cameras have valid frames.

### Calibration

Controls:

- Select ArUco dictionary.
- Set board marker columns and rows.
- Set marker length.
- Set marker separation.
- Start calibration.
- Load existing calibration output.

State:

- Show left reprojection error.
- Show right reprojection error.
- Show stereo calibration error.
- Show baseline.
- Show calibration output path.
- Disable camera selection and capture while calibration is running.
- Report when captured pairs are insufficient or not enough valid markers are detected.

### Result And Ranging

Controls:

- Switch to rectified preview.
- Toggle horizontal guide lines.
- Switch to depth/disparity view.
- Enable click-to-measure behavior.

State:

- Disable rectified and depth views until calibration is available.
- Show clicked image coordinates.
- Show measured distance in meters.
- Show an invalid-depth message when disparity at the clicked point is unavailable.

### Light Diagnostics

Diagnostics should be integrated into the connection group and bottom log, not presented as a separate complex page.

The first version should surface these issues:

- Camera occupied by another application.
- Camera cannot open.
- Read frame failure.
- Black or near-black frames.
- Left and right camera set to the same index.
- USB bandwidth or resolution may be too high.
- Requested resolution or FPS not applied by the device.
- Too few captured pairs.
- Too few valid marker detections.

## Main Preview Area

Use a segmented mode switch above the preview:

- Live Preview
- ArUco Detection
- Rectified Preview
- Depth / Distance

Live Preview shows the left and right camera frames side by side.

ArUco Detection shows the same side-by-side frames with detected markers overlaid.

Rectified Preview shows rectified left and right frames with optional horizontal guide lines.

Depth / Distance shows a disparity or colorized depth preview and keeps click-to-measure enabled when calibration data exists.

Unavailable modes should be disabled or show a clear inline requirement, for example "calibration required".

## Bottom Status And Log

The bottom strip should display compact, scan-friendly status:

- Left camera state.
- Right camera state.
- Current FPS.
- Captured pair count.
- Calibration error summary.
- Baseline.
- Last clicked distance.
- Recent log messages.

Logs should be short and operational. They should help the user recover from camera and calibration issues without reading a separate debug console.

## Interaction Flow

1. User scans cameras.
2. User selects left and right cameras.
3. User sets resolution and FPS.
4. User opens cameras.
5. App validates that both cameras are producing usable frames.
6. User captures varied image pairs.
7. App updates pair count and last saved path.
8. User sets ArUco board parameters.
9. User starts calibration.
10. App shows errors, baseline, and output path.
11. User switches to rectified preview to inspect horizontal alignment.
12. User switches to depth view and clicks a point to validate distance.

## Error Handling

Errors should be shown in three places according to severity:

- Inline near the relevant control for recoverable setup issues.
- Bottom log for chronological operational messages.
- Modal dialog only for blocking failures where the requested action cannot continue.

Examples:

- Same camera selected for left and right: inline warning and disable opening.
- Camera returns black frames: inline warning and log suggestion to check exposure, lens cover, USB bandwidth, or lower resolution.
- Insufficient calibration pairs: modal or prominent inline message when calibration is requested.
- Invalid clicked depth: bottom status text, no modal.

## Testing

Implementation should include focused tests for non-visual logic:

- Camera selection validation.
- Configuration loading and saving.
- Capture count and image-pair listing.
- Calibration precondition errors.
- Depth click handling when calibration or disparity is unavailable.

Manual verification should cover:

- Opening the UI.
- Scanning cameras.
- Opening two camera streams.
- Capturing a pair.
- Running calibration with valid data.
- Switching preview modes.
- Clicking to measure distance after calibration.

