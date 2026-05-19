# Stereo ArUco GUI Design

## Goal

Build a Windows-friendly Python desktop application for calibrating two independent USB cameras with an ArUco GridBoard, verifying stereo rectification, and preparing the calibration output for stereo ranging.

## Hardware Context

The product document in `docs/ZV-TX5M-8-GD5.2.pdf` describes a GC5035 camera module with 2592x1944 resolution, 2.27 mm lens, F2.2 aperture, 76 degree diagonal field of view, less than 1.5 percent distortion, and a nominal 30 cm to 2 m capture range. The document does not include stereo baseline, intrinsic parameters, extrinsic parameters, or factory stereo calibration, so the application must perform calibration from captured image pairs.

## Architecture

The application is a single PySide6 GUI project. The GUI owns user interaction and preview rendering, while small backend modules handle camera capture, ArUco detection, calibration, rectification, depth estimation, configuration, and storage. The application stores captured image pairs under `data/images` and calibration outputs under `data/output`.

## Main Workflow

1. User enters or keeps default camera indexes, resolution, and ArUco GridBoard parameters.
2. User opens the two USB cameras and sees live left/right previews.
3. User captures synchronized image pairs with one button.
4. The application detects ArUco markers in the current previews and reports marker counts.
5. User starts stereo calibration from saved pairs.
6. The application saves `stereo_calib.npz` and `stereo_calib.yaml`.
7. User loads or uses the calibration result to view rectified left/right images with horizontal guide lines.
8. User enables disparity/depth view and clicks a point to read approximate distance.

## Defaults

- Left camera index: `0`
- Right camera index: `1`
- Resolution: `1280 x 960`
- FPS: `30`
- ArUco dictionary: `DICT_4X4_50`
- Board: `5 x 7`
- Marker length: `0.030 m`
- Marker separation: `0.010 m`

## Error Handling

The GUI must report camera-open failures, invalid board parameters, missing image pairs, insufficient detected markers, failed calibration, and missing calibration files in the status area instead of crashing.

## Verification

Verification includes Python bytecode compilation, import checks for backend modules, and a lightweight non-interactive calibration unit test using synthetic ArUco board point data where practical. GUI runtime depends on connected cameras, so camera operation is verified manually after launch.
