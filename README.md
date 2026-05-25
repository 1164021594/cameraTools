# Stereo ArUco GUI

Windows desktop tool for calibrating two independent USB cameras with an ArUco/AprilTag GridBoard and preparing stereo ranging output.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

```powershell
python -m stereo_aruco_gui.main
```

## Basic Workflow

1. Click `Scan Cameras` to enumerate readable USB camera indexes.
2. Select the left and right cameras from the dropdowns. Defaults are `0` and `1`.
3. Set resolution and ArUco GridBoard parameters. Start with `640 x 480`; increase resolution only after live preview is stable.
4. Click `Open Cameras`.
5. Place the board in both views and click `Capture Pair`.
6. Capture 30-60 varied pairs across center, edges, near, middle, far, and tilted poses.
7. Click `Start Calibration`.
8. Use `Rectified Preview` mode to check that corresponding points lie on the same horizontal line.
9. Use `Depth / Distance` mode and click the image to inspect stereo depth after stereo calibration.
10. Use `2D Measurement` mode with `mono_calib.npz`, manually enter `Plane distance mm`, then click two image points to measure distance on that plane.
11. Use `Barcode Detection` mode to decode Code 39, Code 128, Codabar, EAN, ITF25, Code 93, QR Code, and DataMatrix from the current camera view.

Captured images are saved under `data/images`. Calibration output is saved under `data/output`.

Single-camera mode can also capture and calibrate one camera. Single-camera images are saved under `data/images/single`, and the output is saved separately as `data/output/mono_calib.npz` and `data/output/mono_calib.yaml`. Monocular calibration does not produce stereo baseline or `Q`, so it cannot be used for `Depth / Distance`.

`2D Measurement` uses the monocular `K` and `D` from `mono_calib.npz`. The `Plane distance mm` value is the vertical working distance from the camera to the measured plane. This mode assumes the measured plane is parallel to the camera image plane; tilt compensation is not included yet.

`Barcode Detection` uses `zxing-cpp`. If this mode reports that the dependency is missing, run `python -m pip install -r requirements.txt`. For stable decoding, select the expected barcode type instead of `All` when possible, keep the barcode in focus, avoid overexposure, and make sure the smallest bars/modules occupy several pixels.

## Calibration Board

The current board photo `data/images/标定板信息.jpg` shows:

- Dictionary: `DICT_APRILTAG_36h11`
- Printed grid text: `5 x 7`
- OpenCV board model used by this project: `markers_x = 7`, `markers_y = 5`, `board_mirror_x = true`
- Tag size: `80 mm`, use `0.08 m`
- Tag spacing: `24 mm`, use `0.024 m`

For this AprilTag board, the detector uses `markerBorderBits = 2`. If this is wrong, OpenCV may detect zero tags even when the board is visible.

Do not enter this board as OpenCV `5 x 7`. The captured images show that the tag IDs match a `7 x 5` row-major board with the X axis mirrored. With the wrong `5 x 7` model, all tags can still be detected, but the object points are assigned to the wrong physical positions and calibration error becomes extremely large.

During `Start Calibration`, the GUI shows each captured image pair in the left/right preview and prints the current detection status on the image: pair number, left/right marker counts, common corner count, and whether the pair is valid. After calibration, use the `Calibration Pair` dropdown or `Prev Pair` / `Next Pair` buttons above the preview to review every captured pair and find bad images. The result area still shows left error, right error, stereo error, baseline, and quality label.

Pair quality labels:

- `VALID`: `40+` common corners, used for final calibration.
- `WEAK`: `16-39` common corners, shown in review but skipped by final calibration.
- `INVALID`: fewer than `16` common corners, shown in review and skipped.

Current saved image set status:

- `43` readable pairs under `data/images`
- `39` valid detected pairs after strict filtering
- Weak pairs skipped: `0007.png`, `0010.png`, `0011.png`, `0013.png`
- With the corrected board model, current calibration is `left error 1.5237`, `right error 1.5380`, `stereo error 1.5336`, quality `CHECK`.

## Troubleshooting

Camera stability notes are recorded in [`docs/camera-troubleshooting.md`](docs/camera-troubleshooting.md).
Calibration, depth estimation, disparity parameters, ROI display, and Raw/Filtered preview behavior are documented in [`docs/stereo-calibration-depth-notes.md`](docs/stereo-calibration-depth-notes.md).
