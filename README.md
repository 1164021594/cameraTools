# Stereo ArUco GUI

Windows desktop tool for calibrating two independent USB cameras with an ArUco GridBoard and preparing stereo ranging output.

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
5. Place the ArUco board in both views and click `Capture Pair`.
6. Capture 30-60 varied pairs across center, edges, near, middle, far, and tilted poses.
7. Click `Calibrate`.
8. Use `Rectified` mode to check that corresponding points lie on the same horizontal line.
9. Use `Depth` mode and click the image to inspect distance after calibration.

Captured images are saved under `data/images`. Calibration output is saved under `data/output`.
