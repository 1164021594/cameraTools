# Stereo ArUco GUI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an integrated Python GUI for two independent USB cameras to capture ArUco GridBoard pairs, calibrate stereo parameters, verify rectification, and inspect distance.

**Architecture:** A PySide6 GUI coordinates small OpenCV backend modules. Images are saved under `data/images`, calibration files under `data/output`, and defaults are persisted in `config.yaml`.

**Tech Stack:** Python, PySide6, OpenCV contrib, NumPy, PyYAML.

---

### Task 1: Project Skeleton

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `config.yaml`
- Create: `stereo_aruco_gui/main.py`
- Create: `stereo_aruco_gui/app/__init__.py`

- [ ] Create dependency and entry files.
- [ ] Add default camera and ArUco configuration.
- [ ] Add README with install and run commands.

### Task 2: Backend Modules

**Files:**
- Create: `stereo_aruco_gui/app/config.py`
- Create: `stereo_aruco_gui/app/storage.py`
- Create: `stereo_aruco_gui/app/aruco_board.py`
- Create: `stereo_aruco_gui/app/calibration.py`
- Create: `stereo_aruco_gui/app/rectification.py`

- [ ] Implement typed configuration loading/saving with PyYAML fallback defaults.
- [ ] Implement image pair and calibration output storage helpers.
- [ ] Implement ArUco dictionary lookup, GridBoard creation, marker detection, and object/image point extraction for common marker IDs.
- [ ] Implement monocular and stereo calibration from saved left/right image pairs.
- [ ] Implement stereo rectification maps, SGBM disparity, and click-to-distance from `Q`.

### Task 3: Camera and GUI

**Files:**
- Create: `stereo_aruco_gui/app/camera_worker.py`
- Create: `stereo_aruco_gui/app/image_view.py`
- Create: `stereo_aruco_gui/app/main_window.py`

- [ ] Implement a Qt camera worker using `cv2.VideoCapture` for two independent USB cameras.
- [ ] Implement image conversion and clickable image display widgets.
- [ ] Implement main window controls for camera parameters, board parameters, capture, detection, calibration, loading calibration, rectified view, disparity view, and distance click display.

### Task 4: Verification

**Files:**
- Create: `tests/test_aruco_backend.py`

- [ ] Add backend tests for dictionary lookup, board creation, config defaults, and storage paths.
- [ ] Run `python -m compileall stereo_aruco_gui tests`.
- [ ] Run `python -m pytest` when pytest is available.
- [ ] Run an import smoke test for `stereo_aruco_gui.main`.

### Self-Review

- Spec coverage: project skeleton, GUI workflow, camera capture, ArUco detection, calibration, rectification, disparity/depth, output storage, and error reporting are covered.
- Placeholder scan: no placeholder implementation steps remain.
- Type consistency: module names and paths are consistent across tasks.
