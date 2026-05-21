# Precise Click Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add precise original-coordinate clicking, freeze-frame preview, and an original-image zoom viewer to the stereo camera GUI.

**Architecture:** Keep live preview lightweight by continuing to render downscaled frames while preserving original frames in `MainWindow`. Extend `ImageView` so display pixmap size and source-coordinate size can differ, and add a small dedicated dialog for frozen original-frame inspection.

**Tech Stack:** Python, PySide6, OpenCV, NumPy, pytest.

---

### Task 1: Preview Coordinate Mapping

**Files:**
- Modify: `stereo_aruco_gui/app/image_view.py`
- Test: `tests/test_aruco_backend.py`

- [ ] Add tests for click mapping against source dimensions larger than the displayed frame.
- [ ] Run the new image-view click-mapping tests and confirm they fail before implementation.
- [ ] Extend `ImageView` with an explicit source-size override used for coordinate mapping.
- [ ] Re-run the image-view click-mapping tests and confirm they pass.

### Task 2: Freeze State In Main Window

**Files:**
- Modify: `stereo_aruco_gui/app/main_window.py`
- Test: `tests/test_aruco_backend.py`

- [ ] Add tests covering freeze toggling, frozen-frame retention, and disabled behavior when no frame exists.
- [ ] Run the targeted freeze tests and confirm they fail before implementation.
- [ ] Add frozen left/right frame state and freeze controls to `MainWindow`.
- [ ] Update preview refresh logic so frozen frames remain displayed while live capture continues.
- [ ] Re-run the targeted freeze tests and confirm they pass.

### Task 3: Original Image Viewer Dialog

**Files:**
- Create: `stereo_aruco_gui/app/original_image_dialog.py`
- Modify: `stereo_aruco_gui/app/main_window.py`
- Test: `tests/test_aruco_backend.py`

- [ ] Add tests for launching original-view actions only when a source frame exists.
- [ ] Run the viewer-launch tests and confirm they fail before implementation.
- [ ] Implement a minimal zoom/pan original-image dialog with original-coordinate click reporting.
- [ ] Wire left/right original-view buttons into `MainWindow`.
- [ ] Re-run the viewer-launch tests and confirm they pass.

### Task 4: Unified Click Handling

**Files:**
- Modify: `stereo_aruco_gui/app/main_window.py`
- Test: `tests/test_aruco_backend.py`

- [ ] Add tests verifying main-preview clicks and dialog clicks both update point state using original coordinates.
- [ ] Run the targeted click-handling tests and confirm they fail before implementation.
- [ ] Route both preview clicks and original-view dialog clicks through one shared point-handling path.
- [ ] Re-run the targeted click-handling tests and confirm they pass.

### Task 5: Verification

**Files:**
- Modify if needed: `docs/camera-troubleshooting.md`
- Test: `tests/test_aruco_backend.py`

- [ ] Run `pytest -q`.
- [ ] Run `python -m compileall stereo_aruco_gui tests`.
- [ ] If runtime behavior changes need documentation, update the troubleshooting notes for frozen preview and original-coordinate clicking.
