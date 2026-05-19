# Upper Computer UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rework the PySide6 stereo camera application into a preview-first upper-computer UI with capture, calibration, ranging, and light diagnostics.

**Architecture:** Keep OpenCV/camera/calibration logic in backend modules and keep the main Qt window responsible for composition and signal wiring. Add small pure helpers for UI state and storage operations so the important behavior is testable without launching Qt.

**Tech Stack:** Python, PySide6, OpenCV contrib, NumPy, PyYAML, pytest.

---

### Task 1: Storage And UI State Helpers

**Files:**
- Create: `stereo_aruco_gui/app/ui_state.py`
- Modify: `stereo_aruco_gui/app/storage.py`
- Test: `tests/test_aruco_backend.py`

- [x] **Step 1: Write failing tests**

Add tests for deleting the latest captured pair, same-camera warnings, pair count labels, and calibration mode gating:

```python
from stereo_aruco_gui.app.storage import delete_last_image_pair
from stereo_aruco_gui.app.ui_state import camera_selection_warning, pair_count_text, view_mode_enabled


def test_delete_last_image_pair_removes_highest_index_pair(tmp_path):
    left = np.zeros((24, 32, 3), dtype=np.uint8)
    right = np.full((24, 32, 3), 255, dtype=np.uint8)
    save_image_pair(tmp_path, left, right)
    save_image_pair(tmp_path, left, right)

    deleted = delete_last_image_pair(tmp_path)

    assert deleted is not None
    assert deleted.index == 2
    assert [pair.index for pair in list_image_pairs(tmp_path)] == [1]


def test_camera_selection_warning_blocks_duplicate_stereo_indexes():
    assert camera_selection_warning(0, 0, single_mode=False) == "Left and right cameras must be different."
    assert camera_selection_warning(0, 0, single_mode=True) is None


def test_pair_count_text_includes_recommended_target():
    assert pair_count_text(24) == "Pairs: 24 / recommended 30+"


def test_view_mode_enabled_requires_calibration_for_rectified_and_depth():
    assert view_mode_enabled("Live Preview", has_calibration=False) is True
    assert view_mode_enabled("ArUco Detection", has_calibration=False) is True
    assert view_mode_enabled("Rectified Preview", has_calibration=False) is False
    assert view_mode_enabled("Depth / Distance", has_calibration=False) is False
```

- [x] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_aruco_backend.py -q`

Expected: FAIL because `delete_last_image_pair` and `ui_state` do not exist yet.

- [x] **Step 3: Implement helpers**

Add `delete_last_image_pair(root)` to remove the highest indexed pair from both `left` and `right` directories. Add `ui_state.py` with `camera_selection_warning`, `pair_count_text`, and `view_mode_enabled`.

- [x] **Step 4: Run tests to verify helpers pass**

Run: `pytest tests/test_aruco_backend.py -q`

Expected: PASS.

### Task 2: Preview-First Main Window

**Files:**
- Modify: `stereo_aruco_gui/app/main_window.py`

- [x] **Step 1: Replace the two-column layout**

Make the central widget a grid with a left sidebar, main preview area, and bottom status/log strip. Keep existing `ImageView` widgets and camera worker logic.

- [x] **Step 2: Split sidebar groups**

Create groups for Camera Connection, Capture, Calibration, Result And Ranging, and Light Diagnostics. Move existing controls into these groups and add Delete Last Pair and Open Capture Directory.

- [x] **Step 3: Add view mode switching**

Replace independent Rectified/Depth checkboxes with a mode combo containing Live Preview, ArUco Detection, Rectified Preview, and Depth / Distance. Keep horizontal guide toggling as a separate checkbox.

- [x] **Step 4: Add status metrics**

Add labels for left state, right state, FPS, pair count, calibration summary, baseline, last distance, and diagnostic summary. Update them from existing camera stats, capture, calibration, and click handlers.

- [x] **Step 5: Add light diagnostics**

Block duplicate left/right camera selection in stereo mode before opening. Convert worker status messages into concise diagnostic label text and log messages.

### Task 3: Verification

**Files:**
- No planned production file changes unless verification reveals defects.

- [x] **Step 1: Run unit tests**

Run: `pytest -q`

Expected: PASS.

- [x] **Step 2: Launch smoke test**

Run: `python -m stereo_aruco_gui.main`

Expected: Main window opens without import or construction errors. Manual camera verification still requires hardware.

