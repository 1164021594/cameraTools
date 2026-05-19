from __future__ import annotations


CALIBRATION_REQUIRED_MODES = {"Rectified Preview", "Depth / Distance"}


def camera_selection_warning(left_index: int, right_index: int, single_mode: bool) -> str | None:
    if not single_mode and left_index == right_index:
        return "Left and right cameras must be different."
    return None


def pair_count_text(count: int, recommended_minimum: int = 30) -> str:
    return f"Pairs: {count} / recommended {recommended_minimum}+"


def view_mode_enabled(mode: str, has_calibration: bool) -> bool:
    return has_calibration or mode not in CALIBRATION_REQUIRED_MODES
