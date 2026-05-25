from __future__ import annotations

import cv2
import numpy as np


def plane_distance_between_pixels(
    p1: tuple[int, int],
    p2: tuple[int, int],
    K: np.ndarray,
    D: np.ndarray,
    plane_distance_mm: float,
) -> float:
    if plane_distance_mm <= 0:
        raise ValueError("Plane distance must be greater than 0 mm")

    camera_matrix = np.asarray(K, dtype=np.float64)
    dist_coeffs = np.asarray(D, dtype=np.float64)
    points = np.asarray([[p1], [p2]], dtype=np.float64)
    normalized = cv2.undistortPoints(points, camera_matrix, dist_coeffs).reshape(2, 2)
    plane_points_mm = normalized * float(plane_distance_mm)
    return float(np.linalg.norm(plane_points_mm[1] - plane_points_mm[0]))
