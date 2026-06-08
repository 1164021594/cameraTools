from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def read_image_color(path: str | Path) -> np.ndarray | None:
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is not None:
        return image
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def list_image_files(folder: str | Path) -> list[Path]:
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []
    return sorted(path for path in root.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
