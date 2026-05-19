from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass(frozen=True)
class ImagePair:
    index: int
    left_path: Path
    right_path: Path


def ensure_image_dirs(root: Path | str) -> tuple[Path, Path]:
    base = Path(root)
    left = base / "left"
    right = base / "right"
    left.mkdir(parents=True, exist_ok=True)
    right.mkdir(parents=True, exist_ok=True)
    return left, right


def list_image_pairs(root: Path | str) -> list[ImagePair]:
    left_dir, right_dir = ensure_image_dirs(root)
    pairs: list[ImagePair] = []
    for left_path in sorted(left_dir.glob("*.png")):
        right_path = right_dir / left_path.name
        if right_path.exists():
            try:
                index = int(left_path.stem)
            except ValueError:
                index = len(pairs) + 1
            pairs.append(ImagePair(index=index, left_path=left_path, right_path=right_path))
    return pairs


def next_pair_index(root: Path | str) -> int:
    pairs = list_image_pairs(root)
    return max((pair.index for pair in pairs), default=0) + 1


def save_image_pair(root: Path | str, left_frame: np.ndarray, right_frame: np.ndarray) -> ImagePair:
    left_dir, right_dir = ensure_image_dirs(root)
    index = next_pair_index(root)
    name = f"{index:04d}.png"
    left_path = left_dir / name
    right_path = right_dir / name
    if not cv2.imwrite(str(left_path), left_frame):
        raise RuntimeError(f"Failed to save {left_path}")
    if not cv2.imwrite(str(right_path), right_frame):
        raise RuntimeError(f"Failed to save {right_path}")
    return ImagePair(index=index, left_path=left_path, right_path=right_path)


def delete_last_image_pair(root: Path | str) -> ImagePair | None:
    pairs = list_image_pairs(root)
    if not pairs:
        return None

    pair = max(pairs, key=lambda item: item.index)
    pair.left_path.unlink(missing_ok=True)
    pair.right_path.unlink(missing_ok=True)
    return pair


def ensure_output_dir(root: Path | str) -> Path:
    output_dir = Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_calibration_npz(output_dir: Path | str, data: dict[str, np.ndarray | float | tuple[int, int]]) -> Path:
    path = ensure_output_dir(output_dir) / "stereo_calib.npz"
    np.savez(path, **data)
    return path


def save_calibration_yaml(output_dir: Path | str, data: dict[str, np.ndarray | float | tuple[int, int]]) -> Path:
    path = ensure_output_dir(output_dir) / "stereo_calib.yaml"
    serializable = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            serializable[key] = value.tolist()
        elif isinstance(value, tuple):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(serializable, fh, sort_keys=False)
    return path


def load_calibration_npz(output_dir: Path | str) -> dict[str, np.ndarray]:
    path = Path(output_dir) / "stereo_calib.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
