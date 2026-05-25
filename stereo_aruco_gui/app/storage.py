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


@dataclass(frozen=True)
class SingleImage:
    index: int
    path: Path


def ensure_image_dirs(root: Path | str) -> tuple[Path, Path]:
    base = Path(root)
    left = base / "left"
    right = base / "right"
    left.mkdir(parents=True, exist_ok=True)
    right.mkdir(parents=True, exist_ok=True)
    return left, right


def ensure_single_image_dir(root: Path | str) -> Path:
    single = Path(root) / "single"
    single.mkdir(parents=True, exist_ok=True)
    return single


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


def list_single_images(root: Path | str) -> list[SingleImage]:
    single_dir = ensure_single_image_dir(root)
    images: list[SingleImage] = []
    for path in sorted(single_dir.glob("*.png")):
        try:
            index = int(path.stem)
        except ValueError:
            index = len(images) + 1
        images.append(SingleImage(index=index, path=path))
    return images


def next_pair_index(root: Path | str) -> int:
    pairs = list_image_pairs(root)
    return max((pair.index for pair in pairs), default=0) + 1


def next_single_image_index(root: Path | str) -> int:
    images = list_single_images(root)
    return max((image.index for image in images), default=0) + 1


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


def save_single_image(root: Path | str, frame: np.ndarray) -> SingleImage:
    single_dir = ensure_single_image_dir(root)
    index = next_single_image_index(root)
    path = single_dir / f"{index:04d}.png"
    if not cv2.imwrite(str(path), frame):
        raise RuntimeError(f"Failed to save {path}")
    return SingleImage(index=index, path=path)


def delete_last_image_pair(root: Path | str) -> ImagePair | None:
    pairs = list_image_pairs(root)
    if not pairs:
        return None

    pair = max(pairs, key=lambda item: item.index)
    pair.left_path.unlink(missing_ok=True)
    pair.right_path.unlink(missing_ok=True)
    return pair


def delete_last_single_image(root: Path | str) -> SingleImage | None:
    images = list_single_images(root)
    if not images:
        return None
    image = max(images, key=lambda item: item.index)
    image.path.unlink(missing_ok=True)
    return image


def ensure_output_dir(root: Path | str) -> Path:
    output_dir = Path(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _calibration_path(output_dir: Path | str, stem: str, suffix: str) -> Path:
    return ensure_output_dir(output_dir) / f"{stem}.{suffix}"


def save_calibration_npz(
    output_dir: Path | str,
    data: dict[str, np.ndarray | float | str | tuple[int, int]],
    stem: str = "stereo_calib",
) -> Path:
    path = _calibration_path(output_dir, stem, "npz")
    np.savez(path, **data)
    return path


def save_calibration_yaml(
    output_dir: Path | str,
    data: dict[str, np.ndarray | float | str | tuple[int, int]],
    stem: str = "stereo_calib",
) -> Path:
    path = _calibration_path(output_dir, stem, "yaml")
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


def load_mono_calibration_npz(output_dir: Path | str) -> dict[str, np.ndarray]:
    path = Path(output_dir) / "mono_calib.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path) as data:
        return {key: data[key] for key in data.files}
