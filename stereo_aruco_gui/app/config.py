from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


CONFIG_PATH = Path("config.yaml")


@dataclass
class CameraConfig:
    left_index: int = 0
    right_index: int = 1
    width: int = 1280
    height: int = 960
    fps: int = 30
    single_mode: bool = False
    backend: str = "AUTO"
    pixel_format: str = "MJPG"


@dataclass
class ArucoConfig:
    dictionary: str = "DICT_4X4_50"
    markers_x: int = 5
    markers_y: int = 7
    marker_length_m: float = 0.03
    marker_separation_m: float = 0.01
    board_mirror_x: bool = False


@dataclass
class CaptureConfig:
    output_dir: str = "data/images"


@dataclass
class OutputConfig:
    dir: str = "data/output"


@dataclass
class AppConfig:
    camera: CameraConfig = field(default_factory=CameraConfig)
    aruco: ArucoConfig = field(default_factory=ArucoConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)
    output: OutputConfig = field(default_factory=OutputConfig)


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    return value if isinstance(value, dict) else {}


def load_config(path: Path | str = CONFIG_PATH) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        return AppConfig()

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}

    camera = _section(data, "camera")
    aruco = _section(data, "aruco")
    capture = _section(data, "capture")
    output = _section(data, "output")

    return AppConfig(
        camera=CameraConfig(
            left_index=int(camera.get("left_index", 0)),
            right_index=int(camera.get("right_index", 1)),
            width=int(camera.get("width", 1280)),
            height=int(camera.get("height", 960)),
            fps=int(camera.get("fps", 30)),
            single_mode=bool(camera.get("single_mode", False)),
            backend=str(camera.get("backend", "AUTO")),
            pixel_format=str(camera.get("pixel_format", "MJPG")),
        ),
        aruco=ArucoConfig(
            dictionary=str(aruco.get("dictionary", "DICT_4X4_50")),
            markers_x=int(aruco.get("markers_x", 5)),
            markers_y=int(aruco.get("markers_y", 7)),
            marker_length_m=float(aruco.get("marker_length_m", 0.03)),
            marker_separation_m=float(aruco.get("marker_separation_m", 0.01)),
            board_mirror_x=bool(aruco.get("board_mirror_x", False)),
        ),
        capture=CaptureConfig(output_dir=str(capture.get("output_dir", "data/images"))),
        output=OutputConfig(dir=str(output.get("dir", "data/output"))),
    )


def save_config(config: AppConfig, path: Path | str = CONFIG_PATH) -> None:
    data = {
        "camera": vars(config.camera),
        "aruco": vars(config.aruco),
        "capture": vars(config.capture),
        "output": vars(config.output),
    }
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False)
