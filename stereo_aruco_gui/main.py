from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from stereo_aruco_gui.app.config import load_config
from stereo_aruco_gui.app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow(load_config())
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
