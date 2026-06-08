from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from decoder_debugger.main_window import DecoderDebuggerWindow


def main(argv: list[str] | None = None) -> int:
    app = QApplication(sys.argv if argv is None else argv)
    window = DecoderDebuggerWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
