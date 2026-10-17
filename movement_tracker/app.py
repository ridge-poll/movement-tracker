import argparse
import sys
from pathlib import Path
from PySide6.QtWidgets import QApplication
from movement_tracker.models import DEFAULT_MODEL
from movement_tracker.ui.main_window import MainWindow

def main():
    parser = argparse.ArgumentParser(description="Movement Tracker desktop app")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("Movement Tracker")
    window = MainWindow(args.model, args.camera)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
